// Paged decode attention: one query token per row, each row against its own pages.
//
// Shapes (all contiguous):
//   q            [B, Hq, D]                 bf16, post-RoPE
//   k_pool       [num_blocks, Hkv, BS, D]   bf16
//   v_pool       [num_blocks, Hkv, BS, D]   bf16
//   block_table  [B, W]                     int32, physical block of each logical block
//   seq_lens     [B]                        int32, slots written including this step's
//   out          [B, Hq, D]                 bf16
//
// One thread block owns (kv head h, row b, partition p) and serves all G query
// heads that share kv head h. That is the GQA saving made physical: every K and
// V byte is loaded once and used G = 4 times, where a kernel per query head
// would load it four times and give back most of what paging saved.
//
// Within a thread block, 16 lanes cover one token's 128-dim head with one 16-byte
// load each, so a warp reads two tokens per load and walks one 16-token page -- a
// contiguous 4 KB tile -- in eight loads that are all issued before any of them
// is used.
//
// Split-K (flash-decoding): with partition_size P > 0 the row is cut into
// ceil(W*BS / P) partitions along the grid's z axis. Each writes an unnormalised
// partial output plus its running max and sum, and a second kernel rescales and
// combines them. Partitions that start past a row's length exit immediately, so
// the grid is sized by the table width (a host-known constant) while the work
// is sized by each row's real length (known only on the device). That is what
// keeps the launch shape fixed for CUDA graphs without reading padding.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace socrates {

constexpr int kWarp = 32;

__device__ __forceinline__ void unpack8(const uint4& raw, float (&f)[8]) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 v = __bfloat1622float2(h[i]);
    f[2 * i] = v.x;
    f[2 * i + 1] = v.y;
  }
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = kWarp / 2; off > 0; off /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = kWarp / 2; off > 0; off /= 2) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

template <int D, int BS, int G, int NT>
__global__ void __launch_bounds__(NT) paged_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ out,      // single pass: final output
    float* __restrict__ part_o,           // split: [B, Hq, NP, D], unnormalised
    float* __restrict__ part_m,           // split: [B, Hq, NP], running max
    float* __restrict__ part_l,           // split: [B, Hq, NP], running sum
    int Hkv, int W, int part_len, int NP, float scale) {
  constexpr int VEC = 8;                    // bf16 per 16-byte load
  constexpr int LPT = D / VEC;              // lanes per token: 16
  constexpr int TPW = kWarp / LPT;          // tokens per warp per load: 2
  constexpr int NWARPS = NT / kWarp;        // 4
  constexpr int NGROUPS = NT / LPT;         // token groups in the block: 8
  constexpr int ITERS = BS / TPW;           // loads per page per thread: 8
  static_assert(D % VEC == 0 && kWarp % LPT == 0 && BS % TPW == 0, "tile shape");

  const int h = blockIdx.x, b = blockIdx.y, p = blockIdx.z;
  const int tid = threadIdx.x, warp = tid / kWarp, lane = tid % kWarp;
  const int sub = lane / LPT;               // which of the warp's two tokens
  const int l = lane % LPT;                 // which 8-dim slice of the head
  const int Hq = Hkv * G;
  const bool split = part_o != nullptr;

  // Clamped so a bad length can never walk off the end of the table.
  const int seq_len = min(seq_lens[b], W * BS);
  const int start = p * part_len;
  const int end = min(start + part_len, seq_len);

  if (start >= end) {
    // Nothing of this row lives in this partition. The reduce never reads it --
    // it stops at ceil(seq_len / part_len), the same bound -- so part_o is left
    // uninitialised on purpose; m and l are set only so a debugger sees sense.
    if (split) {
      if (tid < G) {
        const size_t i = ((size_t)b * Hq + h * G + tid) * NP + p;
        part_m[i] = -INFINITY;
        part_l[i] = 0.f;
      }
    } else {
      for (int i = tid; i < G * D; i += NT)
        out[((size_t)b * Hq + h * G) * D + i] = __float2bfloat16(0.f);
    }
    return;
  }

  extern __shared__ float smem[];           // logits [G][part_len], later the [NGROUPS][G][D] reduce
  __shared__ float red[NWARPS][G];
  __shared__ float stat_m[G], stat_l[G];
  float* logits = smem;

  // This lane's slice of all G query heads, pre-scaled, held in registers for
  // the whole K pass: 4 heads x 8 dims = 32 floats.
  float qr[G][VEC];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(
        q + ((size_t)b * Hq + h * G + g) * D + l * VEC));
    unpack8(raw, qr[g]);
#pragma unroll
    for (int j = 0; j < VEC; ++j) qr[g][j] *= scale;
  }

  const int* bt = block_table + (size_t)b * W;
  const int first_page = start / BS;        // part_len is a multiple of BS
  const int last_page = (end + BS - 1) / BS;
  const size_t head_off = (size_t)h * BS * D;
  const size_t page_stride = (size_t)Hkv * BS * D;

  // ---- K pass: one page per warp, logits into shared memory ----------------
  for (int page = first_page + warp; page < last_page; page += NWARPS) {
    const __nv_bfloat16* kp = k_pool + (size_t)bt[page] * page_stride + head_off + l * VEC;
    uint4 raw[ITERS];
#pragma unroll
    for (int i = 0; i < ITERS; ++i)
      raw[i] = __ldg(reinterpret_cast<const uint4*>(kp + (i * TPW + sub) * D));
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
      float kf[VEC];
      unpack8(raw[i], kf);
      float s[G];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        s[g] = 0.f;
#pragma unroll
        for (int j = 0; j < VEC; ++j) s[g] = fmaf(qr[g][j], kf[j], s[g]);
        // the 16 lanes of one token; xor below 16 never crosses into the other
#pragma unroll
        for (int off = LPT / 2; off > 0; off /= 2)
          s[g] += __shfl_xor_sync(0xffffffffu, s[g], off);
      }
      const int t = page * BS + i * TPW + sub;
      if (l == 0 && t < end) {
#pragma unroll
        for (int g = 0; g < G; ++g) logits[g * part_len + (t - start)] = s[g];
      }
    }
  }
  __syncthreads();

  // ---- softmax over this partition, per query head --------------------------
  const int n = end - start;
  float mx[G];
#pragma unroll
  for (int g = 0; g < G; ++g) mx[g] = -INFINITY;
  for (int t = tid; t < n; t += NT) {
#pragma unroll
    for (int g = 0; g < G; ++g) mx[g] = fmaxf(mx[g], logits[g * part_len + t]);
  }
#pragma unroll
  for (int g = 0; g < G; ++g) mx[g] = warp_max(mx[g]);
  if (lane == 0) {
#pragma unroll
    for (int g = 0; g < G; ++g) red[warp][g] = mx[g];
  }
  __syncthreads();
  if (tid < G) {
    float m = red[0][tid];
#pragma unroll
    for (int w = 1; w < NWARPS; ++w) m = fmaxf(m, red[w][tid]);
    stat_m[tid] = m;
  }
  __syncthreads();

  float sm[G];
#pragma unroll
  for (int g = 0; g < G; ++g) sm[g] = 0.f;
  for (int t = tid; t < n; t += NT) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float e = __expf(logits[g * part_len + t] - stat_m[g]);
      logits[g * part_len + t] = e;
      sm[g] += e;
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g) sm[g] = warp_sum(sm[g]);
  if (lane == 0) {
#pragma unroll
    for (int g = 0; g < G; ++g) red[warp][g] = sm[g];
  }
  __syncthreads();                          // also publishes the exp'd logits
  if (tid < G) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) s += red[w][tid];
    stat_l[tid] = s;
  }

  // ---- V pass: same page walk, fp32 accumulators for all G heads ------------
  float acc[G][VEC];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[g][j] = 0.f;

  for (int page = first_page + warp; page < last_page; page += NWARPS) {
    const __nv_bfloat16* vp = v_pool + (size_t)bt[page] * page_stride + head_off + l * VEC;
    uint4 raw[ITERS];
#pragma unroll
    for (int i = 0; i < ITERS; ++i)
      raw[i] = __ldg(reinterpret_cast<const uint4*>(vp + (i * TPW + sub) * D));
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
      const int t = page * BS + i * TPW + sub;
      // Skip rather than multiply by zero: a slot past the row's end holds
      // whatever the last owner of that page left there.
      if (t < end) {
        float vf[VEC];
        unpack8(raw[i], vf);
#pragma unroll
        for (int g = 0; g < G; ++g) {
          const float pr = logits[g * part_len + (t - start)];
#pragma unroll
          for (int j = 0; j < VEC; ++j) acc[g][j] = fmaf(pr, vf[j], acc[g][j]);
        }
      }
    }
  }
  __syncthreads();                          // logits are dead; reuse smem

  // ---- combine the 8 token groups, then write out ---------------------------
  const int grp = warp * TPW + sub;
  float* red_o = smem;                      // [NGROUPS][G][D]
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int j = 0; j < VEC; ++j) red_o[(grp * G + g) * D + l * VEC + j] = acc[g][j];
  __syncthreads();

  for (int i = tid; i < G * D; i += NT) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < NGROUPS; ++k) s += red_o[k * G * D + i];
    const int g = i / D, d = i % D;
    const size_t hq = (size_t)b * Hq + h * G + g;
    if (split)
      part_o[(hq * NP + p) * D + d] = s;
    else
      out[hq * D + d] = __float2bfloat16(s / stat_l[g]);
  }
  if (split && tid < G) {
    const size_t i = ((size_t)b * Hq + h * G + tid) * NP + p;
    part_m[i] = stat_m[tid];
    part_l[i] = stat_l[tid];
  }
}

// One thread block per (query head, row), one thread per output dim. Rescales
// each partition's partial by exp(m_p - M) and divides by the rescaled sum.
template <int D>
__global__ void __launch_bounds__(D) paged_reduce_kernel(
    const float* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ out, int Hq, int max_len, int part_len, int NP) {
  const int hq = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
  const int seq_len = min(seq_lens[b], max_len);
  const int np = min(NP, (seq_len + part_len - 1) / part_len);
  const size_t base = ((size_t)b * Hq + hq) * NP;

  float M = -INFINITY;
  for (int p = 0; p < np; ++p) M = fmaxf(M, part_m[base + p]);
  float L = 0.f, o = 0.f;
  for (int p = 0; p < np; ++p) {
    const float w = __expf(part_m[base + p] - M);
    L = fmaf(w, part_l[base + p], L);
    o = fmaf(w, part_o[(base + p) * D + d], o);
  }
  out[((size_t)b * Hq + hq) * D + d] = __float2bfloat16(np > 0 ? o / L : 0.f);
}

namespace {

constexpr int kD = 128, kBS = 16, kNT = 128;
// The default per-block limit with no opt-in attribute, less room for the
// kernel's static arrays (red, stat_m, stat_l), which count against the same 48 KB.
constexpr int kMaxSmem = 47 * 1024;

template <int G>
void launch(const at::Tensor& q, const at::Tensor& k_pool, const at::Tensor& v_pool,
            const at::Tensor& bt, const at::Tensor& sl, at::Tensor& out, float scale,
            int part_len, int NP, cudaStream_t stream) {
  const int B = q.size(0), Hq = q.size(1), Hkv = k_pool.size(1), W = bt.size(1);
  constexpr int NGROUPS = kNT / (kD / 8);
  const size_t smem = sizeof(float) * static_cast<size_t>(
      std::max(G * part_len, NGROUPS * G * kD));
  TORCH_CHECK(smem <= static_cast<size_t>(kMaxSmem), "paged_attention_decode: partition of ",
              part_len, " tokens needs ", smem, " B of shared memory; use a smaller partition_size");

  const dim3 grid(Hkv, B, NP);
  const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  const auto* kp = reinterpret_cast<const __nv_bfloat16*>(k_pool.data_ptr());
  const auto* vp = reinterpret_cast<const __nv_bfloat16*>(v_pool.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const int* btp = bt.data_ptr<int>();
  const int* slp = sl.data_ptr<int>();

  if (NP == 1) {
    paged_decode_kernel<kD, kBS, G, kNT><<<grid, kNT, smem, stream>>>(
        qp, kp, vp, btp, slp, op, nullptr, nullptr, nullptr, Hkv, W, part_len, 1, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  // Scratch from the caching allocator: on the current stream, and inside a
  // captured graph it comes from the graph's private pool, so replay is safe.
  const auto f32 = q.options().dtype(at::kFloat);
  at::Tensor part_o = at::empty({B, Hq, NP, kD}, f32);
  at::Tensor part_m = at::empty({B, Hq, NP}, f32);
  at::Tensor part_l = at::empty({B, Hq, NP}, f32);
  paged_decode_kernel<kD, kBS, G, kNT><<<grid, kNT, smem, stream>>>(
      qp, kp, vp, btp, slp, nullptr, part_o.data_ptr<float>(), part_m.data_ptr<float>(),
      part_l.data_ptr<float>(), Hkv, W, part_len, NP, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  paged_reduce_kernel<kD><<<dim3(Hq, B), kD, 0, stream>>>(
      part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(), slp, op,
      Hq, W * kBS, part_len, NP);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor paged_attention_decode_cuda(const at::Tensor& q_, const at::Tensor& k_pool,
                                       const at::Tensor& v_pool, const at::Tensor& block_table,
                                       const at::Tensor& seq_lens, double scale,
                                       int64_t partition_size) {
  const at::cuda::OptionalCUDAGuard guard(q_.device());
  // q, the table and the lengths are small, so make them contiguous if they are
  // not; the pools are gigabytes and must already be (checked in ops.cpp). q is
  // read with 16-byte loads, so a view at an odd offset gets its own copy.
  at::Tensor q = q_.contiguous();
  if (reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 != 0) q = q.clone();
  const at::Tensor bt = block_table.contiguous();
  const at::Tensor sl = seq_lens.contiguous();

  const int B = q.size(0), Hq = q.size(1), Hkv = k_pool.size(1);
  const int G = Hq / Hkv;
  const int max_len = bt.size(1) * kBS;
  at::Tensor out = at::empty_like(q);
  if (B == 0) return out;

  // One pass when no partition is asked for or one would cover the whole table.
  int part_len = max_len, NP = 1;
  if (partition_size > 0 && partition_size < max_len) {
    TORCH_CHECK(partition_size % kBS == 0, "partition_size must be a multiple of ", kBS);
    part_len = static_cast<int>(partition_size);
    NP = (max_len + part_len - 1) / part_len;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float s = static_cast<float>(scale);
  switch (G) {
    case 1: launch<1>(q, k_pool, v_pool, bt, sl, out, s, part_len, NP, stream); break;
    case 2: launch<2>(q, k_pool, v_pool, bt, sl, out, s, part_len, NP, stream); break;
    case 4: launch<4>(q, k_pool, v_pool, bt, sl, out, s, part_len, NP, stream); break;
    case 8: launch<8>(q, k_pool, v_pool, bt, sl, out, s, part_len, NP, stream); break;
    default: TORCH_CHECK(false, "paged_attention_decode: GQA group ", G, " not compiled");
  }
  return out;
}

}  // namespace socrates
