// The op's schema, its input checks, and its registration with the dispatcher.
//
// Registered with TORCH_LIBRARY rather than bound with pybind11, so the .so has
// no Python in it at all: torch.ops.load_library dlopens it, the static
// initialisers below run, and torch.ops.socrates.paged_attention_decode exists.
// The fake (shape-only) implementation Dynamo traces through lives in Python, in
// model/kernels/__init__.py, next to the code that loads this.
//
// The schema declares no argument as mutated (no Tensor(a!)), and that is
// accurate: the op only reads the pools and returns a fresh tensor. Inductor
// can therefore treat it as a pure function, and the KV writes before it stay
// ordinary in-place updates of module buffers -- which is what lets the decode
// step be captured as a CUDA graph.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>

namespace socrates {

at::Tensor paged_attention_decode_cuda(const at::Tensor& q, const at::Tensor& k_pool,
                                       const at::Tensor& v_pool, const at::Tensor& block_table,
                                       const at::Tensor& seq_lens, double scale,
                                       int64_t partition_size);

namespace {

at::Tensor paged_attention_decode(const at::Tensor& q, const at::Tensor& k_pool,
                                  const at::Tensor& v_pool, const at::Tensor& block_table,
                                  const at::Tensor& seq_lens, double scale,
                                  int64_t partition_size) {
  // Cheap host-side checks only -- nothing here reads a tensor's values, so the
  // op stays sync-free and capturable.
  TORCH_CHECK(q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda() && block_table.is_cuda() &&
                  seq_lens.is_cuda(),
              "paged_attention_decode: every input must be on CUDA");
  TORCH_CHECK(k_pool.device() == q.device() && v_pool.device() == q.device() &&
                  block_table.device() == q.device() && seq_lens.device() == q.device(),
              "paged_attention_decode: inputs on different devices");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k_pool.scalar_type() == at::kBFloat16 &&
                  v_pool.scalar_type() == at::kBFloat16,
              "paged_attention_decode: q and the pools must be bfloat16");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && seq_lens.scalar_type() == at::kInt,
              "paged_attention_decode: block_table and seq_lens must be int32");

  TORCH_CHECK(q.dim() == 3, "q must be [B, Hq, D], got ", q.sizes());
  TORCH_CHECK(k_pool.dim() == 4, "k_pool must be [num_blocks, Hkv, BS, D], got ", k_pool.sizes());
  TORCH_CHECK(k_pool.sizes() == v_pool.sizes(), "k_pool and v_pool shapes differ");
  TORCH_CHECK(k_pool.is_contiguous() && v_pool.is_contiguous(),
              "the pools must be contiguous (they are never copied)");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == q.size(0),
              "block_table must be [B, W] with B = ", q.size(0), ", got ", block_table.sizes());
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == q.size(0),
              "seq_lens must be [B] with B = ", q.size(0), ", got ", seq_lens.sizes());

  const int64_t Hq = q.size(1), D = q.size(2), Hkv = k_pool.size(1), BS = k_pool.size(2);
  TORCH_CHECK(D == 128 && k_pool.size(3) == 128, "compiled for head_dim 128, got ", D);
  TORCH_CHECK(BS == 16, "compiled for 16-slot blocks, got ", BS);
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq (", Hq, ") must be a multiple of Hkv (", Hkv, ")");
  TORCH_CHECK(partition_size >= 0, "partition_size must be >= 0");
  // 16-byte vector loads need 16-byte aligned bases; every allocator we use gives
  // 512, so this only fires on a strange view.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(k_pool.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(v_pool.data_ptr()) % 16 == 0,
              "pools must be 16-byte aligned");

  return paged_attention_decode_cuda(q, k_pool, v_pool, block_table, seq_lens, scale,
                                     partition_size);
}

}  // namespace

TORCH_LIBRARY(socrates, m) {
  m.def(
      "paged_attention_decode(Tensor q, Tensor k_pool, Tensor v_pool, Tensor block_table, "
      "Tensor seq_lens, float scale, int partition_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(socrates, CUDA, m) {
  m.impl("paged_attention_decode", &paged_attention_decode);
}

}  // namespace socrates
