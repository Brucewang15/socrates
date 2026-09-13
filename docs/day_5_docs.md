There are 3 things that bound how fast inference is. Compute bound, meaning better gpus, better kernels. Memory bound, meaning the GPU spends its time moving bytes rather than doing math, every decode step has to read all 8.04 GB of Qwen3-4B's weights out of HBM just to produce one token, so the step takes (bytes / bandwidth) seconds no matter how little arithmetic it does. On an A10G at 600 GB/s that's a 13.4 ms floor per step, and nothing above the memory system can beat it. This is also why batching is nearly free: batch 8 reads the same 8.04 GB and gets 8 tokens out of it. And overhead bound, meaning the GPU is idle waiting on the CPU, too many kernel launches, and too many GPU<->CPU 'syncs'.

A sync is the thing I kept not understanding, so: GPU calls are queued, not executed. When you write `x @ w`, PyTorch does not compute anything. It appends a kernel to a stream and returns in a few microseconds. The CPU then races ahead queueing more work, staying dozens of kernels in front, and the GPU chews through the queue behind it. That lead is the whole reason GPUs are fast in practice.

A sync is anything that makes the CPU ask for an actual *value*. `int(t)`, `.item()`, `.tolist()`, `float(t)`, `print(t)`, `if t > 0`. The value doesn't exist until the kernel that computes it has run, so the CPU has to stop and wait for the entire queue to drain first. Two costs, and the second one is the one I missed: you wait for everything already queued, and then the queue is *empty*, so the GPU sits idle while the CPU walks back through Python refilling it. You don't just lose the wait, you lose the lead.

On my Mac this barely matters because Apple Silicon has unified memory -- CPU and GPU share the same physical RAM, so reading a scalar is close to a normal load. On CUDA it's a PCIe round trip plus a full pipeline drain. That's why the g5 was *slower* than the laptop: the code is sync-bound, and mps hides syncs while CUDA punishes them.

Ours are in qwen_kv_cont.py, inside KVCache.append:

    end = int(positions.max()) + 1                  # 1 sync
    self.lengths[row] = int(positions[i, -1]) + 1   # 1 sync per row

append runs once per layer, so 36 layers x (1 + n_rows) is about 180 syncs per token. And the joke is that `positions` was built in decode_step *from* self.cache.lengths, which is a plain Python list -- we push host integers to the GPU and then stall 180 times reading them back.

Throughput = tok/s. But per-step cost grows as generation goes on, because the KV cache grows by one token per row per step and every step re-reads all of it (144 KB per token of context for this model). So step 500 is more expensive than step 5, and any single tok/s number is an average over a rising curve. Which tok/s also depends on what you divide by: output_tokens/decode_s excludes prefill, output_tokens/total_s includes it. They only agree when the prompt is short.

TTFT = time to first token = queue wait + prefill. Not just prefill -- if the batch is full the request sits in the deque first, and that wait is the scheduler's fault while prefill is the model's. That's why /generate returns queue_s and prefill_s separately instead of one TTFT number.

Currently our tps is 11 tokens/s.
