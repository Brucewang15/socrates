Without cache, assuming each weight is 2 bytes and n = number tokens in current sequence

Memory in GPU:
weights: 8.04 GB (no matter what)
attention scores: [1, 32, n, n] each attention block has 32 parallel heads, each producing a [n, n] attention pattern. 64 bytes * n^2
MLP intermediate: [2, n, 9728] n MLP intermdiate layers running in parallel, each with 9728 activations. 2 for up and gate
between blocks: [1, 2560, n] vocab size * # of tokens in seq, 5KB * n
Logis: [1, 151936, n] 304KB * n