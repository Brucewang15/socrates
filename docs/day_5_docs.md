There are 3 things that bound how fast inference is. Compute bound, meaning better gpus, better kernels. Memory bound, idk what this means. And overhead bound, meaning fewer GPU<->CPU 'syncs' whatever that means, GPU<->CPU memory exchanges what else?

Throughput = tok/s. But decoding compute and memory grows linearly, so this tok/s is an average. tok/s does NOT include prefill. 
TTFT = time to first token, which is perfill