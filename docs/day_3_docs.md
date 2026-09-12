The goal of static batching is to serve multiple requests at once. How this works is we wait until all N requests have reached the queue, and lock the GPU, and only unlock until all requests have finished processing.

At first I thought we should initialize N different model objects, but this results in the model weights loaded in N times, and can't have true parallelism.

So instead, batch process in the model level. input_ids should take in a tensor of input tokens for each request!

In continuous batching's prefill priortization step, there are two ways. One, we could batch the prefill into 1 tensor, with the size being the longest prompt and padding the smaller prompts. This means prefill will happen in parallel. However, with high variance in prompt length, there will be a lot of wasted padding. The second approach is to sequentially prefill, meaning no padding. However, this means slightly more TTFT on average but with no wasted padding. 