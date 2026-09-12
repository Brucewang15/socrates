The goal of static batching is to serve multiple requests at once. How this works is we wait until all N requests have reached the queue, and lock the GPU, and only unlock until all requests have finished processing.

At first I thought we should initialize N different model objects, but this results in the model weights loaded in N times, and can't have true parallelism.

So instead, batch process in the model level. input_ids should take in a tensor of input tokens for each request!