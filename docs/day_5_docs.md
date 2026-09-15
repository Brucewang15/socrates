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


Monitoring:

Prometheus: scrapes a configured endpoint every N seconds. The endpoint returns flat data and prometheus stores time-series data. However, prometheus might miss data in between the scrapes.

If the app's memory is flat, where does the time series come from? Prometheus builds it. The app only stores one number, no history. Prometheus snapshots the odometer every 15s and stamps each reading with a timestamp, so the sequence lives in Prometheus, rate(output_tokens_total[30s]) is just (402-190)/30; we only ever said "402".

Events: a discrete and countable event. A request finished, a token was generated, an error was raised. Contrast with state, which is a condition right now. State has a value at every instant, including between scrapes.

Counter: a number that only goes up, incremented by our code at the moment the event happens. It cannot miss an event because the increment is inside the request handler -- the scrape doesn't detect anything, it reads a total that was already accumulated. A turnstile can't miss a person; the person turns it. Counters must be monotonic so rate() can differentiate them, and so a restart (counter drops to 0) is detectable rather than looking like a negative rate.

Gauge: current state, read at scrape time. Ours use set_function, so the lambda runs when Prometheus calls, not when the value changes. This is why gauges CAN miss: if queue_depth hits 50 at t=1.5 and is back to 2 by t=1.6, nobody was looking and there is no trace of it anywhere. Scraping faster doesn't fix this in general -- there is always a spike shorter than the interval. The fix is to convert state into events: requests_total{outcome="queue_full"} cannot miss the overflow even though queue_depth can.

Histogram: n_buckets + 2 floats. One counter per bucket, plus _sum and _count. Buckets are cumulative, so observe(0.66) increments every bucket at or above 0.66. That's how a 3.1s request stays visible forever without being stored individually. Percentiles are interpolated from bucket rates at query time, over a window -- there is no global p95, only p95 over the last 5 minutes. Use Histogram not Summary, because summaries compute quantiles per-instance and can't be aggregated across replicas.

Why memory is flat: the registry stores one current value per series, not one row per observation. A counter incremented a billion times is still one float. So app memory is O(series), not O(observations) or O(time) -- ours is ~67 series, a few KB, constant forever. The only way it grows is cardinality: every distinct label value is a new series. outcome/route/status are fine (bounded); user_id or prompt text is not (unbounded, and it blows up Prometheus too -- 100k users would be ~400k series, ~69 GB of disk). Per-request detail belongs in logs, not metrics.

History lives in Prometheus, bounded by --storage.tsdb.retention.time=15d. 67 series at 15s scrapes is ~12 MB on disk.

Instrumentation overhead: measured 2.91 us per request for our 5 observes + 3 counter incs. Against a 1.76s request that's 0.0002%. Only matters if you instrument a hot loop (per decode step would be 36 layers x N tokens) or use unbounded labels.

Grafana: frontend for reading prometheus. Stores dashboards and users, no metric data -- it fires PromQL at Prometheus on every panel refresh.


Running it locally:

    make dev                        # model :8080, backend :8000, frontend :3000
    curl localhost:8080/metrics     # prometheus text format
    curl localhost:8000/metrics     # edge RED metrics

Prometheus and Grafana are compose services, and dcgm-exporter needs an NVIDIA host, so the full stack only comes up on the GPU box:

    docker compose up -d prometheus grafana     # skips dcgm-exporter, works on a Mac
    open http://localhost:9090                  # prometheus, check Status > Targets
    open http://localhost:3001                  # grafana, dashboard is auto-provisioned
                                                # in prod it is monitoring.socrates.pianofi.ca

Scraping from a container to a host-run uvicorn needs host.docker.internal instead of model:8080 in monitoring/prometheus.yml. On the EC2 box everything is on the compose network so the service names work as written, and neither port is in the security group -- reach them over SSM port forwarding:

    aws ssm start-session --profile bruce-dev --region us-east-1 --target <id> \
      --document-name AWS-StartPortForwardingSession \
      --parameters '{"portNumber":["3001"],"localPortNumber":["3001"]}'
