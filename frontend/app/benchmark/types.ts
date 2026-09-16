export type Bucket = "short" | "medium" | "long";

export type Req = {
  i: number;
  bucket: Bucket;
  prompt: string;
  prompt_tokens: number;
  output_tokens: number;
  sent: number;
  started: number;
  first_token: number;
  finished: number;
  queue_s: number;
  prefill_s: number;
  decode_s: number;
  total_s: number;
  itl_s: number;
  ttft_s: number;
};

export type Spread = { p50: number; p90: number; p95: number };

/** One throughput bin: tokens that actually arrived inside it. */
export type Tick = {
  t: number;
  width_s: number;
  tokens: number;
  tokens_s: number;
  rows: number;
};

export type Result = {
  context: {
    device: string;
    max_batch: number;
    prompts: number;
    rate: number;
    seed: number;
    output_tokens: number;
    wall_s: number;
    itl_sample: number;
    bin_s: number;
    bins: number;
    transport: string;
  };
  headline: {
    throughput_tps: number;
    throughput_peak_tps: number;
    throughput_p50_tps: number;
    per_stream_tps: number;
    ttft_p95_s: number;
    itl_p50_s: number;
    occupancy: number;
  };
  percentiles: { ttft_s: Spread; itl_s: Spread; latency_s: Spread };
  requests: Req[];
  occupancy: { t: number; generating: number; held: number; queued: number }[];
  throughput: Tick[];
};
