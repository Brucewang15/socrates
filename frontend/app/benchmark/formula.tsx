"use client";

/**
 * Hover-or-focus explainer for a metric, showing exactly how it was computed.
 *
 * Every number on this screen is derived from per-request timestamps, and most
 * of them can be misread -- aggregate throughput looks wrong next to a single
 * prompt, ITL divides by gaps rather than tokens, occupancy counts rows rather
 * than requests. Rather than a legend nobody reads, the arithmetic sits on the
 * number itself.
 *
 * Keyboard reachable and screen-reader visible: the trigger is a real button
 * with aria-describedby pointing at the tooltip, so it is not hover-only.
 */

import { useId } from "react";

export type FormulaProps = {
  /** One-line plain-language statement of what the number means. */
  what: string;
  /** The actual arithmetic, shown monospaced. */
  formula: string;
  /** Optional caveat -- the thing that makes the number easy to misread. */
  caveat?: string;
};

export function Formula({ what, formula, caveat }: FormulaProps) {
  const id = useId();
  return (
    <span className="formula">
      <button
        type="button"
        className="formula-trigger"
        aria-label="how this is calculated"
        aria-describedby={id}
      >
        ?
      </button>
      <span role="tooltip" id={id} className="formula-body">
        <span className="formula-what">{what}</span>
        <code className="formula-math">{formula}</code>
        {caveat && <span className="formula-caveat">{caveat}</span>}
      </span>
    </span>
  );
}

/** Every metric on the page, with the arithmetic that produced it. */
export const FORMULAS: Record<string, FormulaProps> = {
  throughput_tps: {
    what: "Aggregate tokens per second across every request at once.",
    formula: "sum(output_tokens) / wall_s",
    caveat:
      "wall_s runs from the first dispatch to the last response, after a warmup request. This is the whole batch added together, so it is several times what any single caller sees — compare it to per-stream, not to one prompt.",
  },
  per_stream_tps: {
    what: "What one caller experiences once their tokens start flowing.",
    formula: "1 / itl_p50",
    caveat:
      "Median, not mean: a request that emits one token has no gap to measure and would otherwise report its whole decode span as a single interval, which a mean cannot survive. With N rows sharing a decode step, aggregate throughput is roughly N x this.",
  },
  ttft_p95_s: {
    what: "Time to first token, worst case. Queue wait plus prefill.",
    formula: "p95( queue_s + prefill_s )",
    caveat:
      "queue_s is time spent in the engine's pending deque before a row frees. Percentiles are linearly interpolated over 48 samples, which is enough for p90/p95 to be stable but not p99.",
  },
  itl_p50_s: {
    what: "Median seconds between consecutive tokens, once generating.",
    formula: "p50( decode_s / (output_tokens - 1) )",
    caveat:
      "n tokens have n-1 gaps, and the first token is already counted in TTFT. Requests that emitted fewer than 2 tokens are excluded — they have no gap to measure.",
  },
  occupancy: {
    what: "How full the batch was, averaged over the run.",
    formula: "mean(rows generating) / MAX_BATCH",
    caveat:
      "Sampled at 120 points across wall_s. It falls below 1 because the queue drains at the end, when only the longest requests are still running.",
  },
  queue_s: {
    what: "Waiting for a row to free up. The only part the scheduler controls.",
    formula: "admitted - submitted",
  },
  prefill_s: {
    what: "Reading the prompt, up to the first generated token.",
    formula: "first_token - admitted",
  },
  decode_s: {
    what: "Generating every token after the first.",
    formula: "finished - first_token",
  },
  total_s: {
    what: "End to end, as the caller experiences it.",
    formula: "finished - submitted  ( = queue + prefill + decode )",
  },
  ttft_s: {
    what: "Time to first token for this request.",
    formula: "queue_s + prefill_s",
  },
  itl_s: {
    what: "Mean seconds between tokens for this request.",
    formula: "decode_s / (output_tokens - 1)",
  },
  latency_s: {
    what: "End-to-end time per request.",
    formula: "finished - submitted",
  },
  bucket: {
    what: "Expected output length, sixteen prompts each.",
    formula: "short | medium | long",
    caveat:
      "An expectation, not a guarantee — the model decides when to stop. The spread is deliberate: mixed lengths are what separate continuous batching from static.",
  },
};
