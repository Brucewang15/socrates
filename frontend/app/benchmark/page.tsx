"use client";

import Link from "next/link";
import { useState } from "react";
import { Distribution, Occupancy, Timeline } from "./charts";
import type { Result } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const TILES = [
  { key: "throughput_tps", label: "throughput", unit: " tok/s", dp: 1,
    note: "total output tokens over wall time" },
  { key: "ttft_p99_s", label: "TTFT p99", unit: "s", dp: 2,
    note: "queue wait plus prefill, worst case" },
  { key: "itl_p50_s", label: "ITL p50", unit: "s", dp: 3,
    note: "seconds between tokens once generating" },
  { key: "occupancy", label: "occupancy", unit: "", dp: 2,
    note: "mean rows generating over MAX_BATCH" },
] as const;

export default function Benchmark() {
  const [data, setData] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/benchmark`, { method: "POST" });
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      setData(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : `could not reach ${API_URL}`);
    } finally {
      setBusy(false);
    }
  }

  const ctx = data?.context;

  return (
    <main className="bench">
      <header className="bench-head">
        <div>
          <h1>Benchmark</h1>
          <p className="muted">
            16 prompts submitted at once, through {ctx?.max_batch ?? "N"} rows.
          </p>
        </div>
        <div className="bench-actions">
          <Link href="/" className="ghost">← Chat</Link>
          <button onClick={run} disabled={busy}>
            {busy ? "Running…" : "Run benchmark"}
          </button>
        </div>
      </header>

      {busy && (
        <p className="muted">
          Generating on {ctx?.device ?? "the model tier"} — this takes a few minutes.
        </p>
      )}
      {error && <p className="error">{error}</p>}

      {data && (
        <>
          <section className="tiles">
            {TILES.map((t) => (
              <div key={t.key} className="tile">
                <span className="tile-label">{t.label}</span>
                <span className="tile-value">
                  {data.headline[t.key].toFixed(t.dp)}
                  <small>{t.unit}</small>
                </span>
                <span className="tile-note">{t.note}</span>
              </div>
            ))}
          </section>

          <p className="muted">
            {data.context.device} · MAX_BATCH {data.context.max_batch} ·{" "}
            {data.context.prompts} prompts · {data.context.output_tokens} tokens ·{" "}
            {data.context.wall_s.toFixed(1)}s wall
          </p>

          <section className="panel">
            <h2>Timeline</h2>
            <p className="muted">
              Each bar starts when the request was submitted. Grey is waiting for a
              row — the only part the scheduler owns.
            </p>
            <Timeline requests={data.requests} />
          </section>

          <section className="panel">
            <h2>Occupancy</h2>
            <p className="muted">
              Rows generating against rows held. The gap is capacity paid for and
              not used.
            </p>
            <Occupancy data={data.occupancy} maxBatch={data.context.max_batch} />
          </section>

          <div className="grid-2">
            <section className="panel">
              <h2>TTFT</h2>
              <Distribution requests={data.requests} field="ttft_s" unit="s" label="seconds" />
            </section>
            <section className="panel">
              <h2>ITL</h2>
              <Distribution requests={data.requests} field="itl_s" unit="s" label="s / token" />
            </section>
          </div>

          <section className="panel">
            <h2>Percentiles</h2>
            <table>
              <thead>
                <tr><th>metric</th><th>p50</th><th>p95</th><th>p99</th></tr>
              </thead>
              <tbody>
                {Object.entries(data.percentiles).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k.replace("_s", "")}</td>
                    <td>{v.p50.toFixed(3)}s</td>
                    <td>{v.p95.toFixed(3)}s</td>
                    <td>{v.p99.toFixed(3)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="panel">
            <h2>Requests</h2>
            <table>
              <thead>
                <tr>
                  <th>prompt</th><th>in</th><th>out</th><th>queue</th>
                  <th>TTFT</th><th>ITL</th><th>total</th>
                </tr>
              </thead>
              <tbody>
                {data.requests.map((r) => (
                  <tr key={r.i}>
                    <td className="prompt">{r.prompt}</td>
                    <td>{r.prompt_tokens}</td>
                    <td>{r.output_tokens}</td>
                    <td>{r.queue_s.toFixed(2)}s</td>
                    <td>{r.ttft_s.toFixed(2)}s</td>
                    <td>{r.itl_s.toFixed(3)}s</td>
                    <td>{r.total_s.toFixed(2)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </main>
  );
}
