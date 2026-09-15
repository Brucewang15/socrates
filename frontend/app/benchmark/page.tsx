"use client";

import Link from "next/link";
import { useState } from "react";
import { Distribution, Occupancy, Timeline } from "./charts";
import { FORMULAS, Formula } from "./formula";
import type { Result } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const TILES = [
  { key: "throughput_tps", label: "throughput", unit: " tok/s", dp: 1 },
  { key: "per_stream_tps", label: "per stream", unit: " tok/s", dp: 1 },
  { key: "ttft_p95_s", label: "TTFT p95", unit: "s", dp: 2 },
  { key: "itl_p50_s", label: "ITL p50", unit: "s", dp: 3 },
  { key: "occupancy", label: "occupancy", unit: "", dp: 2 },
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
            {ctx?.prompts ?? 128} prompts arriving at {ctx?.rate ?? 4}/s, through{" "}
            {ctx?.max_batch ?? "N"} rows — short, medium and long mixed
            expected output.
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
                <span className="tile-label">
                  {t.label}
                  <Formula {...FORMULAS[t.key]} />
                </span>
                <span className="tile-value">
                  {data.headline[t.key].toFixed(t.dp)}
                  <small>{t.unit}</small>
                </span>
                <span className="tile-note">{FORMULAS[t.key].formula}</span>
              </div>
            ))}
          </section>

          <p className="muted">
            {data.context.device} · MAX_BATCH {data.context.max_batch} ·{" "}
            {data.context.rate}/s seed {data.context.seed} ·{" "}
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
                <tr><th>metric</th><th>p50</th><th>p90</th><th>p95</th></tr>
              </thead>
              <tbody>
                {Object.entries(data.percentiles).map(([k, v]) => (
                  <tr key={k}>
                    <td>
                      {k.replace("_s", "")}
                      {FORMULAS[k] && <Formula {...FORMULAS[k]} />}
                    </td>
                    <td>{v.p50.toFixed(3)}s</td>
                    <td>{v.p90.toFixed(3)}s</td>
                    <td>{v.p95.toFixed(3)}s</td>
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
                  <th>prompt</th>
                  <th>len<Formula {...FORMULAS.bucket} /></th>
                  <th>in</th><th>out</th>
                  <th>queue<Formula {...FORMULAS.queue_s} /></th>
                  <th>TTFT<Formula {...FORMULAS.ttft_s} /></th>
                  <th>ITL<Formula {...FORMULAS.itl_s} /></th>
                  <th>total<Formula {...FORMULAS.total_s} /></th>
                </tr>
              </thead>
              <tbody>
                {data.requests.map((r) => (
                  <tr key={r.i}>
                    <td className="prompt">{r.prompt}</td>
                    <td><span className={`bucket bucket-${r.bucket}`}>{r.bucket}</span></td>
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
