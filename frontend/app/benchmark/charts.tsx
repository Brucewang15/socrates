"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useSeries } from "./palette";
import type { Req, Result } from "./types";

const AXIS = { fontSize: 11, fill: "var(--text-secondary)" };
const GRID = "var(--hairline)";

function tip(unit: string) {
  return {
    contentStyle: {
      background: "var(--surface-2)",
      border: "1px solid var(--hairline)",
      borderRadius: 8,
      fontSize: 12,
      color: "var(--text-primary)",
    },
    formatter: (v: unknown, n: unknown) =>
      [typeof v === "number" ? `${v.toFixed(3)}${unit}` : String(v), String(n)] as [string, string],
  };
}

export function Timeline({ requests }: { requests: Req[] }) {
  const c = useSeries();
  // sent is the offset from run start; a transparent spacer puts each bar there
  const data = requests.map((r) => ({
    name: r.prompt.length > 26 ? r.prompt.slice(0, 26) + "…" : r.prompt,
    offset: r.sent,
    queue: r.queue_s,
    prefill: r.prefill_s,
    decode: r.decode_s,
  }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(240, data.length * 26)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 8 }}>
        <CartesianGrid stroke={GRID} horizontal={false} />
        <XAxis type="number" tick={AXIS} stroke={GRID}
               label={{ value: "seconds since run start", position: "insideBottom",
                        offset: -4, style: AXIS }} />
        <YAxis type="category" dataKey="name" width={190} tick={AXIS} stroke={GRID} />
        <Tooltip {...tip("s")} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="offset" stackId="a" fill="transparent" legendType="none" />
        <Bar dataKey="queue" stackId="a" fill={c.queue} name="queue" />
        <Bar dataKey="prefill" stackId="a" fill={c.prefill} name="prefill" />
        <Bar dataKey="decode" stackId="a" fill={c.decode} name="decode"
             radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Occupancy({ data, maxBatch }: { data: Result["occupancy"]; maxBatch: number }) {
  const c = useSeries();
  // queued requests hold no row, so they are drawn on their own axis rather than
  // stacked into the row count -- otherwise the total exceeds maxBatch
  const queuedMax = Math.max(...data.map((d) => d.queued), 1);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data} margin={{ left: 8, right: 16, top: 8 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="t" tick={AXIS} stroke={GRID} tickFormatter={(v) => `${v.toFixed(0)}s`} />
        <YAxis yAxisId="rows" domain={[0, maxBatch]} allowDecimals={false} tick={AXIS} stroke={GRID}
               label={{ value: "rows", angle: -90, position: "insideLeft", style: AXIS }} />
        <YAxis yAxisId="q" orientation="right" domain={[0, queuedMax]} allowDecimals={false}
               tick={AXIS} stroke={GRID}
               label={{ value: "queued", angle: 90, position: "insideRight", style: AXIS }} />
        <Tooltip {...tip("")} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Area yAxisId="q" type="stepAfter" dataKey="queued" stroke={c.queue} fill={c.queue}
              fillOpacity={0.18} name="queued (right axis, holds no row)" />
        <Area yAxisId="rows" type="stepAfter" dataKey="held" stroke={c.held} fill={c.held}
              fillOpacity={0.5} name="held (admitted, incl. prefill)" />
        <Area yAxisId="rows" type="stepAfter" dataKey="generating" stroke={c.decode} fill={c.decode}
              fillOpacity={0.75} name="generating" />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function Distribution({
  requests, field, unit, label,
}: { requests: Req[]; field: "ttft_s" | "itl_s"; unit: string; label: string }) {
  const c = useSeries();
  const data = [...requests]
    .sort((a, b) => a[field] - b[field])
    .map((r, i) => ({ rank: i + 1, v: r[field], prompt: r.prompt }));
  const p95 = data[Math.floor((data.length - 1) * 0.95)]?.v ?? 0;
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ left: 8, right: 16, top: 8 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="rank" tick={AXIS} stroke={GRID}
               label={{ value: "requests, sorted", position: "insideBottom",
                        offset: -4, style: AXIS }} />
        <YAxis tick={AXIS} stroke={GRID}
               label={{ value: label, angle: -90, position: "insideLeft", style: AXIS }} />
        <Tooltip {...tip(unit)} />
        <Bar dataKey="v" name={label} radius={[4, 4, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.v >= p95 ? c.prefill : c.decode} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
