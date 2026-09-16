"use client";

import { useRef, useState, type ReactNode } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useSeries } from "./palette";
import type { Req, Result, Tick } from "./types";

const AXIS = { fontSize: 11, fill: "var(--text-secondary)" };
const GRID = "var(--hairline)";
const SVG_STYLE_PROPERTIES = [
  "color", "fill", "fill-opacity", "font-family", "font-size", "font-style",
  "font-weight", "opacity", "stroke", "stroke-dasharray", "stroke-linecap",
  "stroke-linejoin", "stroke-opacity", "stroke-width", "text-anchor",
];

type LegendItem = { label: string; color: string };
type DownloadState = "idle" | "saving" | "saved" | "error";

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

/** Make CSS variables and class-based SVG styles survive outside the page. */
function inlineSvgStyles(source: SVGSVGElement, clone: SVGSVGElement) {
  const sourceNodes: Element[] = [source, ...Array.from(source.querySelectorAll("*"))];
  const cloneNodes: Element[] = [clone, ...Array.from(clone.querySelectorAll("*"))];
  sourceNodes.forEach((node, i) => {
    const computed = window.getComputedStyle(node);
    const declarations = SVG_STYLE_PROPERTIES.map(
      (property) => `${property}:${computed.getPropertyValue(property)}`,
    ).join(";");
    cloneNodes[i]?.setAttribute(
      "style",
      `${cloneNodes[i].getAttribute("style") ?? ""};${declarations}`,
    );
  });
}

function ChartFrame({
  children,
  filename,
  title,
  legend,
  xLabel,
}: {
  children: ReactNode;
  filename: string;
  title: string;
  legend: LegendItem[];
  xLabel?: string;
}) {
  const plotRef = useRef<HTMLDivElement>(null);
  const [downloadState, setDownloadState] = useState<DownloadState>("idle");

  async function downloadPng() {
    const source = plotRef.current?.querySelector("svg");
    if (!source) return;

    setDownloadState("saving");
    try {
      const bounds = source.getBoundingClientRect();
      const width = Math.ceil(bounds.width);
      const height = Math.ceil(bounds.height);
      if (!width || !height) throw new Error("chart has no rendered size");

      const clone = source.cloneNode(true) as SVGSVGElement;
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.setAttribute("width", String(width));
      clone.setAttribute("height", String(height));
      clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
      inlineSvgStyles(source, clone);

      const headerHeight = 46;
      const captionHeight = xLabel ? 32 : 10;
      const legendHeight = legend.length * 22 + 18;
      const exportHeight = headerHeight + height + captionHeight + legendHeight;
      const scale = Math.min(2, 4096 / Math.max(width, exportHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.ceil(width * scale);
      canvas.height = Math.ceil(exportHeight * scale);
      const context = canvas.getContext("2d");
      if (!context) throw new Error("browser does not support canvas export");
      context.scale(scale, scale);

      const panel = plotRef.current?.closest(".panel") ?? document.body;
      const panelStyle = window.getComputedStyle(panel);
      context.fillStyle = panelStyle.backgroundColor || "#2a2b32";
      context.fillRect(0, 0, width, exportHeight);
      context.fillStyle = panelStyle.color || "#ececf1";
      context.font = "600 16px ui-sans-serif, system-ui, sans-serif";
      context.fillText(title, 16, 28);

      const serialized = new XMLSerializer().serializeToString(clone);
      const blob = new Blob([serialized], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      try {
        const image = new Image();
        await new Promise<void>((resolve, reject) => {
          image.onload = () => resolve();
          image.onerror = () => reject(new Error("could not render chart image"));
          image.src = url;
        });
        context.drawImage(image, 0, headerHeight, width, height);
      } finally {
        URL.revokeObjectURL(url);
      }

      let y = headerHeight + height;
      if (xLabel) {
        context.fillStyle = "#acacbe";
        context.font = "12px ui-sans-serif, system-ui, sans-serif";
        context.textAlign = "center";
        context.fillText(xLabel, width / 2, y + 20);
        context.textAlign = "left";
      }
      y += captionHeight + 15;
      context.font = "12px ui-sans-serif, system-ui, sans-serif";
      for (const item of legend) {
        context.fillStyle = item.color;
        context.fillRect(16, y - 10, 12, 12);
        context.fillStyle = "#acacbe";
        context.fillText(item.label, 36, y);
        y += 22;
      }

      const png = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (result) => result ? resolve(result) : reject(new Error("PNG encoding failed")),
          "image/png",
        );
      });
      const pngUrl = URL.createObjectURL(png);
      const anchor = document.createElement("a");
      anchor.href = pngUrl;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(pngUrl);
      setDownloadState("saved");
    } catch (error) {
      console.error("chart export failed", error);
      setDownloadState("error");
    }
  }

  const downloadLabel = {
    idle: "Download PNG",
    saving: "Saving…",
    saved: "Downloaded ✓",
    error: "Try download again",
  }[downloadState];

  return (
    <div className="chart-frame">
      <div className="chart-toolbar">
        <div className="chart-legend" aria-label="Chart legend">
          {legend.map((item) => (
            <span className="chart-legend-item" key={item.label}>
              <span className="chart-swatch" style={{ backgroundColor: item.color }} />
              {item.label}
            </span>
          ))}
        </div>
        <button
          type="button"
          className="chart-download"
          onClick={downloadPng}
          disabled={downloadState === "saving"}
          aria-live="polite"
        >
          <span aria-hidden="true">↓</span> {downloadLabel}
        </button>
      </div>
      <div className="chart-plot" ref={plotRef}>{children}</div>
      {xLabel && <div className="chart-axis-caption">{xLabel}</div>}
    </div>
  );
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
    <ChartFrame
      filename="socrates-benchmark-timeline.png"
      title="Benchmark request timeline"
      xLabel="seconds since run start"
      legend={[
        { label: "queued", color: c.queue },
        { label: "prefill", color: c.prefill },
        { label: "decode", color: c.decode },
      ]}
    >
      <ResponsiveContainer width="100%" height={Math.max(260, data.length * 28)}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 8, bottom: 8 }}>
          <CartesianGrid stroke={GRID} horizontal={false} />
          <XAxis type="number" tick={AXIS} stroke={GRID} />
          <YAxis type="category" dataKey="name" width={200} tick={AXIS} stroke={GRID} />
          <Tooltip {...tip("s")} />
          <Bar dataKey="offset" stackId="a" fill="transparent" legendType="none" />
          <Bar dataKey="queue" stackId="a" fill={c.queue} name="queue" />
          <Bar dataKey="prefill" stackId="a" fill={c.prefill} name="prefill" />
          <Bar dataKey="decode" stackId="a" fill={c.decode} name="decode"
               radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function Throughput({
  data, mean, binS, maxBatch,
}: { data: Tick[]; mean: number; binS: number; maxBatch: number }) {
  const c = useSeries();
  const unit = binS === 1 ? "second" : `${binS}s bin`;
  return (
    <ChartFrame
      filename="socrates-benchmark-throughput.png"
      title={`Throughput per ${unit}`}
      xLabel="seconds since run start"
      legend={[
        { label: `tokens/s in each ${unit} (left axis)`, color: c.decode },
        { label: "run average, sum(tokens) / wall (dashed)", color: c.prefill },
        { label: "rows generating (right axis)", color: c.queue },
      ]}
    >
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ left: 12, right: 18, top: 8, bottom: 8 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="t" tick={AXIS} stroke={GRID}
                 tickFormatter={(v) => `${v.toFixed(0)}s`} />
          <YAxis yAxisId="tps" tick={AXIS} stroke={GRID}
                 label={{ value: "tokens / s", angle: -90, position: "insideLeft", style: AXIS }} />
          <YAxis yAxisId="rows" orientation="right" domain={[0, maxBatch]} allowDecimals={false}
                 tick={AXIS} stroke={GRID}
                 label={{ value: "rows", angle: 90, position: "insideRight", style: AXIS }} />
          <Tooltip {...tip("")} />
          <Bar yAxisId="tps" dataKey="tokens_s" fill={c.decode} name="tokens/s"
               radius={[3, 3, 0, 0]} />
          <Line yAxisId="rows" type="stepAfter" dataKey="rows" stroke={c.queue}
                strokeWidth={1.5} dot={false} name="rows generating" />
          <ReferenceLine yAxisId="tps" y={mean} stroke={c.prefill} strokeDasharray="5 4"
                         ifOverflow="extendDomain" />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function Occupancy({ data, maxBatch }: { data: Result["occupancy"]; maxBatch: number }) {
  const c = useSeries();
  // queued requests hold no row, so they are drawn on their own axis rather than
  // stacked into the row count -- otherwise the total exceeds maxBatch
  const queuedMax = Math.max(...data.map((d) => d.queued), 1);
  return (
    <ChartFrame
      filename="socrates-benchmark-occupancy.png"
      title="Continuous batching occupancy"
      legend={[
        { label: "queued (right axis; holds no row)", color: c.queue },
        { label: "held (admitted, including prefill)", color: c.held },
        { label: "generating", color: c.decode },
      ]}
    >
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data} margin={{ left: 12, right: 18, top: 8, bottom: 8 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="t" tick={AXIS} stroke={GRID} tickFormatter={(v) => `${v.toFixed(0)}s`} />
          <YAxis yAxisId="rows" domain={[0, maxBatch]} allowDecimals={false} tick={AXIS} stroke={GRID}
                 label={{ value: "rows", angle: -90, position: "insideLeft", style: AXIS }} />
          <YAxis yAxisId="q" orientation="right" domain={[0, queuedMax]} allowDecimals={false}
                 tick={AXIS} stroke={GRID}
                 label={{ value: "queued", angle: 90, position: "insideRight", style: AXIS }} />
          <Tooltip {...tip("")} />
          <Area yAxisId="q" type="stepAfter" dataKey="queued" stroke={c.queue} fill={c.queue}
                fillOpacity={0.18} name="queued (right axis, holds no row)" />
          <Area yAxisId="rows" type="stepAfter" dataKey="held" stroke={c.held} fill={c.held}
                fillOpacity={0.5} name="held (admitted, incl. prefill)" />
          <Area yAxisId="rows" type="stepAfter" dataKey="generating" stroke={c.decode} fill={c.decode}
                fillOpacity={0.75} name="generating" />
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
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
    <ChartFrame
      filename={`socrates-benchmark-${field.replace("_s", "")}.png`}
      title={`${label.toUpperCase()} distribution`}
      xLabel="requests, sorted fastest to slowest"
      legend={[
        { label: "below p95", color: c.decode },
        { label: "p95 and above", color: c.prefill },
      ]}
    >
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ left: 12, right: 18, top: 8, bottom: 8 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="rank" tick={AXIS} stroke={GRID} />
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
    </ChartFrame>
  );
}
