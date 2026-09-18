"use client";

import Link from "next/link";
import { Logo } from "./logo";
import { useEffect, useRef, useState } from "react";
import { FaGithub } from "react-icons/fa6";
import { LuChartNoAxesColumn, LuPlus, LuArrowUp } from "react-icons/lu";

// Inlined at build time, not read at runtime. Locally this comes from
// frontend/.env.local; on Vercel from the project's environment variables,
// where a missing value silently ships the localhost fallback below.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const REPO = "https://github.com/Brucewang15/inference";

type Message = { role: "user" | "assistant"; content: string };

// Card text is what you read; prompt is what gets sent. For examples they are
// the same thing, so the quotes and arrow live in the markup rather than here.
const EXAMPLES = [
  "Explain attention in simple terms",
  "Why is decode memory-bandwidth bound?",
  "What does a KV cache actually store?",
];
const CAPABILITIES = [
  {
    text: "Runs a Qwen3-4B forward pass written from scratch",
    prompt: "What happens inside a single transformer forward pass?",
  },
  {
    text: "Serves requests through a hand-built batching engine",
    prompt: "What is continuous batching and why does it beat static batching?",
  },
  {
    text: "Streams tokens as they are generated",
    prompt: "Why do language models emit one token at a time?",
  },
];
const LIMITATIONS = [
  "No authentication, sessions, or persistence",
  "Single node, single GPU for now",
];

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLTextAreaElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  // Follow the stream, but stop fighting the user the moment they scroll up.
  const stick = useRef(true);

  useEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  function onScroll() {
    const el = scroller.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }

  async function send(text: string) {
    if (!text || busy) return;

    stick.current = true;
    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    setBusy(true);
    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text }),
      });
      if (!res.ok || !res.body) throw new Error(await res.text());

      setMessages((m) => [...m, { role: "assistant", content: "" }]);
      const append = (delta: string) =>
        setMessages((m) => {
          const last = m[m.length - 1];
          return [...m.slice(0, -1), { ...last, content: last.content + delta }];
        });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        // one JSON object per line; the tail may be half a line
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          const ev = JSON.parse(line);
          if (ev.delta) append(ev.delta);
        }
      }
    } catch {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Could not reach the backend at ${API_URL}.` },
      ]);
    } finally {
      setBusy(false);
      box.current?.focus();
    }
  }

  function submit(e: React.FormEvent) {
    e.preventDefault();
    send(input.trim());
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <button className="new-chat" onClick={() => setMessages([])}>
          <LuPlus size={16} /> New chat
        </button>
        <div className="sidebar-bottom">
          <Link href="/benchmark" className="bench-link">
            <LuChartNoAxesColumn size={16} /> Benchmark
          </Link>
          <div className="sidebar-footer">
            <Logo size={18} />
            <span className="brand">Socrates</span>
          </div>
        </div>
      </aside>

      <main className="main">
        <div className="messages" ref={scroller} onScroll={onScroll}>
          {messages.length === 0 ? (
            <div className="empty">
              <h1>Socrates</h1>
              <div className="columns">
                <div className="column">
                  <h2>
                    <span className="icon">☀️</span>Examples
                  </h2>
                  {EXAMPLES.map((t) => (
                    <button
                      className="card clickable"
                      key={t}
                      disabled={busy}
                      onClick={() => send(t)}
                    >
                      &ldquo;{t}&rdquo; →
                    </button>
                  ))}
                </div>
                <div className="column">
                  <h2>
                    <span className="icon">⚡</span>Capabilities
                  </h2>
                  {CAPABILITIES.map((c) => (
                    <button
                      className="card clickable"
                      key={c.text}
                      disabled={busy}
                      onClick={() => send(c.prompt)}
                    >
                      {c.text}
                    </button>
                  ))}
                </div>
                <div className="column">
                  <h2>
                    <span className="icon">⚠️</span>Limitations
                  </h2>
                  {LIMITATIONS.map((t) => (
                    <div className="card" key={t}>
                      {t}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            messages.map((m, i) => (
              <div className={`row ${m.role}`} key={i}>
                <div className="row-inner">
                  <div className={`avatar ${m.role}`}>
                    {m.role === "user" ? "You" : <Logo size={16} />}
                  </div>
                  <div>{m.content}</div>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="composer">
          <form onSubmit={submit}>
            <textarea
              ref={box}
              rows={1}
              value={input}
              placeholder="Send a message."
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) submit(e);
              }}
            />
            <button type="submit" disabled={!input.trim() || busy}>
              <LuArrowUp size={18} />
            </button>
          </form>
          <p className="disclaimer">
            <a href={REPO} target="_blank" rel="noreferrer" className="repo">
              <FaGithub size={14} /> Brucewang15/inference
            </a>
          </p>
        </div>
      </main>
    </div>
  );
}
