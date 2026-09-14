"use client";

import Link from "next/link";
import { Logo } from "./logo";
import { useRef, useState } from "react";

// Inlined at build time, not read at runtime. Locally this comes from
// frontend/.env.local; on Vercel from the project's environment variables,
// where a missing value silently ships the localhost fallback below.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Message = { role: "user" | "assistant"; content: string };

const EXAMPLES = [
  '"Explain attention in simple terms" →',
  '"Why is decode memory-bandwidth bound?" →',
  '"What does a KV cache actually store?" →',
];
const CAPABILITIES = [
  "Runs a Qwen3-4B forward pass written from scratch",
  "Serves requests through a hand-built batching engine",
  "Streams tokens as they are generated",
];
const LIMITATIONS = [
  "The inference engine is not wired up yet",
  "No authentication, sessions, or persistence",
  "Single node, single GPU for now",
];

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLTextAreaElement>(null);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;

    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    setBusy(true);
    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text }),
      });
      const data = await res.json();
      setMessages((m) => [...m, { role: "assistant", content: data.response }]);
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

  return (
    <div className="app">
      <aside className="sidebar">
        <button className="new-chat" onClick={() => setMessages([])}>
          <span>+</span> New chat
        </button>
        <div className="sidebar-bottom">
          <Link href="/benchmark" className="bench-link">
            <span>▤</span> Benchmark
          </Link>
          <div className="sidebar-footer">
            <Logo size={18} />
            <span className="brand">Socrates</span>
          </div>
        </div>
      </aside>

      <main className="main">
        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty">
              <h1>Socrates</h1>
              <div className="columns">
                <div className="column">
                  <h2>
                    <span className="icon">☀️</span>Examples
                  </h2>
                  {EXAMPLES.map((t) => (
                    <div className="card" key={t}>
                      {t}
                    </div>
                  ))}
                </div>
                <div className="column">
                  <h2>
                    <span className="icon">⚡</span>Capabilities
                  </h2>
                  {CAPABILITIES.map((t) => (
                    <div className="card" key={t}>
                      {t}
                    </div>
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
                    {m.role === "user" ? "You" : "S"}
                  </div>
                  <div>{m.content}</div>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="composer">
          <form onSubmit={send}>
            <textarea
              ref={box}
              rows={1}
              value={input}
              placeholder="Send a message."
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) send(e);
              }}
            />
            <button type="submit" disabled={!input.trim() || busy}>
              ➤
            </button>
          </form>
          <p className="disclaimer">
            socrates may produce incorrect tokens. The engine is a work in progress.
          </p>
        </div>
      </main>
    </div>
  );
}
