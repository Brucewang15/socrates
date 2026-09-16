streaming

the problem
decode_step() produces one token for every live row at once, so a step's output
is a batch of tokens belonging to 18 different clients. Something has to route
each token to the right connection.

throughput is measured wrong today
OUT_TOKENS.inc(n) fires in the HTTP handler when a request completes, so all 800
of its tokens land at one instant, 30s after the first was produced. rate() is a
completion histogram, not throughput: zero while the GPU generates flat out, then
a spike when a long request retires. Prefill has the same bug on the adjacent
line -- which is why prefill and decode hit zero together. Same line, same moment,
nothing to do with prefill running.

record() is where a token is produced. Count there.

why the metrics kept disappearing
Not the GIL -- PyTorch releases it inside CUDA ops and inference was fine at 82%
MBU. It is threadpool exhaustion:

    def generate(...)   sync -> anyio threadpool, then blocks on
                        r.event.wait(300) for the WHOLE generation
    def metrics(...)    sync -> also needs a threadpool thread
    threadpool 40,      MAX_QUEUE 288

Every in-flight request holds a thread until it finishes. Past 40 concurrent
there is none left, so /metrics queues behind them and dies at the 900ms scrape
timeout. The graphs blanked when in-flight crossed 40, not when the GPU got busy.

Fix is async def, not processes. Streaming forces it anyway: an SSE handler is an
async generator, so it cannot be a sync def blocking on a threading.Event.

in-process design (build this first)
The handler holds the Request, so there is no lookup and no id:

    record()   req.stream.put(token)       # engine side, per token
    handler    await req.stream.get()      # yield SSE until a sentinel

asyncio.Queue is not thread-safe and the decode loop is a thread, so record()
uses loop.call_soon_threadsafe(q.put_nowait, token).

where the line goes, if we split
The scheduler cannot leave the cache. A "row" is a slice of the KV tensor:
admit() calls cache.reset(row), retire() calls cache.move_row() which copies GPU
memory, decode_step() builds the model's input from cache.lengths and n_active.
Separate them and the scheduler manages memory it cannot touch.

HTTP is the only clean seam -- it shares nothing but messages:

    process 1  model server   handlers, streams{id: Queue}, reader task
    process 2  the engine     pending deque, rows, KVCache, Qwen3, run()

server.py loses serve(); the loop moves to process 2. submit() stops being a
function call and becomes a write to the input socket. prefill() and
decode_step() do not change at all.

when to split
Only when the decode loop measurably slows with streams attached. Streaming makes
handlers wake every step instead of once per request -- ~330 wakeups/s at batch
18, in the same process as the loop. Until that costs tokens, do not split. The
per-request queue is identical either way, so nothing is wasted.

ipc
ZMQ, two sockets. A socket is one endpoint -- an fd you read/write like a file,
where the other end is another process. ZMQ adds message framing and queueing on
top. Submitting crosses the boundary too: a fire-and-forget message, not a call.

One message per STEP with every row's token, not one per token: 18 msg/s instead
of 327. A ZMQ message goes to exactly one reader, so 18 handlers reading the same
socket would have the first recv() eat everyone's message -- hence one reader
task owning the socket, demuxing into streams{id: Queue}.

measuring throughput across the split
No timestamps in the messages. The engine exposes its own /metrics
(prometheus_client.start_http_server -- no framework, no blocking handlers) and
increments the counter in decode_step. Prometheus scrapes two targets and
timestamps each scrape itself. This also dodges the original bug: the engine's
endpoint is not behind 40 threadpool slots held by blocked handlers.

Timestamps in the stream answer a different question -- lag between a token being
produced and delivered. A diagnostic, not the throughput metric.

both proxies must not buffer
backend does `r = await client.post(...)`, which waits for the whole response and
silently kills streaming. Needs client.stream() + aiter_bytes() inside a
StreamingResponse, and nothing in between may materialise the body.

cancellation stops being optional
retire() only runs when generation finishes naturally. A closed tab raises
GeneratorExit in the handler; the decode loop never hears about it and generates
to MAX_NEW_TOKENS. One abandoned request holds a row ~56s = 5.6% of capacity;
three is 17% of the batch generating for nobody. Needs a cancelled flag the loop
checks alongside done, and across a boundary, an abort message.

architecture

    browser
      | SSE
    backend :8000      auth, history, rate limit; MUST NOT BUFFER
      | SSE
    model server :8080                     the engine
      handler --+                            own GIL, own threadpool
      handler --+-> streams{id:Q} <- reader <-PULL- PUSH- record()
      handler --+                                     1 msg/step
          |                                           [{id, token}, ...]
          +----------PUSH submit / abort----------> pending deque
                                                          |
      /metrics <- prometheus -> /metrics            admit() -> rows -> KVCache
      (handler RED)             (tokens, depth)     decode_step() -> Qwen3

Frontend also needs updating.
