streaming

the problem
decode_step() produces one token for every live row at once, so a step's output
is a batch of tokens belonging to 18 different clients. Something has to route
each token to the right connection.

why the metrics kept disappearing
threadpool exhaustion:

    def generate(...)   sync -> anyio threadpool, then blocks on
                        r.event.wait(300) for the WHOLE generation
    def metrics(...)    sync -> also needs a threadpool thread
    threadpool 40,      MAX_QUEUE 288

Every in-flight request holds a thread until it finishes. Past 40 concurrent
there is none left, so /metrics queues behind them and dies at the 900ms scrape
timeout. The graphs blanked when in-flight crossed 40, not when the GPU got busy.

throughput is measured wrong today
OUT_TOKENS.inc(n) fires in that same handler when the request completes, so all
800 of its tokens land at one instant, 30s after the first was produced. rate()
is a completion histogram, not throughput: zero while the GPU generates flat out,
then a spike when a long request retires. Prefill has the same bug on the
adjacent line -- which is why prefill and decode hit zero together. Same line,
same moment, nothing to do with prefill running.

streaming fixes both
SSE is an async generator, so the handler cannot be a sync def blocking on a
threading.Event. Making it async is not optional, and it is exactly the fix:

    async def generate(...)   StreamingResponse, holds no thread, awaits a queue
    async def metrics(...)    runs on the event loop, no thread at all
    async def health(...)     same

With no sync handlers the threadpool is never touched and cannot be exhausted.
Concurrency is then bounded by memory -- one coroutine per request, a few KB --
not by 40 threads.

And once tokens visibly leave record() one at a time, counting them there is the
obvious thing, which makes rate() mean tokens/sec.

the design
The handler holds the Request object, so there is no lookup table and no id:

    record()   req.stream.put(token)       # engine side, per token produced
    handler    await req.stream.get()      # yield SSE until a sentinel

asyncio.Queue is not thread-safe and the decode loop is a thread, so record()
uses loop.call_soon_threadsafe(q.put_nowait, token).

Cost: one queue push per token per row. At batch 18 that is 327/s, against an
event loop that does ~2M callbacks/s. 0.02% of capacity.

backend must not buffer
backend does `r = await client.post(...)`, which waits for the whole response and
silently kills streaming -- the client gets everything at once at the end,
exactly like today. Needs client.stream() + aiter_bytes() inside a
StreamingResponse, and nothing in between may materialise the body (no
GZipMiddleware, no response-model validation).

cancellation stops being optional
retire() only runs when generation finishes naturally. A closed tab raises
GeneratorExit in the handler; the decode loop never hears about it and generates
to MAX_NEW_TOKENS. One abandoned request holds a row ~56s = 5.6% of capacity;
three is 17% of the batch generating for nobody. Needs a cancelled flag the loop
checks alongside done.

what to change
    1. async def on generate / metrics / health
    2. req.stream queue; record() pushes via call_soon_threadsafe
    3. StreamingResponse on both tiers; backend switches to client.stream()
    4. cancelled flag so closed tabs stop generating
    5. OUT_TOKENS.inc() moves into decode_step(), IN_TOKENS into prefill()

architecture

    browser
      | EventSource, SSE
      v
    backend :8000        auth, history, rate limit
      |                  client.stream() -> aiter_bytes() -> StreamingResponse
      | SSE              MUST NOT BUFFER
      v
    +-----------------------------------------------------------+
    | model :8080                                                |
    |                                                            |
    |   async handler --await--> req.stream (asyncio.Queue)      |
    |        | yield SSE                        ^                |
    |        v                                  | call_soon_     |
    |     client                                | threadsafe     |
    |                                           |                |
    |   decode thread: admit() -> decode_step() -> record()      |
    |                  18 rows, 1 token each, every ~55ms        |
    |                  counts tokens here, not at completion     |
    +------------------------------------------------------------+

Frontend also needs updating.
