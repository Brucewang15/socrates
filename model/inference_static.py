import queue
import time
from dataclasses import dataclass, field

import torch
from model.qwen.qwen_kv_seq import MODEL_ID, KVCache, Qwen3, load_config, load_weights
from transformers import AutoTokenizer

BATCH_SIZE = 4
MAX_NEW_TOKENS = 2048
DEVICE = "mps"
DTYPE = torch.bfloat16


@dataclass
class Request:
    prompt: str
    output: list[int] = field(default_factory=list)
    done: bool = False
    submitted: float = field(default_factory=time.perf_counter)
    first_token: float = 0.0
    finished: float = 0.0
    delivered: float = 0.0
    steps: int = 0


class Engine:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.cfg = load_config()
        model = Qwen3(self.cfg)
        model.load_state_dict(load_weights(), strict=True)
        self.model = model.eval().to(DEVICE, DTYPE)
        self.pending: queue.Queue[Request] = queue.Queue()

    def submit(self, req: Request) -> None:
        self.pending.put(req)

    def encode(self, batch: list[Request]) -> torch.Tensor:
        enc = [
            self.tok(
                self.tok.apply_chat_template(
                    [{"role": "user", "content": r.prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                ),
                return_tensors="pt",
            ).input_ids
            for r in batch
        ]
        n = max(e.shape[1] for e in enc)
        pad = self.tok(" ").input_ids[0]
        return torch.cat(
            [torch.nn.functional.pad(e, (n - e.shape[1], 0), value=pad) for e in enc],
            dim=0,
        ).to(DEVICE)

    def run_batch(self, batch: list[Request]) -> None:
        # ids now is a tensor [Batch, max_input_prompt_len]
        ids = self.encode(batch)
        # Share kv cache for all requests, which is reset after every batch.
        cache = KVCache(
            len(batch), self.cfg["num_hidden_layers"], self.cfg["num_key_value_heads"],
            self.cfg["head_dim"], dtype=DTYPE, device=DEVICE,
        )
        steps = 0
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                step = ids if len(cache) == 0 else ids[:, -1:]
                # Forward pass for all input tokens for all reqs in the batch
                next_ids = self.model(step, cache)[:, -1].argmax(-1, keepdim=True)
                ids = torch.cat([ids, next_ids], dim=1)
                for b, req in enumerate(batch):
                    if req.done:
                        continue
                    token = next_ids[b, 0].item()
                    if token in self.tok.all_special_ids:
                        req.done = True
                        req.finished = time.perf_counter()
                    else:
                        req.output.append(token)
                        if not req.first_token:
                            req.first_token = time.perf_counter()
                steps += 1
                if all(r.done for r in batch):
                    break
        now = time.perf_counter()
        for req in batch:
            if not req.finished:
                req.finished = now
            req.delivered = now
            req.steps = steps

    def loop(self) -> None:
        while True:
            if self.pending.qsize() >= BATCH_SIZE:
                batch = [self.pending.get() for _ in range(BATCH_SIZE)]
                self.run_batch(batch)
                for req in batch:
                    print(f"> {req.prompt}\n{self.tok.decode(req.output)}\n")


def main() -> None:
    engine = Engine()
    prompts = ["how to make pizza?", "what is 2+2?", "who are you?", "name a color"]
    for prompt in prompts:
        engine.submit(Request(prompt))
    engine.loop()


if __name__ == "__main__":
    main()
