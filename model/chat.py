"""Run the from-scratch model locally. No KV cache: every step re-runs the full
forward pass over the whole sequence.

    uv run -m model.chat
"""

import torch
from model.qwen import MODEL_ID, Qwen3, load_config, load_weights
from transformers import AutoTokenizer


def main():
    prompt = "how to make pizza?"

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Qwen3(load_config())
    model.load_state_dict(load_weights(), strict=True)
    model = model.eval().to("mps", torch.bfloat16)

    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = tok(text, return_tensors="pt").input_ids.to("mps")

    print(f"> {prompt}\n")
    with torch.no_grad():
        while True:
            # model(ids) runs forward pass for the model, which returns the logits (a matrix still)
            # to get actual logits, get the last one [0][-1]
            next_id = model(ids)[0][-1].argmax(-1).view(1, 1)
            # if next_id (token) is a special id, like end-of-turn token, then break
            if next_id.item() in tok.all_special_ids:
                break
            ids = torch.cat([ids, next_id], dim=1)
            print(tok.decode(next_id[0]), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
