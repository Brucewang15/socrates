"""Greedy outputs of the paged engine against the dense one, token by token.

Two engines do not fit on one L4 at once (8 GB of weights + 5.3 GB of cache
each), so each runs in its own process and writes JSON, and a third call
compares them:

    python bench/day_8/equivalence.py run --engine cont  --out /out/eq_cont.json
    python bench/day_8/equivalence.py run --engine paged --out /out/eq_paged.json
    python bench/day_8/equivalence.py run --engine paged --force /out/eq_cont.json \\
        --out /out/eq_paged_forced.json
    python bench/day_8/equivalence.py compare /out/eq_cont.json /out/eq_paged.json \\
        --forced /out/eq_paged_forced.json

All run exactly as served: compiled decode, CUDA graphs, all 16 PROMPTS
admitted together so rows are ragged and retire at different steps.

Identity is not guaranteed in bf16, and the comparison is built to tell a
bug from a rounding difference. The two attention paths sum the same products
in a different order, so logits differ in the last bits. Where the top two
tokens are nearly tied, that is enough to flip an argmax, and from that point
the continuations legitimately diverge. So every generated token carries the
gap between its top-1 and top-2 logits, reported in bf16 ulps of the top logit:
0 is an exact tie that any perturbation can flip, a handful is rounding-sized,
and dozens would be a bug.

The free-running comparison has one weakness: after the first flip the two
engines are answering different questions, so it says nothing about the rest of
the answer. The teacher-forced run fixes that. It feeds the paged engine the
dense engine's tokens instead of its own, so both see the same prefix at every
step, and checks the argmax at every position independently.
"""

import argparse
import importlib
import json
import math
import sys

PREFIX = 32          # the gate: this many leading tokens must match on every prompt


def run(engine_name: str, out: str, force: str | None = None) -> None:
    import torch

    mod = importlib.import_module({"cont": "model.inference_cont",
                                   "paged": "model.inference_paged"}[engine_name])
    engine = mod.Engine()
    margins: dict[int, list[float]] = {}
    tops: dict[int, list[float]] = {}
    own: dict[int, list[int]] = {}

    def watch(model):
        def wrapped(ids, rows, *rest):
            logits = model(ids, rows, *rest)
            top = logits[:, -1].float().topk(2).values
            for req, (t1, t2) in zip(engine.rows[rows], top.tolist()):
                margins.setdefault(id(req), []).append(t1 - t2)
                tops.setdefault(id(req), []).append(t1)
            return logits
        return wrapped

    # A host sync per step, which is why this is an equivalence run and not a
    # benchmark. The wrapped callables are what the engine calls, so compiled
    # decode and eager prefill both stay exactly as served.
    engine.decode_model = watch(engine.decode_model)
    engine.eager_model = watch(engine.eager_model)

    # Teacher forcing: every row is fed the reference run's next token instead of
    # its own argmax, and stops where the reference stopped. Both engines then see
    # the same prefix at every step, so one near-tie cannot cascade into a
    # different continuation, and every position becomes an independent test of
    # "same argmax given the same context".
    if force:
        ref = {p["prompt"]: p["tokens"] for p in json.load(open(force))["prompts"]}
        stop = engine.tok.all_special_ids[0]
        free_record = engine.record

        def record(req, token):
            k = len(req.output)
            own.setdefault(id(req), []).append(token)
            forced = ref[req.prompt]
            free_record(req, forced[k] if k < len(forced) else stop)

        engine.record = record

    reqs = [engine.submit(p) for p in mod.PROMPTS]
    with torch.no_grad():
        engine.run()
    result = {"engine": engine_name, "forced": bool(force), "prompts": [
        {"prompt": r.prompt, "tokens": r.output, "margins": margins.get(id(r), []),
         "top1": tops.get(id(r), []),
         "own": own.get(id(r), []), "text": engine.tok.decode(r.output)} for r in reqs]}
    if force:
        result["special_ids"] = list(engine.tok.all_special_ids)
        result["max_new_tokens"] = mod.MAX_NEW_TOKENS
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{engine_name}{' (forced)' if force else ''}: "
          f"{sum(len(r.output) for r in reqs)} tokens -> {out}", flush=True)


def ulps(p: dict, k: int) -> str:
    """Top-1/top-2 gap at step k in units of one bf16 ulp at the top logit.

    Logits leave the model in bf16, whose spacing is 2^(e-7) for a value in
    [2^e, 2^(e+1)): 0.125 at 16-32, 0.25 at 32-64. A gap of 0 is an exact tie,
    and argmax then picks by vocabulary index -- any perturbation can flip it.
    """
    if k >= len(p["margins"]) or k >= len(p.get("top1", [])):
        return "-"
    gap, top = p["margins"][k], abs(p["top1"][k])
    ulp = 2.0 ** (math.floor(math.log2(top)) - 7) if top > 0 else 1.0
    return f"{gap / ulp:.0f}"


def compare(a_path: str, b_path: str) -> int:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"{a['engine']} vs {b['engine']}, gate: first {PREFIX} tokens identical\n")
    print(f"{'prompt':<38} {'len a':>6} {'len b':>6} {'diverge':>8} {'gap a':>7} {'gap b':>7}"
          f" {'ulps a':>7} {'ulps b':>7}")
    ok, identical = True, 0
    for pa, pb in zip(a["prompts"], b["prompts"]):
        ta, tb = pa["tokens"], pb["tokens"]
        i = next((k for k, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if i is None and len(ta) != len(tb):
            i = min(len(ta), len(tb))            # one stopped where the other went on
        if i is None:
            identical += 1
            div, ga, gb, ua, ub = "same", "", "", "", ""
        else:
            div = str(i)
            ga = f"{pa['margins'][i]:.3f}" if i < len(pa["margins"]) else "-"
            gb = f"{pb['margins'][i]:.3f}" if i < len(pb["margins"]) else "-"
            ua, ub = ulps(pa, i), ulps(pb, i)
        # a prompt shorter than PREFIX passes only if it is identical outright
        passed = i is None or i >= PREFIX
        ok &= passed
        print(f"{pa['prompt'][:38]:<38} {len(ta):>6} {len(tb):>6} {div:>8} {ga:>7} {gb:>7}"
              f" {ua:>7} {ub:>7}{'' if passed else '   <-- FAIL'}")
    total = len(a["prompts"])
    print(f"\n{identical}/{total} fully identical; "
          f"gate (first {PREFIX} tokens on all {total}): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def compare_forced(a_path: str, f_path: str) -> int:
    """Per-position argmax agreement of a teacher-forced run against the run it
    was forced with. A bug disagrees often and by a lot; rounding disagrees
    rarely and only where the top two logits are within an ulp or two."""
    a, f = json.load(open(a_path)), json.load(open(f_path))
    special = set(f.get("special_ids", []))
    cap = f.get("max_new_tokens", 1024)
    print(f"\nteacher-forced: {f['engine']} fed {a['engine']}'s tokens, argmax compared "
          f"at every position\n")
    print(f"{'prompt':<38} {'steps':>6} {'agree':>6} {'max ulps at a disagreement':>27}")
    steps = agree = 0
    worst = 0.0
    for pa, pf in zip(a["prompts"], f["prompts"]):
        ref, own = pa["tokens"], pf["own"]
        n = ok = 0
        row_worst = None
        for k, tok in enumerate(own):
            if k > len(ref) or (k == len(ref) and len(ref) >= cap):
                break                # the reference hit the length cap, not a stop token
            # at len(ref) the reference emitted a stop token
            same = tok == ref[k] if k < len(ref) else tok in special
            n += 1
            ok += same
            u = ulps(pf, k)
            if not same and u != "-":
                row_worst = max(row_worst or 0.0, float(u))
        steps += n
        agree += ok
        worst = max(worst, row_worst or 0.0)
        shown = "" if row_worst is None else f"{row_worst:.0f}"
        print(f"{pa['prompt'][:38]:<38} {n:>6} {ok:>6} {shown:>27}")
    print(f"\n{agree}/{steps} positions agree ({100 * agree / max(steps, 1):.2f}%); "
          f"largest top-2 gap at any disagreement: {worst:.0f} bf16 ulps")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", choices=["cont", "paged"], required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--force", help="teacher-force with this run's tokens")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--forced", help="also report a teacher-forced run against a")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.engine, args.out, args.force)
    else:
        code = compare(args.a, args.b)
        if args.forced:
            compare_forced(args.a, args.forced)
        sys.exit(code)


if __name__ == "__main__":
    main()
