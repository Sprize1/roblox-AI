#!/usr/bin/env python3
"""Phase-2 inference: given a layout + a target part, generate the attached Luau script.

Rebuilds the SAME [PARTS] context as training (target part + K z-order neighbours) + the
script header, then lets the Phase-2 LoRA generate the script. Tests the geometry->script
coupling: does the model write a script that fits the local geometry?

For evaluation we pick a part that HAS a ground-truth script, so generated vs real are shown
side by side. (Later, feed a Phase-1 generated layout + chosen part instead.)

Usage:
  python scripts/generate_phase2.py --adapter models/demo_lora_phase2 --game-index -1 --nth 0
"""
import sys, json, argparse, torch
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          StoppingCriteria, StoppingCriteriaList)
from peft import PeftModel


class StopOnText(StoppingCriteria):
    """Stop generation as soon as `stop_str` appears in the new tokens (one clean script)."""
    def __init__(self, tokenizer, stop_str, prompt_len):
        self.tok, self.stop, self.plen = tokenizer, stop_str, prompt_len

    def __call__(self, input_ids, scores, **kw):
        return self.stop in self.tok.decode(input_ids[0][self.plen:], skip_special_tokens=True)

sys.path.insert(0, str(Path(__file__).parent))
from make_demo_chunks import morton, part_line


def build_context(parts, target_id, k):
    lo = [min(p["pos"][j] for p in parts) for j in range(3)]
    order = sorted(range(len(parts)), key=lambda i: morton(parts[i], lo, 4))
    rank = {parts[order[r]]["id"]: r for r in range(len(order))}
    if target_id not in rank:
        return None, None
    r = rank[target_id]
    b = min(len(order), max(r - k // 2, 0) + k)
    start = max(0, b - k)
    window = order[start:b]
    local = {parts[i]["id"]: idx for idx, i in enumerate(window)}
    lines = [part_line(local[parts[i]["id"]], parts[i]) for i in window]
    return "[PARTS]\n" + "\n".join(lines), local[target_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--adapter", default="models/demo_lora_phase2")
    ap.add_argument("--data", default="data/structured_v3.jsonl")
    ap.add_argument("--game-index", type=int, default=-1, help="-1 = first game with attached scripts")
    ap.add_argument("--nth", type=int, default=0, help="which attached-script part to target")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--min-src", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--rep", type=float, default=1.2)
    ap.add_argument("--no-repeat-ngram", type=int, default=4, help="0 to disable")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # --- pick a game that has attached scripts (with ground truth for comparison) ---
    game = None
    with open(a.data, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if a.game_index >= 0 and i != a.game_index:
                continue
            d = json.loads(line)
            att = [s for s in d.get("scripts", [])
                   if s.get("attach") is not None and len(s.get("source", "").strip()) >= a.min_src
                   and s["attach"] in {p["id"] for p in d.get("parts", [])}]
            if att and d.get("parts"):
                game = d; game["_att"] = att
                if a.game_index < 0:
                    break
            if a.game_index >= 0:
                break
    if not game:
        print("no suitable game found"); return

    att = game["_att"]
    s = att[min(a.nth, len(att) - 1)]
    target_id, cls, truth = s["attach"], s.get("class", "Script"), s["source"].strip()
    ctx, tlocal = build_context(game["parts"], target_id, a.k)
    prompt = ctx + f"\n[{cls} -> P{tlocal}]\n"
    print(f"game='{game.get('name','?')}'  parts={len(game['parts'])}  attached-scripts={len(att)}  target=P{tlocal} ({cls})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
        attn_implementation="eager")
    model = PeftModel.from_pretrained(model, a.adapter)
    model.to(dev).eval()

    ids = tok(prompt, return_tensors="pt").to(dev)
    plen = ids["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=a.max_new, do_sample=True,
                             temperature=a.temp, repetition_penalty=a.rep,
                             no_repeat_ngram_size=a.no_repeat_ngram or 0,
                             stopping_criteria=StoppingCriteriaList([StopOnText(tok, "[/SCRIPT]", plen)]),
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][plen:], skip_special_tokens=True)
    gen = gen.split("[/SCRIPT]")[0].rstrip()

    block = (f"=== CONTEXT (header) ===\n[{cls} -> P{tlocal}]\n"
             f"\n=== GENERATED SCRIPT ===\n{gen}\n"
             f"\n=== GROUND TRUTH ===\n{truth}\n")
    print(block)
    if a.out:
        Path(a.out).write_text(f"=== FULL PROMPT ===\n{prompt}\n{block}", encoding="utf-8")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
