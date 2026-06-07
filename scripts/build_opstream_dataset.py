#!/usr/bin/env python3
"""Build the unified op-stream dataset (geometry + lowered scripts, one vocab) for the
scaling-signal PoC. Streams the structured games -> serialize_game -> {"text"} jsonl,
train/val split. Keeps games with parts AND >=1 script (the cross-modal signal lives there).
"""
import sys, json, argparse, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from serialize_game import serialize_game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", default="data/structured_v3b.jsonl")
    ap.add_argument("--out-prefix", default="data/opstream")
    ap.add_argument("--max-parts", type=int, default=300)
    ap.add_argument("--max-scripts", type=int, default=30)
    ap.add_argument("--max-script-ops", type=int, default=120)
    ap.add_argument("--min-parts", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    a = ap.parse_args()

    if not Path(a.inp).exists():
        a.inp = "data/structured_v3.jsonl"
    rows, tot_chars, att_ops = [], 0, 0
    with open(a.inp, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if len(d.get("parts", [])) < a.min_parts:
                continue
            if not any(s.get("source", "").strip() for s in d.get("scripts", [])):
                continue
            text, st = serialize_game(d, a.max_parts, a.max_scripts, a.max_script_ops)
            if not text or st.get("script_ops", 0) == 0:
                continue
            rows.append(text)
            tot_chars += len(text)
            att_ops += text.count("ATTACH #")

    random.seed(0); random.shuffle(rows)
    nval = max(1, int(len(rows) * a.val_frac))
    val, train = rows[:nval], rows[nval:]
    for name, part in (("train", train), ("val", val)):
        p = f"{a.out_prefix}_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for t in part:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part):,} games -> {p}")
    print(f"total {len(rows):,} games  ~{tot_chars//4:,} tokens (~4 char/tok)  ATTACH-bindings ~{att_ops:,}")


if __name__ == "__main__":
    main()
