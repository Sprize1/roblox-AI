#!/usr/bin/env python3
"""Phase-2 dataset: (layout context -> attached script) pairs, for the coupling model.

For each ATTACHED script: take the attach part + its K z-order neighbours (the local geometry
the script controls), serialize them, then the script header (-> P{local_id}) + the Luau source.
A pretrained code-LLM (LoRA) learns to generate the script GIVEN the surrounding layout =
the geometry<->script coupling. Phase 1 generates the layout; Phase 2 scripts it.

Output: jsonl of {"text"} ready for train_demo_lora.py.
"""
import sys, json, argparse, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_demo_chunks import morton, part_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", default="data/structured_v3.jsonl")
    ap.add_argument("out", nargs="?", default="data/script_pairs.jsonl")
    ap.add_argument("--k", type=int, default=30, help="layout-context parts around the attach part")
    ap.add_argument("--max-src", type=int, default=1500)
    ap.add_argument("--min-src", type=int, default=40)
    ap.add_argument("--max-scripts-per-game", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=5_000_000)
    a = ap.parse_args()

    offsets = []
    with open(a.inp, "rb") as f:
        off = f.tell(); ln = f.readline()
        while ln:
            offsets.append(off); off = f.tell(); ln = f.readline()
    random.seed(0); random.shuffle(offsets)
    print(f"{len(offsets)} games; building (layout->script) pairs to ~{a.max_tokens:,} tok...")

    total, n = 0, 0
    with open(a.inp, "rb") as fi, open(a.out, "w", encoding="utf-8") as fo:
        for off in offsets:
            fi.seek(off)
            try:
                d = json.loads(fi.readline().decode("utf-8", "ignore"))
            except Exception:
                continue
            parts = d.get("parts", [])
            scripts = [s for s in d.get("scripts", [])
                       if s.get("attach") is not None and len(s.get("source", "").strip()) >= a.min_src]
            if not parts or not scripts:
                continue
            lo = [min(p["pos"][k] for p in parts) for k in range(3)]
            order = sorted(range(len(parts)), key=lambda i: morton(parts[i], lo, 4))
            rank_of = {parts[order[r]]["id"]: r for r in range(len(order))}     # part id -> z-order rank
            random.shuffle(scripts)
            for s in scripts[:a.max_scripts_per_game]:
                att = s["attach"]
                if att not in rank_of:
                    continue
                rank = rank_of[att]
                b = min(len(order), max(rank - a.k // 2, 0) + a.k)
                start = max(0, b - a.k)
                window = order[start:b]
                local = {parts[i]["id"]: r for r, i in enumerate(window)}
                lines = [part_line(local[parts[i]["id"]], parts[i]) for i in window]
                text = ("[PARTS]\n" + "\n".join(lines)
                        + f"\n[{s.get('class', 'Script')} -> P{local[att]}]\n"
                        + s["source"][:a.max_src].strip() + "\n[/SCRIPT]")
                fo.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                total += len(text) // 4; n += 1
            if total >= a.max_tokens:
                break

    print(f"wrote {n:,} (layout->script) pairs -> {a.out}  (~{total:,} tokens)")


if __name__ == "__main__":
    main()
