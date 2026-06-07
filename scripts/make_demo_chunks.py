#!/usr/bin/env python3
"""Chunk ALL games into spatially-ordered (z-order) windows for AR-LoRA training.

Uses the full dataset within compute limits via small chunks. Per game: Morton/z-order
the parts (so consecutive parts are spatially adjacent -> learnable for AR), reindex,
serialize part-lines + script blocks, split into ~chunk-tok windows. --max-tokens caps total.

v1 note: a script's `-> Pid` may reference a part in another chunk (cross-chunk coupling not
preserved); fine for volume validation of "ordering + volume kills the repetition".

Output: jsonl of {"text": chunk} ready for train_demo_lora.py.
"""
import sys, json, argparse, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_demo_obby import _suffix


def morton(p, lo, q):
    """z-order key: quantize (pos-lo) to grid q, interleave bits -> spatial locality."""
    xs = [min(max(int((p["pos"][k] - lo[k]) // q), 0), 1023) for k in range(3)]  # 10 bits/axis
    key = 0
    for b in range(10):
        for k in range(3):
            key |= ((xs[k] >> b) & 1) << (3 * b + k)
    return key


def part_line(i, p):
    x, y, z = (round(v) for v in p["pos"])
    sz = "x".join(str(round(v, 1)) for v in p["size"])
    return f"P{i} {p['shape']} @{x},{y},{z} {sz}" + _suffix(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", default="data/structured_v3.jsonl")
    ap.add_argument("out", nargs="?", default="data/demo_chunks.jsonl")
    ap.add_argument("--max-tokens", type=int, default=5_000_000)
    ap.add_argument("--chunk-tok", type=int, default=1800)
    ap.add_argument("--max-src", type=int, default=1000)
    ap.add_argument("--min-parts", type=int, default=20)
    ap.add_argument("--max-chunks-per-game", type=int, default=6)  # diversity: don't let big games dominate
    a = ap.parse_args()

    # pass 1: collect line byte-offsets only (tiny RAM — avoids loading all games)
    offsets = []
    with open(a.inp, "rb") as f:
        off = f.tell()
        ln = f.readline()
        while ln:
            offsets.append(off)
            off = f.tell()
            ln = f.readline()
    random.seed(0)
    random.shuffle(offsets)
    print(f"{len(offsets)} games; streaming to ~{a.max_tokens:,} tokens...")

    chunk_chars = a.chunk_tok * 4
    total_tok, n_chunks, n_games = 0, 0, 0
    with open(a.inp, "rb") as fi, open(a.out, "w", encoding="utf-8") as fo:
        for off in offsets:
            fi.seek(off)
            try:
                d = json.loads(fi.readline().decode("utf-8", "ignore"))
            except Exception:
                continue
            if d.get("parts_count", 0) < a.min_parts or not d.get("parts"):
                continue
            parts = d["parts"]
            lo = [min(p["pos"][k] for p in parts) for k in range(3)]
            order = sorted(range(len(parts)), key=lambda i: morton(parts[i], lo, q=4))
            newid = {parts[i]["id"]: rank for rank, i in enumerate(order)}
            lines = [part_line(rank, parts[i]) for rank, i in enumerate(order)]
            for s in d.get("scripts", []):
                att = newid.get(s.get("attach"))
                lines.append(f"[Script -> P{att}]" if att is not None else "[Script]")
                lines.append(s.get("source", "")[:a.max_src].strip())
                lines.append("[/Script]")

            name = d.get("id", "")[:60]
            gchunks, buf, clen = [], [], 0
            for ln in lines:
                if clen + len(ln) > chunk_chars and buf:
                    gchunks.append("\n".join(buf))
                    buf, clen = [], 0
                buf.append(ln)
                clen += len(ln) + 1
            if buf:
                gchunks.append("\n".join(buf) + "\n[/GAME]")
            random.shuffle(gchunks)                                  # vary which regions we keep
            for body in gchunks[:a.max_chunks_per_game]:
                text = f"[GAME: {name}]\n" + body
                fo.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                total_tok += len(text) // 4
                n_chunks += 1
            n_games += 1
            if total_tok >= a.max_tokens:
                break

    print(f"wrote {n_chunks:,} chunks from {n_games} games -> {a.out}  (~{total_tok:,} tokens)")


if __name__ == "__main__":
    main()
