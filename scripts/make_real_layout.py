#!/usr/bin/env python3
"""Extract REAL Roblox layouts as fixed-K box-clusters for the checker-grounding A/B.

From each game: z-order the parts, slide windows of K spatially-adjacent parts, normalize each
to a local frame (min corner -> 0), emit as 'B x y z sx sy sz' lines (axis-aligned; rotation
dropped — consistent simplification, the A/B stays fair). Fixed K -> every layout has K parts,
so F and U are AUTOMATICALLY matched on part-count (clean anti-Goodhart by construction).
"""
import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_demo_chunks import morton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", default="data/structured_v3b.jsonl")
    ap.add_argument("--out", default="data/real_layout.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max-per-game", type=int, default=12)
    ap.add_argument("--max-extent", type=int, default=200)
    a = ap.parse_args()

    n = 0
    with open(a.inp, encoding="utf-8") as fi, open(a.out, "w", encoding="utf-8") as fo:
        for line in fi:
            try:
                d = json.loads(line)
            except Exception:
                continue
            parts = d.get("parts", [])
            if len(parts) < a.k:
                continue
            lo_all = [min(p["pos"][k] for p in parts) for k in range(3)]
            order = sorted(range(len(parts)), key=lambda i: morton(parts[i], lo_all, 4))
            made = 0
            for start in range(0, len(order) - a.k + 1, a.stride):
                if made >= a.max_per_game:
                    break
                win = [parts[order[start + t]] for t in range(a.k)]
                lo = [min(p["pos"][k] for p in win) for k in range(3)]
                lines, ok = [], True
                for p in win:
                    x, y, z = (round(p["pos"][k] - lo[k]) for k in range(3))
                    sx, sy, sz = (max(1, round(abs(v))) for v in p["size"])
                    if max(x, y, z, sx, sy, sz) > a.max_extent:
                        ok = False; break
                    lines.append(f"B {x} {y} {z} {sx} {sy} {sz}")
                if ok:
                    fo.write(json.dumps({"text": "\n".join(lines)}) + "\n")
                    n += 1; made += 1
    print(f"wrote {n} real K={a.k} layouts (extent<={a.max_extent}) -> {a.out}")


if __name__ == "__main__":
    main()
