#!/usr/bin/env python3
"""Synthetic games with DENSE, controllable cross-part binding — to test whether the binding
is learnable AT ALL (the real-data signal was ~800 examples, unmeasurable).

Each game: K parts with UNIQUE names (#0..#K-1). Scripts reference OTHER parts by name as
`workspace.<name>#<id>`. The name->id map is RANDOM PER GAME (a name maps to different ids in
different games) -> the model can't memorize globally, it must resolve from THIS game's CREATE
lines = in-context binding. Geometry is all up front, so the recency baseline = always the last
id = 1/K accuracy. A model scoring >> 1/K has learned real name->id binding.
"""
import json, random, argparse

SYLL = [c + v for c in "bcdfgklmnprstvz" for v in "aeiou"]          # 75 syllables
POOL = sorted({a + b for a in SYLL for b in SYLL})                  # ~5600 two-syllable names


def make_game(gid, kmin, kmax, scripts, refs, rng):
    K = rng.randint(kmin, kmax)
    names = rng.sample(POOL, K)                                     # unique names this game
    out = [f"[GAME synth{gid}]"]
    for i in range(K):
        out.append(f'CREATE Block "{names[i]}" #{i}')           # name BEFORE id -> forward-induction friendly
    for _ in range(scripts):
        r = rng.randrange(K)
        out.append(f"ATTACH #{r}")
        out.append(f"ON #{r}.Touched")
        for _ in range(refs):
            j = rng.randrange(K)                                   # random target (not recency)
            if rng.random() < 0.5:
                out.append(f"SET workspace.{names[j]}#{j}.Transparency = 0")
            else:
                out.append(f"CALL workspace.{names[j]}#{j}:Destroy()")
        out.append("BLOCK_END")
        out.append("END")
    out.append("[/GAME]")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--kmin", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=24)
    ap.add_argument("--scripts", type=int, default=8)
    ap.add_argument("--refs", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out-prefix", default="data/synth")
    a = ap.parse_args()
    rng = random.Random(0)
    games = [make_game(i, a.kmin, a.kmax, a.scripts, a.refs, rng) for i in range(a.n)]
    nval = max(1, int(a.n * a.val_frac))
    for name, part in (("val", games[:nval]), ("train", games[nval:])):
        p = f"{a.out_prefix}_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for g in part:
                f.write(json.dumps({"text": g}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} games -> {p}")
    refs_total = a.n * a.scripts * a.refs
    print(f"~{refs_total:,} cross-part refs total | K in [{a.kmin},{a.kmax}] -> recency baseline ~{100/((a.kmin+a.kmax)/2):.0f}%")


if __name__ == "__main__":
    main()
