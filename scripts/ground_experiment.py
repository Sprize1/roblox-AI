#!/usr/bin/env python3
"""Does checker-grounding measurably improve a generator? A/B, measured mechanically.

A char-level GPT generates box layouts. The EXACT geometric checker scores each by deep
interpenetration (parts jammed into each other). One round of expert-iteration:
  base -> generate a pool -> checker scores it ->
    F = fine-tune base on the LOW-overlap 40% (checker-filtered)
    U = fine-tune base on a RANDOM 40% (same count, same compute)
Then generate from F and U and measure deep-overlap on their NOVEL outputs.

Non-rigged: the validity numbers come from exact geometry on the models' own new generations,
not from anything I hand-pick. Anti-Goodhart guard: also report avg parts & size, so a model
that "cheats" by emptying the scene is caught.
"""
import sys, json, time, random, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_opstream import GPT
from checker import deep_overlap_rate

CHARS = ["<pad>", "<bos>", "<eos>"] + list("B0123456789 -\n")
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
BOS, EOS, PAD = STOI["<bos>"], STOI["<eos>"], STOI["<pad>"]
BLOCK = 320
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = DEV == "cuda" and torch.cuda.is_bf16_supported()


def enc(s):
    return [STOI[c] for c in s if c in STOI]


def dec(ids):
    return "".join(ITOS[i] for i in ids if i in ITOS and i not in (BOS, EOS, PAD))


def synth_layout(rng):
    """K boxes packed in a small volume -> frequent overlaps (the noisy prior to clean up)."""
    K = rng.randint(6, 12)
    lines = []
    for _ in range(K):
        x, y, z = rng.randint(-20, 20), rng.randint(0, 18), rng.randint(-20, 20)
        sx, sy, sz = rng.randint(3, 9), rng.randint(3, 9), rng.randint(3, 9)
        lines.append(f"B {x} {y} {z} {sx} {sy} {sz}")
    return "\n".join(lines)


def parse(text):
    parts = []
    for ln in text.split("\n"):
        t = ln.strip().split()
        if len(t) == 7 and t[0] == "B":
            try:
                v = [int(x) for x in t[1:]]
            except ValueError:
                continue
            sz = [max(1, abs(v[3])), max(1, abs(v[4])), max(1, abs(v[5]))]
            parts.append({"pos": [v[0], v[1], v[2]], "size": sz,
                          "rot": [1, 0, 0, 0, 1, 0, 0, 0, 1]})
    return parts


def make_ids(texts):
    ids = []
    for t in texts:
        ids.append(BOS); ids.extend(enc(t)); ids.append(EOS)
    return torch.tensor(ids, dtype=torch.long)


def train(model, data, steps, lr=3e-4, bs=64, log=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train(); t0 = time.time()
    n = (len(data) - 1) // BLOCK
    for s in range(steps):
        idx = torch.randint(0, n, (bs,))
        x = torch.stack([data[i * BLOCK:(i + 1) * BLOCK] for i in idx]).to(DEV)
        y = torch.stack([data[i * BLOCK + 1:(i + 1) * BLOCK + 1] for i in idx]).to(DEV)
        with torch.autocast(device_type=DEV, dtype=torch.bfloat16, enabled=BF16):
            _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"  [{log}] step {s}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)


@torch.no_grad()
def gen(model, nsamp, temp=0.9, maxnew=BLOCK):
    model.eval(); out = []
    for _ in range(nsamp):
        x = torch.tensor([[BOS]], device=DEV)
        for _ in range(maxnew):
            logits, _ = model(x[:, -BLOCK:])
            p = torch.softmax(logits[0, -1] / temp, -1)
            nxt = torch.multinomial(p, 1).item()
            if nxt == EOS:
                break
            x = torch.cat([x, torch.tensor([[nxt]], device=DEV)], 1)
        out.append(dec(x[0].tolist()))
    return out


def score_set(texts):
    """Return per-layout: deep-overlap rate, n_parts, mean size. Skip <2-part garbage."""
    rows = []
    for t in texts:
        parts = parse(t)
        if len(parts) < 2:
            continue
        dr = deep_overlap_rate(parts)
        ms = float(np.mean([np.mean(p["size"]) for p in parts]))
        rows.append((t, dr, len(parts), ms))
    return rows


def summarize(rows, label):
    if not rows:
        print(f"{label}: AUCUN layout parsable"); return None
    dr = np.mean([r[1] for r in rows]); npc = np.mean([r[2] for r in rows]); ms = np.mean([r[3] for r in rows])
    print(f"{label}: deep-overlap/part {dr:.3f} | parts {npc:.1f} | taille moy {ms:.1f} | n={len(rows)}")
    return dr, npc, ms


def new_model():
    return GPT(len(CHARS), BLOCK, n_layer=6, n_embd=256, n_head=8).to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-n", type=int, default=6000)
    ap.add_argument("--base-steps", type=int, default=3000)
    ap.add_argument("--pool", type=int, default=800)
    ap.add_argument("--ft-steps", type=int, default=1500)
    ap.add_argument("--eval", type=int, default=250)
    ap.add_argument("--out", default="models/ground_result.json")
    ap.add_argument("--real", default="", help="jsonl of real layouts; if set, use them as the base distribution")
    a = ap.parse_args()
    rng = random.Random(0); torch.manual_seed(0)
    print(f"DEV={DEV} bf16={BF16} vocab={len(CHARS)}", flush=True)

    if a.real:
        reals = [json.loads(l)["text"] for l in open(a.real, encoding="utf-8")]
        random.Random(0).shuffle(reals)
        base_texts = [reals[i % len(reals)] for i in range(a.base_n)]
        print(f"REAL layouts: {len(reals)} uniques -> base_n={a.base_n}", flush=True)
    else:
        base_texts = [synth_layout(rng) for _ in range(a.base_n)]   # synthetic noisy prior
    base_ref = summarize(score_set(base_texts[:400]), "DISTRIB DE BASE (synthétique)")

    base = new_model()
    print(f"params={sum(p.numel() for p in base.parameters())/1e6:.1f}M", flush=True)
    train(base, make_ids(base_texts), a.base_steps, log="base")
    base_state = {k: v.clone() for k, v in base.state_dict().items()}

    print("\n--- génération du pool depuis base ---", flush=True)
    pool = score_set(gen(base, a.pool))
    summarize(pool, "POOL (base)")
    if len(pool) < 50:
        print("pool trop petit -> abort"); json.dump({"error": "pool small"}, open(a.out, "w")); return

    band = [r for r in pool if 7 <= r[2] <= 11]       # match part-count -> isolate PLACEMENT, not count
    print(f"pool dans la bande 7-11 parts: {len(band)}/{len(pool)}", flush=True)
    band.sort(key=lambda r: r[1])                     # ascending deep-overlap
    k = int(len(band) * 0.4)
    F_texts = [r[0] for r in band[:k]]                # checker-filtered: LOW overlap, matched count
    U_texts = [r[0] for r in random.Random(1).sample(band, k)]  # random, same count + same band
    print(f"F-set (filtré checker) {len(F_texts)} | U-set (aléatoire) {len(U_texts)}  "
          f"| parts F={np.mean([parse(t).__len__() for t in F_texts]):.1f} U={np.mean([parse(t).__len__() for t in U_texts]):.1f}", flush=True)

    res = {"base_ref": base_ref}
    for tag, texts in (("F", F_texts), ("U", U_texts)):
        m = new_model(); m.load_state_dict(base_state)
        train(m, make_ids(texts), a.ft_steps, log=f"ft-{tag}")
        ev = summarize(score_set(gen(m, a.eval)), f"EVAL {tag}")
        res[tag] = ev

    print("\n========== VERDICT ==========", flush=True)
    if res.get("F") and res.get("U"):
        fdr, fpc, _ = res["F"]; udr, upc, _ = res["U"]
        print(f"deep-overlap/part :  F={fdr:.3f}  vs  U={udr:.3f}   (plus bas = mieux)")
        print(f"parts/layout      :  F={fpc:.1f}    vs  U={upc:.1f}     (anti-triche: doivent rester proches)")
        better = (udr - fdr) / max(udr, 1e-9) * 100
        print(f"=> F réduit l'interpénétration de {better:.0f}% vs U", flush=True)
    json.dump(res, open(a.out, "w"))
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
