#!/usr/bin/env python3
"""3D-native set-diffusion over box primitives — tests whether a SPATIAL inductive bias learns
geometric coherence from few games better than the text model.

Each layout = a set of K boxes, each a continuous vector (x,y,z,sx,sy,sz). A DDPM with a
transformer denoiser (self-attention over the boxes -> reasons about relative 3D positions)
learns the distribution. We then sample novel layouts and measure coherence with the EXACT
checker (deep-overlap) + part sizes (anti-cheat). Head-to-head vs the text baseline (0.739
deep-overlap) and the real distribution (0.333).
"""
import sys, json, math, time, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from checker import deep_overlap_rate

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 10


def load(path):
    sets = []
    for line in open(path, encoding="utf-8"):
        boxes = []
        for ln in json.loads(line)["text"].split("\n"):
            t = ln.split()
            if len(t) == 7 and t[0] == "B":
                boxes.append([float(x) for x in t[1:]])
        if len(boxes) == K:
            sets.append(boxes)
    return np.array(sets, dtype=np.float32)                     # (N, K, 6)


class Denoiser(nn.Module):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__()
        self.inp = nn.Linear(6, d)
        self.tproj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, 6)
        self.d = d

    def temb(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * f[None]
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, x, t):
        h = self.inp(x) + self.tproj(self.temb(t))[:, None, :]
        return self.out(self.tr(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_layout.jsonl")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--gen", type=int, default=400)
    a = ap.parse_args()

    X = load(a.data)
    print(f"DEV={DEV} layouts={len(X)} shape={X.shape[1:]}", flush=True)
    mean = X.reshape(-1, 6).mean(0); std = X.reshape(-1, 6).std(0) + 1e-6
    Xn = torch.tensor((X - mean) / std, device=DEV)
    mean_t = torch.tensor(mean, device=DEV); std_t = torch.tensor(std, device=DEV)

    betas = torch.linspace(1e-4, 0.02, a.T, device=DEV)
    alpha = 1 - betas; abar = torch.cumprod(alpha, 0)

    model = Denoiser().to(DEV)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t0 = time.time()
    for s in range(a.steps):
        idx = torch.randint(0, len(Xn), (a.bs,), device=DEV)
        x0 = Xn[idx]
        t = torch.randint(0, a.T, (a.bs,), device=DEV)
        noise = torch.randn_like(x0)
        xt = abar[t][:, None, None].sqrt() * x0 + (1 - abar[t])[:, None, None].sqrt() * noise
        pred = model(xt, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0:
            print(f"  step {s}/{a.steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)

    # DDPM sampling
    model.eval()
    with torch.no_grad():
        x = torch.randn(a.gen, K, 6, device=DEV)
        for ti in reversed(range(a.T)):
            t = torch.full((a.gen,), ti, device=DEV)
            pred = model(x, t)
            ac, ab = alpha[ti], abar[ti]
            mean_x = (x - (1 - ac) / (1 - ab).sqrt() * pred) / ac.sqrt()
            x = mean_x + (betas[ti].sqrt() * torch.randn_like(x) if ti > 0 else 0)
        gen = (x * std_t + mean_t).cpu().numpy()

    # evaluate coherence with the exact checker
    drs, sizes = [], []
    for layout in gen:
        parts = [{"pos": [float(b[0]), float(b[1]), float(b[2])],
                  "size": [max(1.0, abs(float(b[3]))), max(1.0, abs(float(b[4]))), max(1.0, abs(float(b[5])))],
                  "rot": [1, 0, 0, 0, 1, 0, 0, 0, 1]} for b in layout]
        drs.append(deep_overlap_rate(parts))
        sizes.append(np.mean([np.mean(p["size"]) for p in parts]))
    real_size = float(np.mean([np.mean(np.abs(b[3:])) for L in X for b in L]))
    print("\n========== 3D-NATIF (diffusion sur primitives) ==========", flush=True)
    print(f"deep-overlap/part : {np.mean(drs):.3f}   (réel={0.333}, texte-baseline={0.739} ; plus bas=mieux)")
    print(f"taille moy générée: {np.mean(sizes):.1f}   (réel={real_size:.1f} ; anti-triche: doit rester proche)")
    print(f"=> 3D-natif vs texte: {(0.739-np.mean(drs))/0.739*100:.0f}% moins d'interpénétration que le texte")

    def boxes(layout):
        return [[float(b[0]), float(b[1]), float(b[2]),
                 max(1.0, abs(float(b[3]))), max(1.0, abs(float(b[4]))), max(1.0, abs(float(b[5])))]
                for b in layout]
    out = {"generated": [boxes(gen[i]) for i in range(min(6, len(gen)))],
           "real": [boxes(X[i]) for i in range(min(4, len(X)))]}
    json.dump(out, open("data/diff_gen.json", "w"))
    print("saved -> data/diff_gen.json (premiers échantillons, non cherry-pické)", flush=True)


if __name__ == "__main__":
    main()
