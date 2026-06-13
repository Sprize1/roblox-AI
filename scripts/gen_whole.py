#!/usr/bin/env python3
"""Generate a few whole-game layouts from the saved diffusion_whole model (KMAX=1024) for viz.
Recomputes the box mean/std from the same data, loads models/whole_model.pt, samples N games."""
import sys, json
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from diffusion_whole import load_games, augment, to_tensor, Denoiser, KMAX, DEV

N = 6
base = load_games("data/structured_v3b.jsonl")
games = [ag for g in base for ag in augment(g)]
X = to_tensor(games)
box = X[:, :, :6][X[:, :, 6] > 0]
mean = torch.tensor(box.mean(0), device=DEV); std = torch.tensor(box.std(0) + 1e-6, device=DEV)
print(f"games={len(base)} units={len(games)}", flush=True)

T = 200
betas = torch.linspace(1e-4, 0.02, T, device=DEV); alpha = 1 - betas; abar = torch.cumprod(alpha, 0)
model = Denoiser().to(DEV)
model.load_state_dict(torch.load("models/whole_model.pt", map_location=DEV)); model.eval()

with torch.no_grad():
    x = torch.randn(N, KMAX, 7, device=DEV)
    for ti in reversed(range(T)):
        t = torch.full((N,), ti, device=DEV)
        pred = model(x, t)
        ac, ab = alpha[ti], abar[ti]
        x = (x - (1 - ac) / (1 - ab).sqrt() * pred) / ac.sqrt()
        if ti > 0:
            x = x + betas[ti].sqrt() * torch.randn_like(x)
    x = x.cpu().numpy()
pres = x[:, :, 6] > 0
boxes6 = x[:, :, :6] * std.cpu().numpy() + mean.cpu().numpy()


def to_boxes(layout):
    return [[float(b[0]), float(b[1]), float(b[2]),
             max(1.0, abs(float(b[3]))), max(1.0, abs(float(b[4]))), max(1.0, abs(float(b[5])))]
            for b in layout]


gen = [to_boxes(boxes6[i][pres[i]]) for i in range(N)]
real = [to_boxes(g) for g in base[:6]]
json.dump({"generated": gen, "real": real}, open("data/whole_gen.json", "w"))
print("parts/jeu généré:", [len(g) for g in gen], flush=True)
print("saved -> data/whole_gen.json", flush=True)
