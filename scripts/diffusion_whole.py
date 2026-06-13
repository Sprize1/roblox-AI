#!/usr/bin/env python3
"""3D-native set-diffusion over WHOLE GAMES (variable K) — the user's unit: a whole game is
coherent by construction. Each game = a set of <=KMAX boxes (x,y,z,sx,sy,sz) + a presence
channel for variable count. Augmented (Y-rot 90deg x4, mirror) so coherence is preserved while
multiplying data. Measures COHERENCE (checker deep-overlap) AND NOVELTY (voxel-occupancy NN to
train -> is it generating new games or memorizing the ~1774?).
"""
import sys, json, math, time, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from checker import deep_overlap_rate

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KMAX = 4096


def load_games(path, pmin=8, pmax=4096, max_extent=4000):
    games = []
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        parts = d.get("parts", [])
        if not (pmin <= len(parts) <= pmax):
            continue
        lo = [min(p["pos"][k] for p in parts) for k in range(3)]
        B = []
        ok = True
        for p in parts:
            x, y, z = (p["pos"][k] - lo[k] for k in range(3))
            sx, sy, sz = (max(1.0, abs(v)) for v in p["size"])
            if max(x, y, z, sx, sy, sz) > max_extent:
                ok = False; break
            B.append([x, y, z, sx, sy, sz])
        if ok and len(B) >= pmin:
            games.append(np.array(B, dtype=np.float32))
    return games


def augment(B):
    out = []
    x, y, z, sx, sy, sz = [B[:, i] for i in range(6)]
    for k in range(4):                                          # Y-rotations 90deg
        if k == 0: nx, nz, nsx, nsz = x, z, sx, sz
        elif k == 1: nx, nz, nsx, nsz = z, -x, sz, sx
        elif k == 2: nx, nz, nsx, nsz = -x, -z, sx, sz
        else: nx, nz, nsx, nsz = -z, x, sz, sx
        for mir in (False, True):
            mx = -nx if mir else nx
            g = np.stack([mx, y, nz, nsx, sy, nsz], 1).astype(np.float32)   # mirror flips pos, not size
            g[:, [0, 1, 2]] -= g[:, [0, 1, 2]].min(0)            # re-translate min->0
            out.append(g)
    return out


def to_tensor(games):
    X = np.zeros((len(games), KMAX, 7), dtype=np.float32)
    X[:, :, 6] = -1.0                                            # presence: -1 = pad
    for i, g in enumerate(games):
        n = min(len(g), KMAX)
        X[i, :n, :6] = g[:n]
        X[i, :n, 6] = 1.0
    return X


class LinBlock(nn.Module):
    """Non-causal LINEAR attention over the set of parts: O(n.d^2), sub-quadratic, no N^2 matrix.
    out_i = phi(q_i) . (sum_j phi(k_j) v_j^T) / (phi(q_i) . sum_j phi(k_j)),  phi = elu+1."""
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)                            # learned per-token gate: suppress padding in the KV sum
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, N, C = x.shape
        h1 = self.ln1(x)
        qkv = self.qkv(h1).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, h, N, d)
        g = torch.sigmoid(self.gate(h1)).reshape(B, 1, N, 1)  # (B,1,N,1) -- gate keys, not queries
        q = F.elu(q) + 1; k = (F.elu(k) + 1) * g               # gated key: low-presence tokens contribute ~0
        kv = k.transpose(-2, -1) @ v                           # (B, h, d, d) -- linear in N
        denom = (q * k.sum(dim=2, keepdim=True)).sum(-1, keepdim=True) + 1e-6
        a = (q @ kv) / denom                                   # (B, h, N, d)
        a = a.transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x


class Denoiser(nn.Module):
    def __init__(self, d=192, layers=5, heads=6):
        super().__init__()
        self.inp = nn.Linear(7, d)
        self.tproj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([LinBlock(d, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.out = nn.Linear(d, 7); self.d = d

    def temb(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * f[None]
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, x, t):
        h = self.inp(x) + self.tproj(self.temb(t))[:, None, :]
        for b in self.blocks:
            h = b(h)
        return self.out(self.lnf(h))


def voxel_feat(g, R=8):
    """coarse occupancy of box centers in the game's own bbox -> 512-bit signature for novelty NN."""
    pos = g[:, :3]
    span = pos.max(0) - pos.min(0) + 1e-6
    idx = np.clip(((pos - pos.min(0)) / span * R).astype(int), 0, R - 1)
    v = np.zeros((R, R, R), dtype=np.float32)
    for a, b, c in idx:
        v[a, b, c] = 1
    return v.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/structured_v3b.jsonl")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--gen", type=int, default=200)
    a = ap.parse_args()

    base = load_games(a.data)
    print(f"DEV={DEV} jeux entiers (<=128 parts, extent<=256): {len(base)}", flush=True)
    games = [ag for g in base for ag in augment(g)]
    print(f"après augmentation (rot+miroir): {len(games)} unités", flush=True)
    X = to_tensor(games)
    real_feats = np.stack([voxel_feat(g) for g in base])         # for novelty NN

    box = X[:, :, :6][X[:, :, 6] > 0]
    mean = box.mean(0); std = box.std(0) + 1e-6
    Xn = X.copy(); Xn[:, :, :6] = (Xn[:, :, :6] - mean) / std
    Xn = torch.tensor(Xn, device=DEV)
    mean_t = torch.tensor(mean, device=DEV); std_t = torch.tensor(std, device=DEV)

    betas = torch.linspace(1e-4, 0.02, a.T, device=DEV); alpha = 1 - betas; abar = torch.cumprod(alpha, 0)
    model = Denoiser().to(DEV)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t0 = time.time()
    for s in range(a.steps):
        idx = torch.randint(0, len(Xn), (a.bs,), device=DEV); x0 = Xn[idx]
        t = torch.randint(0, a.T, (a.bs,), device=DEV); noise = torch.randn_like(x0)
        xt = abar[t][:, None, None].sqrt() * x0 + (1 - abar[t])[:, None, None].sqrt() * noise
        loss = F.mse_loss(model(xt, t), noise)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 2000 == 0:
            print(f"  step {s}/{a.steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)

    import os; os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/whole_model.pt")      # save BEFORE the risky generation
    model.eval()
    gbs = 4                                                       # mini-batch gen: avoid OOM at large KMAX
    outs = []
    with torch.no_grad():
        for s0 in range(0, a.gen, gbs):
            b = min(gbs, a.gen - s0)
            x = torch.randn(b, KMAX, 7, device=DEV)
            for ti in reversed(range(a.T)):
                t = torch.full((b,), ti, device=DEV); pred = model(x, t)
                ac, ab = alpha[ti], abar[ti]
                x = (x - (1 - ac) / (1 - ab).sqrt() * pred) / ac.sqrt()
                if ti > 0:
                    x = x + betas[ti].sqrt() * torch.randn_like(x)
            outs.append(x.cpu().numpy())
    x = np.concatenate(outs, 0)
    pres = x[:, :, 6] > 0
    boxes6 = x[:, :, :6] * std + mean

    drs, nparts, novelty, saved = [], [], [], []
    for i in range(a.gen):
        sel = boxes6[i][pres[i]]
        if len(sel) < 2:
            continue
        parts = [{"pos": [float(b[0]), float(b[1]), float(b[2])],
                  "size": [max(1.0, abs(float(b[3]))), max(1.0, abs(float(b[4]))), max(1.0, abs(float(b[5])))],
                  "rot": [1, 0, 0, 0, 1, 0, 0, 0, 1]} for b in sel]
        drs.append(deep_overlap_rate(parts)); nparts.append(len(parts))
        f = voxel_feat(np.abs(sel))
        novelty.append(np.min(np.abs(real_feats - f).sum(1)))     # Hamming NN to train
        if len(saved) < 6:
            saved.append([[float(v) for v in b] for b in sel])
    # baseline: train-vs-train NN (how far real games are from each other)
    base_nn = []
    for i in range(min(300, len(real_feats))):
        dd = np.abs(real_feats - real_feats[i]).sum(1); dd[i] = 1e9
        base_nn.append(dd.min())
    real_dr = np.mean([deep_overlap_rate([{"pos": list(b[:3]), "size": [max(1, abs(v)) for v in b[3:]],
                       "rot": [1,0,0,0,1,0,0,0,1]} for b in g]) for g in base[:200]])

    print("\n========== JEU ENTIER (diffusion K-variable) ==========", flush=True)
    print(f"parts générées/jeu : {np.mean(nparts):.0f} (réel médiane ~{int(np.median([len(g) for g in base]))})")
    print(f"COHÉRENCE deep-overlap/part : généré {np.mean(drs):.3f}  vs réel {real_dr:.3f}")
    print(f"NOUVEAUTÉ (NN voxel au train): généré {np.mean(novelty):.1f}  vs train-train {np.mean(base_nn):.1f}")
    print(f"  -> si généré ~ train-train: nouveau ; si ~0: mémorisé", flush=True)
    json.dump({"generated": saved, "real": [[[float(v) for v in b] for b in g] for g in base[:6]]},
              open("data/whole_gen.json", "w"))
    print("saved -> data/whole_gen.json", flush=True)


if __name__ == "__main__":
    main()
