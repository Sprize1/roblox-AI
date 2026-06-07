#!/usr/bin/env python3
"""Voxelize Roblox builds -> 3D block grid (Minecraft-style) for the VQ-VAE/AR pipeline.

v1: axis-aligned AABB rasterization; block-type = material id; rotation ignored (counted).
This is the FEASIBILITY check: at what voxel size do small games fit a <=MAXDIM^3 grid,
and how dense/usable is the result?
"""
import sys, json
import numpy as np

# block-type vocabulary (voxel value = index; 0 = empty)
MATS = ["_empty", "Plastic", "SmoothPlastic", "Wood", "WoodPlanks", "Slate", "Concrete",
        "Brick", "Metal", "Grass", "Sand", "Ice", "Glass", "Neon", "Marble", "Granite",
        "Cobblestone", "DiamondPlate", "CorrodedMetal", "Foil"]
MAT_ID = {m: i for i, m in enumerate(MATS)}
IDENT = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def bbox(parts):
    lo = np.array([min(p["pos"][k] - p["size"][k] / 2 for p in parts) for k in range(3)])
    hi = np.array([max(p["pos"][k] + p["size"][k] / 2 for p in parts) for k in range(3)])
    return lo, hi


def voxelize(parts, vox, lo, dims):
    grid = np.zeros(dims, dtype=np.uint8)
    dmax = np.array(dims)
    for p in parts:
        c, s = np.array(p["pos"]), np.array(p["size"]) / 2
        a = np.clip(np.floor((c - s - lo) / vox).astype(int), 0, dmax - 1)
        b = np.clip(np.ceil((c + s - lo) / vox).astype(int), 0, dmax)
        grid[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = MAT_ID.get(p.get("material", "Plastic"), 1) or 1
    return grid


def main():
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/structured_v3.jsonl"
    MAXDIM = 64
    shown, total_small = 0, 0
    for line in open(fp, encoding="utf-8", errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        pc = d.get("parts_count", 0)
        if pc < 20 or pc > 300:
            continue
        total_small += 1
        if shown >= 10:
            continue
        parts = d.get("parts", [])
        if not parts:
            continue
        lo, hi = bbox(parts)
        ext = hi - lo
        vox = max(ext.max() / MAXDIM, 1.0)                       # voxel size so longest axis <= MAXDIM
        dims = tuple(min(int(np.ceil(e / vox)) + 1, MAXDIM) for e in ext)
        g = voxelize(parts, vox, lo, dims)
        occ = (g > 0).sum() / g.size * 100
        nrot = sum(1 for p in parts if p.get("rot") != IDENT)
        print(f"parts={pc:4} grid={str(dims):16} vox={vox:5.1f}stud occ={occ:5.1f}%  rotated_parts={nrot}/{pc}")
        shown += 1
    print(f"\ntotal small games (20..300 parts): {total_small}")


if __name__ == "__main__":
    main()
