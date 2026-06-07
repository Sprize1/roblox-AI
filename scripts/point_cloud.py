#!/usr/bin/env python3
"""Unified point-cloud extraction from v3b (local only, no network).

Each game -> a single point cloud: points sampled on every object's surface, each point tagged
(x,y,z, r,g,b, material_id, object_id). Boxes sampled exactly (6 faces, pos+size+rotation).
MeshParts/Unions -> sampled on their BOUNDING BOX (true triangles aren't in the .rbxlx, only a
MeshId), so meshes degrade to boxes locally — flagged in stats.

Scripts = a SEPARATE SPARSE stream (most objects have none): per script, its referenced
object_ids (script.Parent + game.Workspace.<Name>/findFirstChild("<Name>") resolved to objects
present in the game). Matches reality: few scripts, sparse many-to-many refs over the geometry.
"""
import sys, json, re, argparse
from pathlib import Path
import numpy as np

MAT = {}                                                        # material name -> id (built on the fly)
WS = re.compile(r'(?:game\.)?[Ww]orkspace\.([A-Za-z_]\w*)')
FFC = re.compile(r'(?:[Ff]ind[Ff]irstChild|[Ww]aitForChild)\(["\'](\w+)["\']')


def mat_id(name):
    return MAT.setdefault(name or "Plastic", len(MAT))


def sample_box(c, size, R, m, rng):
    """m points on the surface of an oriented box (area-weighted faces) -> world coords (m,3)."""
    sx, sy, sz = (max(1e-3, s) for s in size)
    h = np.array([sx, sy, sz]) / 2.0
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy])
    faces = rng.choice(6, size=m, p=areas / areas.sum())
    loc = (rng.random((m, 3)) * 2 - 1) * h                      # random inside box
    for k, ax in enumerate([0, 0, 1, 1, 2, 2]):                 # pin the chosen face's axis to ±h
        sel = faces == k
        loc[sel, ax] = h[ax] * (1 if k % 2 == 0 else -1)
    return c + loc @ R.T                                        # R rows-> apply (R columns are world axes)


def extract_game(d, max_points=4096, min_pp=4, max_pp=48, density=0.04):
    parts = d.get("parts", [])
    if not parts:
        return None
    rng = np.random.default_rng(0)
    P, meta = [], []
    n_mesh = 0
    for oid, p in enumerate(parts):
        size = [abs(v) for v in p["size"]]
        is_mesh = p.get("class") in ("MeshPart", "UnionOperation") or p.get("meshid")
        n_mesh += int(bool(is_mesh))
        area = 2 * (size[0]*size[1] + size[1]*size[2] + size[0]*size[2])
        m = int(np.clip(area * density, min_pp, max_pp))
        c = np.array(p["pos"], float)
        R = np.array(p.get("rot", [1,0,0,0,1,0,0,0,1]), float).reshape(3, 3)
        pts = sample_box(c, size, R, m, rng)
        col = p.get("color", [0.64, 0.64, 0.64])
        mid = mat_id(p.get("material"))
        for q in pts:
            P.append([q[0], q[1], q[2], col[0], col[1], col[2], mid, oid])
    P = np.array(P, dtype=np.float32)
    if len(P) > max_points:                                    # subsample to budget
        P = P[rng.choice(len(P), max_points, replace=False)]
    # normalize positions to local frame (min corner -> 0), keep scale
    P[:, :3] -= P[:, :3].min(0)

    # sparse script stream
    name2id = {}
    for oid, p in enumerate(parts):
        nm = p.get("name", "")
        if nm and nm not in ("Part",):
            name2id.setdefault(nm, oid)                        # first occurrence
    scripts = []
    for s in d.get("scripts", []):
        src = s.get("source", "")
        if not src.strip():
            continue
        refs = set()
        if s.get("attach") is not None:
            refs.add(s["attach"])
        for nm in set(WS.findall(src)) | set(FFC.findall(src)):
            if nm in name2id:
                refs.add(name2id[nm])
        scripts.append({"len": len(src), "refs": sorted(refs)})
    return {"points": P, "n_parts": len(parts), "n_mesh": n_mesh, "scripts": scripts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", default="data/structured_v3b.jsonl")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    pts_per_game, mesh_frac, nscr, nref, withref = [], [], [], [], 0
    nscr_tot = 0
    games = 0
    with open(a.inp, encoding="utf-8") as f:
        for line in f:
            if games >= a.n:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            g = extract_game(d)
            if g is None:
                continue
            games += 1
            pts_per_game.append(len(g["points"]))
            mesh_frac.append(g["n_mesh"] / max(g["n_parts"], 1))
            for s in g["scripts"]:
                nscr_tot += 1
                nref.append(len(s["refs"]))
                withref += int(len(s["refs"]) > 0)
            nscr.append(len(g["scripts"]))

    pg = np.array(pts_per_game)
    print(f"=== POINT CLOUD (échantillon {games} jeux) ===")
    print(f"points/jeu       : médiane {int(np.median(pg))}, moy {pg.mean():.0f}, max {pg.max()}")
    print(f"part MeshPart/Union: {np.mean(mesh_frac)*100:.0f}% des parts (approximées en boîte)")
    print(f"matériaux distincts: {len(MAT)}")
    print(f"\n=== SCRIPTS (flux sparse) ===")
    print(f"scripts/jeu      : médiane {int(np.median(nscr))}, moy {np.mean(nscr):.1f}")
    print(f"scripts total    : {nscr_tot:,}")
    print(f"scripts avec >=1 ref objet : {withref/max(nscr_tot,1)*100:.0f}%")
    if nref:
        print(f"refs/script      : moy {np.mean(nref):.2f} (sparse, many-to-many confirmé)")
    # save one sample game for inspection
    with open(a.inp, encoding="utf-8") as f:
        for line in f:
            g = extract_game(json.loads(line))
            if g and len(g["points"]) > 500:
                np.save("data/pc_sample.npy", g["points"])
                print(f"\nexemple sauvé: data/pc_sample.npy  shape {g['points'].shape} "
                      f"(bbox {g['points'][:,:3].max(0).round(1)})")
                break


if __name__ == "__main__":
    main()
