#!/usr/bin/env python3
"""Geometric checker — computes exact spatial facts from part coords, where naive reasoning errs.

Overlap = exact oriented-box intersection (Separating Axis Theorem), NOT center-distance or AABB.
Includes self-tests showing where the naive shortcuts (AABB / center-distance) DISAGREE with the
exact answer, then runs at scale on real games (deterministic, fast).
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np


def part_obb(p):
    c = np.array(p["pos"], float)
    h = np.array(p["size"], float) / 2.0
    R = np.array(p.get("rot", [1, 0, 0, 0, 1, 0, 0, 0, 1]), float).reshape(3, 3)
    A = R[:, 0], R[:, 1], R[:, 2]            # local axes in world = columns
    return c, h, np.array(A)


def aabb_half(c, h, A):
    return np.array([sum(h[i] * abs(A[i][k]) for i in range(3)) for k in range(3)])


def aabb_overlap(p1, p2):                    # naive broad-phase test
    c1, h1, A1 = part_obb(p1); c2, h2, A2 = part_obb(p2)
    e1, e2 = aabb_half(c1, h1, A1), aabb_half(c2, h2, A2)
    return np.all(np.abs(c1 - c2) <= e1 + e2 + 1e-6)


def center_far(p1, p2, thresh=3.0):          # what a quick reasoner might use: "centers far -> fine"
    return np.linalg.norm(np.array(p1["pos"]) - np.array(p2["pos"])) > thresh


def obb_overlap(p1, p2, eps=1e-6):           # EXACT (SAT over 15 axes)
    c1, h1, A1 = part_obb(p1); c2, h2, A2 = part_obb(p2)
    t = c2 - c1
    axes = list(A1) + list(A2)
    for a in A1:
        for b in A2:
            cr = np.cross(a, b)
            if np.linalg.norm(cr) > eps:
                axes.append(cr / np.linalg.norm(cr))
    for L in axes:
        rA = sum(h1[i] * abs(np.dot(A1[i], L)) for i in range(3))
        rB = sum(h2[i] * abs(np.dot(A2[i], L)) for i in range(3))
        if abs(np.dot(t, L)) > rA + rB + eps:
            return False                     # found a separating axis -> no overlap
    return True


def obb_pen(p1, p2, eps=1e-6):
    """Penetration depth (studs) of two oriented boxes; 0 if separated. SAT min-margin."""
    c1, h1, A1 = part_obb(p1); c2, h2, A2 = part_obb(p2); t = c2 - c1
    axes = list(A1) + list(A2)
    for a in A1:
        for b in A2:
            cr = np.cross(a, b)
            if np.linalg.norm(cr) > eps:
                axes.append(cr / np.linalg.norm(cr))
    mind = 1e9
    for L in axes:
        rA = sum(h1[i] * abs(np.dot(A1[i], L)) for i in range(3))
        rB = sum(h2[i] * abs(np.dot(A2[i], L)) for i in range(3))
        m = rA + rB - abs(np.dot(t, L))
        if m < 0:
            return 0.0
        mind = min(mind, m)
    return mind


def deep_overlap_rate(parts, thresh=0.6):
    """Mean deep-interpenetration count per part (parts jammed >thresh studs into each other;
    resting-on / welds ~0 don't count). Lower = cleaner. Computed exactly, no opinion."""
    n = len(parts)
    if n < 2:
        return 0.0
    obbs = [part_obb(p) for p in parts]
    aabbs = [(c, aabb_half(c, h, A)) for c, h, A in obbs]
    deep = 0
    for i in range(n):
        ci, ei = aabbs[i]
        for j in range(i + 1, n):
            cj, ej = aabbs[j]
            if np.all(np.abs(ci - cj) <= ei + ej) and obb_pen(parts[i], parts[j]) > thresh:
                deep += 1
    return deep / n


def self_test():
    print("=== SELF-TEST : naïf vs exact ===")
    # Case 1: thin bar rotated 45° about Y, a small box near a CORNER of its AABB but off the bar.
    s = 0.70710678
    barR = [s, 0, s, 0, 1, 0, -s, 0, s]
    A = {"pos": [0, 0, 0], "size": [12, 1, 1], "rot": barR}
    B = {"pos": [4, 0, 4], "size": [1, 1, 1], "rot": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    print(f"Cas 1 (boîte près du COIN de l'AABB d'une barre diagonale):")
    print(f"  AABB (naïf)      -> {'OVERLAP' if aabb_overlap(A, B) else 'séparés'}")
    print(f"  SAT  (exact)     -> {'OVERLAP' if obb_overlap(A, B) else 'séparés'}   <- la vérité")

    # Case 2: two long bars CROSSING, centers far apart -> a quick 'centers far, fine' is WRONG.
    A2 = {"pos": [-4, 0, 0], "size": [20, 1, 1], "rot": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    B2 = {"pos": [4, 0, 5], "size": [1, 1, 20], "rot": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    d = np.linalg.norm(np.array(A2["pos"]) - np.array(B2["pos"]))
    print(f"Cas 2 (deux barres qui se croisent, centres distants de {d:.1f}):")
    print(f"  'centres loin'   -> {'séparés (FAUX)' if center_far(A2, B2) else 'overlap'}")
    print(f"  SAT  (exact)     -> {'OVERLAP' if obb_overlap(A2, B2) else 'séparés'}   <- la vérité")


def run_real(path, n_games, pmin, pmax):
    print(f"\n=== RÉEL : overlaps exacts sur de vrais jeux ({path}) ===")
    shown = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if shown >= n_games:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            parts = d.get("parts", [])
            if not (pmin <= len(parts) <= pmax):
                continue
            t0 = time.time()
            # broad-phase: precompute AABBs, only run SAT on AABB-overlapping pairs
            obbs = [part_obb(p) for p in parts]
            aabbs = [(c, aabb_half(c, h, A)) for (c, h, A) in obbs]
            npair = nover = 0
            example = None
            for i in range(len(parts)):
                ci, ei = aabbs[i]
                for j in range(i + 1, len(parts)):
                    cj, ej = aabbs[j]
                    if np.all(np.abs(ci - cj) <= ei + ej):
                        npair += 1
                        if obb_overlap(parts[i], parts[j]):
                            nover += 1
                            if example is None:
                                example = (i, j)
            dt = time.time() - t0
            ex = ""
            if example:
                i, j = example
                ex = f" | ex: '{parts[i].get('name','?')[:14]}' X '{parts[j].get('name','?')[:14]}'"
            print(f"  {d.get('id','?')[:34]:34} {len(parts):4}p  {npair:5} paires testées -> {nover:5} overlaps exacts  {dt*1000:.0f}ms{ex}")
            shown += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/structured_v3b.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--pmin", type=int, default=40)
    ap.add_argument("--pmax", type=int, default=400)
    a = ap.parse_args()
    self_test()
    run_real(a.data, a.n, a.pmin, a.pmax)


if __name__ == "__main__":
    main()
