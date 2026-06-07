#!/usr/bin/env python3
"""Serialize a game into ONE unified op-stream: geometry + lowered scripts, same vocabulary.

  - Parts (z-ordered, reindexed #0..#N) -> CREATE ops  (static workspace = t=0).
  - Each attached script -> ATTACH #r { lowered op-stream } where `script.Parent` is RESOLVED
    to `#r` -> the geometry<->logic binding edge is now an explicit pointer in the same stream.
    A runtime `CREATE` (Instance.new) is the SAME token as a static part's CREATE.

`serialize_game(d)` -> (text, stats). CLI prints sample game streams to eyeball.
"""
import sys, json, re, argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_demo_chunks import morton
from luau_lower import lower

BLOCK_ONLY = {"BLOCK_END", "ELSE", "DO", "REPEAT", "BREAK"}
WS_REF = re.compile(r'((?:game\.)?[Ww]orkspace\.)([A-Za-z_]\w*)')              # game.Workspace.Lava
NAME_CALL = re.compile(r'((?:[Ff]ind[Ff]irstChild|[Ww]aitForChild)\(["\'])(\w+)(["\']\))')  # findFirstChild("Lava")


def geom_line(rank, p):
    x, y, z = (round(v) for v in p["pos"])
    sz = "x".join(str(round(v, 1)) for v in p["size"])
    s = f"CREATE {p['shape']} #{rank} @{x},{y},{z} {sz}"
    if p.get("material") and p["material"] != "Plastic":
        s += f" {p['material']}"
    nm = p.get("name", "")
    if nm and nm not in ("Part", p["shape"]):
        s += f' "{nm[:24]}"'
    return s


def resolve(text, r, name2rank):
    """Bind references to part ids. script.Parent -> #r (recency/trivial edge). Cross-part refs
    by NAME (game.Workspace.Lava, findFirstChild("Lava")) -> keep name + append #id (the NON-trivial
    edge: model must recall Lava=#id from its CREATE line)."""
    if r is not None:
        text = text.replace("script.Parent", f"#{r}")
    text = WS_REF.sub(lambda m: f"{m.group(1)}{m.group(2)}#{name2rank[m.group(2)]}"
                      if m.group(2) in name2rank else m.group(0), text)
    text = NAME_CALL.sub(lambda m: f"{m.group(0)}#{name2rank[m.group(2)]}"
                         if m.group(2) in name2rank else m.group(0), text)
    return text


def emit_script(out, s, r, max_ops, name2rank):
    out.append(f"ATTACH #{r}" if r is not None else "SCRIPT")
    ops, _, _ = lower(s["source"])
    n = 0
    for k, t in ops[:max_ops]:
        out.append(k if k in BLOCK_ONLY else f"{k} {resolve(t, r, name2rank)}")
        n += 1
    if len(ops) > max_ops:
        out.append("TRUNC")
    out.append("END")
    return n


def serialize_game(d, max_parts=500, max_scripts=40, max_script_ops=140):
    parts = d.get("parts", [])
    if not parts:
        return "", {}
    lo = [min(p["pos"][k] for p in parts) for k in range(3)]
    order = sorted(range(len(parts)), key=lambda i: morton(parts[i], lo, 4))[:max_parts]
    rank = {parts[i]["id"]: r for r, i in enumerate(order)}

    namecount = Counter(parts[i].get("name", "") for i in order)                # unique names -> #rank
    name2rank = {parts[i].get("name", ""): r for r, i in enumerate(order)
                 if parts[i].get("name") and namecount[parts[i].get("name", "")] == 1
                 and parts[i].get("name") not in ("Part",)}

    by_rank, unattached = {}, []                      # interleave: script next to its part's CREATE
    for s in d.get("scripts", []):
        if not s.get("source", "").strip():
            continue
        r = rank.get(s.get("attach"))
        (unattached if r is None else by_rank.setdefault(r, [])).append(s)

    out = [f"[GAME {d.get('id', '')[:48]}]"]
    n_ops = n_scr = 0
    for r, i in enumerate(order):
        out.append(geom_line(r, parts[i]))
        for s in by_rank.get(r, []):                  # attached scripts immediately after the part
            if n_scr >= max_scripts:
                break
            n_ops += emit_script(out, s, r, max_script_ops, name2rank); n_scr += 1
    for s in unattached:                              # part-less scripts at the end
        if n_scr >= max_scripts:
            break
        n_ops += emit_script(out, s, None, max_script_ops, name2rank); n_scr += 1
    out.append("[/GAME]")
    text = "\n".join(out)
    return text, {"parts": len(order), "scripts": n_scr, "script_ops": n_ops, "chars": len(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", default="data/structured_v3.jsonl")
    ap.add_argument("--show", type=int, default=2, help="how many sample games to print")
    ap.add_argument("--min-parts", type=int, default=8)
    ap.add_argument("--max-parts", type=int, default=40)
    ap.add_argument("--min-scripts", type=int, default=2)
    a = ap.parse_args()
    shown = 0
    with open(a.inp, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            np = len(d.get("parts", []))
            ns = sum(1 for s in d.get("scripts", []) if s.get("attach") is not None and s.get("source", "").strip())
            if not (a.min_parts <= np <= a.max_parts and ns >= a.min_scripts):
                continue
            text, st = serialize_game(d)
            print(f"\n########## GAME ({st}) ##########")
            print(text[:2600])
            shown += 1
            if shown >= a.show:
                break


if __name__ == "__main__":
    main()
