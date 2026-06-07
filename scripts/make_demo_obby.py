#!/usr/bin/env python3
"""
Build a small OBBY demo dataset from structured_v3.

Filters obby-ish games, caps size, and serializes each to a COMPACT TEXT document
(layout + scripts + part->script coupling) for a LoRA fine-tune proof-of-concept.
Text-only — this is the demo path, NOT the geometry-modality model. See docs/hybrid_3d_llm_design.md.

Output: jsonl of {"id", "text"} ready for LoRA SFT.
Usage:
  python scripts/make_demo_obby.py [in.jsonl] [out.jsonl] [--min-parts N] [--max-parts N] [--max-games N] [--max-src N]
"""
import json, argparse

OBBY_KW = ("checkpoint", "killbrick", "kill brick", "killpart", "lava", "obby",
           "respawn", "resetonspawn", "stage", "wins", "winpart")
DEFAULT_MAT = "Plastic"
IDENT = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def is_obby(d):
    if "obby" in d.get("id", "").lower():
        return True
    if not any(p["class"] == "SpawnLocation" for p in d.get("parts", [])):
        return False
    src = " ".join(s.get("source", "")[:2000].lower() for s in d.get("scripts", [])[:25])
    return sum(k in src for k in OBBY_KW) >= 2


def _suffix(p):
    s = ""
    if p.get("material") and p["material"] != DEFAULT_MAT:
        s += f" {p['material']}"
    if p.get("rot") and p["rot"] != IDENT:
        s += " rot[" + ",".join(str(round(r, 2)) for r in p["rot"]) + "]"
    if p.get("meshid"):
        s += " mesh"
    nm = p.get("name", "")
    if nm and nm not in ("Part", p["shape"]):
        s += f' "{nm[:24]}"'
    return s


def _attrs(p):
    """Everything except position — parts in a ROW run must share this."""
    return (p["shape"], tuple(round(v, 1) for v in p["size"]), _suffix(p))


def serialize(d, max_parts, max_src):
    parts = d.get("parts", [])[:max_parts]
    out = [f"[GAME: {d.get('id', '')[:60]}]", f"[PARTS {len(parts)}]"]
    i, n = 0, len(parts)
    while i < n:
        p = parts[i]
        sz = "x".join(str(round(v, 1)) for v in p["size"])
        x, y, z = (round(v) for v in p["pos"])
        # arithmetic run: same attrs + constant integer pos step, length >= 3 -> collapse to ROW
        step, j = None, i
        if i + 1 < n and _attrs(parts[i + 1]) == _attrs(p):
            step = [round(parts[i + 1]["pos"][k] - p["pos"][k]) for k in range(3)]
            j = i + 1
            while (j + 1 < n and _attrs(parts[j + 1]) == _attrs(p)
                   and [round(parts[j + 1]["pos"][k] - parts[j]["pos"][k]) for k in range(3)] == step):
                j += 1
        run = j - i + 1
        if run >= 3 and step and step != [0, 0, 0]:
            out.append(f"ROW{run} {p['shape']} {sz} @{x},{y},{z} step {step[0]},{step[1]},{step[2]}" + _suffix(p))
            i = j + 1
        else:
            out.append(f"P{p['id']} {p['shape']} @{x},{y},{z} {sz}" + _suffix(p))
            i += 1
    scripts = d.get("scripts", [])
    out.append(f"[SCRIPTS {len(scripts)}]")
    for s in scripts:
        att = s.get("attach")
        out.append(f"[{s.get('class', 'Script')} {s.get('name', '')[:24]}" +
                   (f" -> P{att}]" if att is not None else "]"))
        out.append(s.get("source", "")[:max_src].strip())
        out.append("[/Script]")
    out.append("[/GAME]")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", default="data/structured_v3.jsonl")
    ap.add_argument("out", nargs="?", default="data/demo_obby.jsonl")
    ap.add_argument("--min-parts", type=int, default=20)
    ap.add_argument("--max-parts", type=int, default=250)
    ap.add_argument("--min-scripts", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=28000, help="skip games whose serialized text exceeds this (keeps the demo short)")
    ap.add_argument("--obby", action="store_true", help="restrict to obby-ish games only")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--max-src", type=int, default=1200)
    a = ap.parse_args()

    kept = 0
    lens = []
    by_name = 0
    with open(a.inp, encoding="utf-8", errors="ignore") as f, open(a.out, "w", encoding="utf-8") as o:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            pc = d.get("parts_count", 0)
            if pc < a.min_parts or pc > a.max_parts:
                continue
            if d.get("scripts_count", 0) < a.min_scripts:
                continue
            if a.obby and not is_obby(d):
                continue
            text = serialize(d, a.max_parts, a.max_src)
            if len(text) > a.max_chars:                       # keep the demo set uniformly short
                continue
            o.write(json.dumps({"id": d.get("id", ""), "text": text}, ensure_ascii=False) + "\n")
            kept += 1
            lens.append(len(text))
            if "obby" in d.get("id", "").lower():
                by_name += 1
            if a.max_games and kept >= a.max_games:
                break

    if kept:
        lens.sort()
        med = lens[len(lens) // 2]
        p90 = lens[int(len(lens) * 0.9)]
        print(f"kept {kept} obby games ({by_name} by name, {kept - by_name} by signals) -> {a.out}")
        print(f"text chars: median={med:,} (~{med // 4} tok)  p90={p90:,} (~{p90 // 4} tok)  max={lens[-1]:,} (~{lens[-1] // 4} tok)")
    else:
        print("no games matched - broaden the filter (relax keywords or size cap)")


if __name__ == "__main__":
    main()
