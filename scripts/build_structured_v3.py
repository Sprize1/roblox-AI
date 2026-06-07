#!/usr/bin/env python3
"""
Structured game dataset v3 — geometry-complete records for the hybrid 3D-LLM.

Fixes the 3 losses of build_structured_v2.py:
  - rotation (full CFrame matrix R00..R22 — v2 kept only position)
  - hierarchy (each part's parent Model/Folder/Part is recorded)
  - shape       (primitive enum + class; MeshPart/legacy mesh -> meshId reference)
Plus: full script source (no 3000-char cut), no duplicate code/text field.

Emits one JSON record per place (NOT flat text). Downstream a collator maps each
record to the mixed-token stream (STRUCT/GEO/REF/TEXT). See docs/hybrid_3d_llm_design.md.

Record schema (per place):
  {
    "id": str,
    "parts": [ {
        "id": int,                 # sequential, stable within the place
        "name": str, "class": str, "shape": str,   # shape in {Block,Ball,Cylinder,Wedge,CornerWedge,Truss,Mesh,Union}
        "pos":  [x,y,z],           # absolute (collator can relativize to parent)
        "size": [x,y,z],
        "rot":  [9 floats],        # 3x3 rotation matrix, row-major; identity if absent
        "color":[r,g,b],           # 0..1
        "material": str,
        "anchored": bool, "cancollide": bool,
        "meshid": str,             # "" for primitives; rbxassetid for MeshPart / legacy SpecialMesh
        "parent_part": int|None,   # parent part id, if the parent is itself a part
        "parent_name": str, "parent_class": str   # immediate parent (Model/Folder/...) for grouping
    }, ... ],
    "scripts": [ {
        "id": int, "name": str, "class": str,
        "source": str,             # FULL source
        "attach": int|None,        # part id the script is parented to (else None)
        "requires": [str, ...]
    }, ... ],
    "parts_count": int, "scripts_count": int,
    "rot_ok": bool                 # False if parsed via a path that drops rotation (binary / streaming)
  }

Usage:
  python scripts/build_structured_v3.py <input_dir> [output.jsonl] [limit]
Defaults: input ../RobloxRBXLArchive, output data/structured_v3.jsonl, limit 100000
"""
import json, sys, gc, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))                                   # rbxlx_parser.py (same dir)
sys.path.insert(0, str((Path(__file__).parent.parent.parent / "Roblox-rbxl-extractor" / "src").resolve()))

from rbxlx_parser import parse_file_fromstring, parse_file, _get_brick_color
try:
    from rbxl_extractor.rbx_binary_parser import parse as parse_rbxl              # binary .rbxl (rotation limited)
except Exception:
    parse_rbxl = None

PART_CLASSES = {"Part", "MeshPart", "WedgePart", "CornerWedgePart", "TrussPart",
                "Seat", "VehicleSeat", "SpawnLocation", "UnionOperation"}
SCRIPT_CLASSES = {"Script", "LocalScript", "ModuleScript"}
MESH_CLASSES = {"SpecialMesh", "FileMesh"}                                        # legacy mesh children of a Part
# structural containers: a script under these is NOT bound to a single part -> stop the climb.
NONCLIMB = {"Model", "Folder", "Workspace", "Tool", "HopperBin", "Backpack", "Configuration",
            "StarterPack", "StarterGui", "StarterPlayer", "StarterCharacterScripts",
            "StarterPlayerScripts", "ServerScriptService", "ServerStorage", "ReplicatedStorage",
            "ReplicatedFirst", "Lighting", "Players", "Teams", "SoundService", "Camera", "Terrain"}
SHAPE_ENUM = {0: "Ball", 1: "Block", 2: "Cylinder"}                              # Enum.PartType
CLASS_SHAPE = {"WedgePart": "Wedge", "CornerWedgePart": "CornerWedge",
               "TrussPart": "Truss", "MeshPart": "Mesh", "UnionOperation": "Union"}


def _f(v, d=0.0):
    try: return float(v)
    except (TypeError, ValueError): return d


def vec3(v, default=(0, 0, 0)):
    if v is None: return list(default)
    if hasattr(v, "x"): return [_f(v.x), _f(v.y), _f(v.z)]
    if isinstance(v, dict): return [_f(v.get("x", 0)), _f(v.get("y", 0)), _f(v.get("z", 0))]
    if isinstance(v, (list, tuple)) and len(v) >= 3: return [_f(v[0]), _f(v[1]), _f(v[2])]
    return list(default)


def rot_matrix(cf):
    """3x3 rotation as flat 9-list (row-major) from a CF object; identity if absent."""
    if cf is None or not hasattr(cf, "R00"):
        return [1, 0, 0, 0, 1, 0, 0, 0, 1]
    return [_f(cf.R00), _f(cf.R01), _f(cf.R02),
            _f(cf.R10), _f(cf.R11), _f(cf.R12),
            _f(cf.R20), _f(cf.R21), _f(cf.R22)]


def color_of(p):
    c = p.get("Color3uint8")
    if c is None: c = p.get("Color")
    if c is None: c = p.get("BrickColor")
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return [round(_f(c[0]), 3), round(_f(c[1]), 3), round(_f(c[2]), 3)]
    if isinstance(c, (int, float)):                                              # raw BrickColor index
        return [round(x, 3) for x in _get_brick_color(int(c))]
    return [0.64, 0.64, 0.64]


def shape_of(cn, p):
    if cn in CLASS_SHAPE: return CLASS_SHAPE[cn]
    s = p.get("shape", p.get("Shape", p.get("PartType")))
    if isinstance(s, (int, float)): return SHAPE_ENUM.get(int(s), "Block")
    return "Block"


def extract_requires(src):
    return [m.strip("\"' ") for m in re.findall(r"require\s*\(\s*([^)]+)\s*\)", src)][:8]


def _load(fp):
    """Return (instances_dict, rot_ok). Route by MAGIC bytes (extension is unreliable).
    Only XML-fromstring yields full rotation; binary orientation-IDs decode to identity (partial)."""
    head = open(fp, "rb").read(8)
    if head == b"<roblox ":                                                      # XML
        if Path(fp).stat().st_size <= 30_000_000:                               # small → fromstring (fast)
            try: return parse_file_fromstring(str(fp)), True
            except Exception: pass
        try: return parse_file(str(fp)), True                                   # streaming: memory-safe + reads rotation
        except Exception: return None, False
    if head == b"<roblox!":                                                      # binary: bundled parser is broken
        if Path(fp).with_suffix(".rbxlx").exists():                              # prefer the converted XML twin
            return None, False
        if parse_rbxl is None: return None, False
        try: return parse_rbxl(open(fp, "rb").read()).get("instances", {}), False
        except Exception: return None, False
    return None, False


def _pos_rot(cf, p):
    """(pos[3], rot[9]) from a CFrame — handles XML CF objects AND binary ([x,y,z],[9])."""
    if hasattr(cf, "x") and hasattr(cf, "R00"):                                  # XML CF object
        return [_f(cf.x), _f(cf.y), _f(cf.z)], rot_matrix(cf)
    if isinstance(cf, (list, tuple)) and len(cf) == 2 and isinstance(cf[0], (list, tuple)) and len(cf[0]) >= 3:
        r = cf[1] if (isinstance(cf[1], (list, tuple)) and len(cf[1]) == 9) else [1, 0, 0, 0, 1, 0, 0, 0, 1]
        return [_f(cf[0][0]), _f(cf[0][1]), _f(cf[0][2])], [_f(x) for x in r]
    return vec3(p.get("Position") or p.get("position")), [1, 0, 0, 0, 1, 0, 0, 0, 1]


def extract_game(fp):
    instances, rot_ok = _load(fp)
    if not instances: return None

    def cn_of(i): return getattr(i, "class_name", "") or (i.get("class_name", "") if isinstance(i, dict) else "")
    def props_of(i): return getattr(i, "properties", None) or (i.get("properties", {}) if isinstance(i, dict) else {})

    # Unified parent map: XML exposes parent_ref; binary exposes .children (Instance objects).
    parent = {}
    for ref, inst in instances.items():
        pr = getattr(inst, "parent_ref", None)
        if pr is not None: parent[ref] = pr
        for ch in (getattr(inst, "children", None) or []):
            cref = getattr(ch, "referent", None)
            cref = cref if cref is not None else (ch if isinstance(ch, str) else None)
            if cref is not None: parent[str(cref)] = ref

    ref_info = {ref: (cn_of(i), str(props_of(i).get("Name", cn_of(i)))[:40]) for ref, i in instances.items()}

    mesh_by_parent = {}                                                          # legacy SpecialMesh -> parent meshId
    for ref, inst in instances.items():
        if cn_of(inst) in MESH_CLASSES:
            mid = props_of(inst).get("MeshId") or props_of(inst).get("MeshID") or ""
            par = parent.get(ref)
            if mid and par: mesh_by_parent[par] = str(mid)

    parts, scripts, pid = [], [], {}
    for ref, inst in instances.items():
        cn, p = cn_of(inst), props_of(inst)
        if cn in PART_CLASSES:
            pos, rot = _pos_rot(p.get("CFrame") or p.get("cframe"), p)
            meshid = str(p.get("MeshId") or p.get("MeshID") or "") if cn == "MeshPart" else mesh_by_parent.get(ref, "")
            pid[ref] = len(parts)
            pcls, pname = ref_info.get(parent.get(ref), ("", ""))
            parts.append({
                "id": len(parts),
                "name": str(p.get("Name", cn))[:40], "class": cn, "shape": shape_of(cn, p),
                "pos": [round(x, 3) for x in pos], "rot": [round(x, 4) for x in rot],
                "size": [round(x, 3) for x in vec3(p.get("size") or p.get("Size"), (1, 1, 1))],
                "color": color_of(p), "material": str(p.get("Material") or "Plastic"),
                "anchored": bool(p.get("Anchored", False)), "cancollide": bool(p.get("CanCollide", True)),
                "meshid": meshid, "_ref": ref, "parent_class": pcls, "parent_name": pname,
            })
        elif cn in SCRIPT_CLASSES:
            src = p.get("Source") or p.get("source") or ""
            if isinstance(src, bytes): src = src.decode("utf-8", "ignore")
            if len(src.strip()) > 20:
                scripts.append({"id": len(scripts), "name": str(p.get("Name", cn))[:40], "class": cn,
                                "source": src, "_ref": ref, "requires": extract_requires(src)})
    if not parts and not scripts: return None

    def attach_for(ref):
        """First Part ancestor walking up (through Decal/Mesh/Gui/Sound/Light wrappers);
        stop (None) at a structural container so scripts in Models/Tools stay unbound."""
        cur = parent.get(ref)
        for _ in range(6):
            if cur is None:
                return None
            if cur in pid:
                return pid[cur]
            if ref_info.get(cur, ("", ""))[0] in NONCLIMB:
                return None
            cur = parent.get(cur)
        return None

    for x in parts:                                                              # resolve parent-part id
        x["parent_part"] = pid.get(parent.get(x.pop("_ref")))
    for s in scripts:
        s["attach"] = attach_for(s.pop("_ref"))

    return {"id": Path(fp).stem[:80], "parts": parts, "scripts": scripts,
            "parts_count": len(parts), "scripts_count": len(scripts), "rot_ok": rot_ok}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", help="one or more archive directories")
    ap.add_argument("--out", default="data/structured_v3.jsonl")
    ap.add_argument("--limit", type=int, default=100000)
    a = ap.parse_args()
    roots = a.roots or ["../RobloxRBXLArchive"]
    limit = a.limit
    output_file = Path(a.out)

    files = []
    for r in roots:
        files += list(Path(r).rglob("*.rbxl")) + list(Path(r).rglob("*.rbxlx"))
    print(f"Found {len(files)} files in {len(roots)} root(s), processing up to {min(len(files), limit)}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ok = err = no_rot = sc_tot = att_tot = dup = 0
    seen_ids = set()
    with open(output_file, "w", encoding="utf-8") as f:
        for i, fp in enumerate(files[:limit]):
            if i % 50 == 0: gc.collect()
            try:
                game = extract_game(str(fp))
                if not game: continue
                if game["id"] in seen_ids:                                       # dedup same place across roots
                    dup += 1; continue
                seen_ids.add(game["id"])
                if not game["rot_ok"]: no_rot += 1
                sc_tot += len(game["scripts"])
                att_tot += sum(1 for s in game["scripts"] if s["attach"] is not None)
                f.write(json.dumps(game, ensure_ascii=False) + "\n")
                ok += 1
                if ok % 50 == 0:
                    print(f"  [{ok}] {game['id'][:40]} ({game['parts_count']}p {game['scripts_count']}s)")
            except Exception as e:
                err += 1
                if err <= 10: print(f"  ! {Path(fp).name[:40]}: {type(e).__name__}: {str(e)[:80]}")

    rate = att_tot / max(sc_tot, 1) * 100
    print(f"Done: {ok} games -> {output_file}  ({no_rot} without rotation, {err} errors, {dup} dup-id skipped)")
    print(f"ATTACH: {att_tot:,}/{sc_tot:,} scripts bound to a part ({rate:.0f}%)")


if __name__ == "__main__":
    main()
