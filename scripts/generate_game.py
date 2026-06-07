#!/usr/bin/env python3
"""Two-stage end-to-end: prompt -> COMPLETE Roblox game (layout + coupled Luau scripts).

  Phase 1 (layout adapter, demo_lora_chunks2): prompt -> z-order part lines, parsed back to parts.
  Select a few parts to script (evenly spread across the layout).
  Phase 2 (script adapter, demo_lora_phase2): for each, rebuild its local [PARTS] context
    (part + K z-order neighbours) -> generate the attached Luau.
  Assemble parts + {part -> script} = the coupled game. Saved as JSON + readable text.

One base model, two LoRA adapters swapped via set_adapter (saves VRAM).

Usage:
  python scripts/generate_game.py --prompt "[GAME: Lava Jump Obby]" --n-scripts 4 --out models/game_demo
"""
import sys, re, json, argparse, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from generate_phase2 import StopOnText, build_context
from make_demo_obby import DEFAULT_MAT, IDENT

PART_RE = re.compile(r'^P(\d+)\s+(\S+)\s+@(-?\d+),(-?\d+),(-?\d+)\s+([\d.]+)x([\d.]+)x([\d.]+)(.*)$')


def parse_parts(text):
    """Phase-1 text -> part dicts (reindexed). Non-part lines (scripts/[/GAME]) are skipped."""
    parts = []
    for line in text.splitlines():
        m = PART_RE.match(line.strip())
        if not m:
            continue
        rest = m.group(9)
        rot = IDENT
        rm = re.search(r'rot\[([-\d.,]+)\]', rest)
        if rm:
            try:
                rot = [float(x) for x in rm.group(1).split(",")]
            except ValueError:
                rot = IDENT
        nm = re.search(r'"([^"]*)"', rest)
        head = rest.strip().split()
        material = head[0] if head and not head[0].startswith(("rot[", '"')) and head[0] != "mesh" else DEFAULT_MAT
        parts.append({
            "id": len(parts), "shape": m.group(2),
            "pos": [float(m.group(3)), float(m.group(4)), float(m.group(5))],
            "size": [float(m.group(6)), float(m.group(7)), float(m.group(8))],
            "rot": rot, "material": material,
            "meshid": "mesh" if " mesh" in rest else None,
            "name": nm.group(1) if nm else "",
        })
    return parts


def gen(model, tok, prompt, dev, max_new, temp, rep, stop):
    ids = tok(prompt, return_tensors="pt").to(dev)
    plen = ids["input_ids"].shape[1]
    crit = StoppingCriteriaList([StopOnText(tok, stop, plen)]) if stop else None
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=temp,
                             repetition_penalty=rep, no_repeat_ngram_size=0 if stop else 3,
                             stopping_criteria=crit,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][plen:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--layout-adapter", default="models/demo_lora_chunks2")
    ap.add_argument("--script-adapter", default="models/demo_lora_phase2")
    ap.add_argument("--prompt", default="[GAME: Lava Jump Obby]")
    ap.add_argument("--n-scripts", type=int, default=4)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--max-layout", type=int, default=900)
    ap.add_argument("--max-script", type=int, default=500)
    ap.add_argument("--temp-layout", type=float, default=0.8)
    ap.add_argument("--temp-script", type=float, default=0.5)
    ap.add_argument("--out", default="models/game_demo")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
        attn_implementation="eager")
    model = PeftModel.from_pretrained(model, a.layout_adapter, adapter_name="layout")
    model.load_adapter(a.script_adapter, adapter_name="scripts")
    model.to(dev).eval()

    # --- Phase 1: layout ---
    model.set_adapter("layout")
    prompt = a.prompt if a.prompt.endswith("\n") else a.prompt + "\n"
    layout_txt = gen(model, tok, prompt, dev, a.max_layout, a.temp_layout, 1.15, "[/GAME]")
    parts = parse_parts(layout_txt)
    print(f"Phase 1: parsed {len(parts)} parts from layout")
    if not parts:
        print("no parts parsed — aborting"); return

    # --- select parts to script (evenly spread across the layout) ---
    n = min(a.n_scripts, len(parts))
    sel = [parts[round(i * (len(parts) - 1) / max(n - 1, 1))] for i in range(n)]

    # --- Phase 2: scripts conditioned on local geometry ---
    model.set_adapter("scripts")
    scripts = []
    for p in sel:
        ctx, tlocal = build_context(parts, p["id"], a.k)
        if ctx is None:
            continue
        src = gen(model, tok, ctx + f"\n[Script -> P{tlocal}]\n", dev,
                  a.max_script, a.temp_script, 1.2, "[/SCRIPT]")
        src = src.split("[/SCRIPT]")[0].rstrip()
        scripts.append({"attach": p["id"], "source": src})
        print(f"Phase 2: scripted part P{p['id']} ({p['shape']} '{p['name']}') -> {len(src)} chars")

    # --- assemble ---
    game = {"name": a.prompt, "parts": parts, "scripts": scripts}
    Path(a.out + ".json").write_text(json.dumps(game, ensure_ascii=False, indent=1), encoding="utf-8")
    txt = [f"=== {a.prompt} ===", f"{len(parts)} parts, {len(scripts)} coupled scripts", "", "--- PARTS (first 40) ---"]
    txt += [f"P{p['id']} {p['shape']} @{[round(v) for v in p['pos']]} {p['name']}" for p in parts[:40]]
    for s in scripts:
        txt += ["", f"--- SCRIPT on P{s['attach']} ---", s["source"]]
    Path(a.out + ".txt").write_text("\n".join(txt), encoding="utf-8")
    print(f"-> {a.out}.json / {a.out}.txt  ({len(parts)} parts, {len(scripts)} scripts)")


if __name__ == "__main__":
    main()
