#!/usr/bin/env python3
"""Measure TRUE op-DSL lowering fidelity (line granularity) on UNIQUE scripts.

Turns the optimistic pattern-match 76% into a measured number: classify every code line of
each unique script into an op template (CREATE/SET/CALL/ON/IF/LOOP/WAIT/FUNC/RETURN/BLOCK).
A line is COVERED if it maps; SET with a non-representable RHS is OPAQUE (structure ok, value
not); anything else is UNCOVERED. Per-script coverage = covered / code lines.

Honest about granularity: this is line-level structural coverage, the practical proxy for
"can this script be represented as an op-stream", plus a few sample lowerings to eyeball.
"""
import sys, json, re, hashlib
from collections import Counter

INP = sys.argv[1] if len(sys.argv) > 1 else "data/structured_v3.jsonl"
SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 4

KNOWN_METHODS = ("Destroy", "remove", "Remove", "Clone", "clone", "BreakJoints", "MakeJoints",
                 "makeJoints", "FindFirstChild", "findFirstChild", "WaitForChild", "GetChildren",
                 "Play", "Stop", "Connect", "connect", "Disconnect", "Kill", "LoadAnimation",
                 "TweenSize", "Resize")
# RHS values we can represent exactly:
REPR_RHS = re.compile(r'^\s*(-?\d+\.?\d*|"[^"]*"|\'[^\']*\'|true|false|nil|'
                      r'Vector3\.new\([^()]*\)|CFrame\.new\([^()]*\)|UDim2\.new\([^()]*\)|'
                      r'Color3\.\w+\([^()]*\)|BrickColor\.new\([^()]*\)|Enum\.[\w.]+|'
                      r'[\w.]+(\.\w+)*)\s*$')
HARD = re.compile(r'RemoteEvent|RemoteFunction|FireServer|FireClient|OnServerEvent|OnClientEvent|'
                  r'InvokeServer|BindableEvent|\brequire\s*\(|setmetatable|coroutine|pcall|'
                  r'DataStoreService|HttpService|task\.spawn')


def strip_comments(src):
    src = re.sub(r'--\[\[.*?\]\]', '', src, flags=re.S)
    out = []
    for ln in src.splitlines():
        ln = re.sub(r'--.*$', '', ln).rstrip()
        out.append(ln)
    return out


def classify(line):
    s = line.strip()
    if not s:
        return None                                   # blank, not counted
    if HARD.search(s):
        return ("UNCOVERED_HARD", s)
    if re.match(r'^(end|until|\})[\s);,]*$', s):
        return ("BLOCK", None)
    if re.match(r'^(else|elseif\b.*\bthen)\s*$', s) or s == "else":
        return ("CTRL", None)
    if re.match(r'^(local\s+)?function\b', s) or re.search(r'=\s*function\b', s):
        return ("FUNC", None)
    if "Instance.new" in s:
        return ("CREATE", None)
    if re.search(r'[:.][Cc]onnect\s*\(', s) or re.search(r'\.\w+:[Cc]onnect', s):
        return ("ON", None)
    if re.match(r'^if\b.*\bthen\b', s):
        return ("IF", None)
    if re.match(r'^(while\b.*\bdo|for\b.*\bdo|repeat)\b', s) or s == "do":
        return ("LOOP", None)
    if re.match(r'^wait\s*\(', s) or re.match(r'^task\.wait', s):
        return ("WAIT", None)
    if re.match(r'^print\s*\(', s):
        return ("PRINT", None)
    if re.match(r'^return\b', s):
        return ("RETURN", None)
    # assignment: LHS = RHS
    m = re.match(r'^(local\s+)?([\w.\[\]"\']+)\s*=\s*(.+)$', s)
    if m:
        rhs = m.group(3).strip().rstrip(";")
        return ("SET", None) if REPR_RHS.match(rhs) else ("SET_OPAQUE", s)
    # bare known-method call statement
    mm = re.search(r':(\w+)\s*\(', s)
    if mm and mm.group(1) in KNOWN_METHODS:
        return ("CALL", None)
    return ("UNCOVERED", s)


def main():
    seen = set()
    cls = Counter()
    cov_hist = []                  # per-script coverage fraction (covered incl opaque)
    full = mostly = 0
    uncovered_samples, lower_samples = [], []

    with open(INP, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            for sc in d.get("scripts", []):
                src = sc.get("source", "")
                if not src or not src.strip():
                    continue
                h = hashlib.md5(re.sub(r'\s+', ' ', src.strip()).encode("utf-8", "ignore")).digest()
                if h in seen:
                    continue
                seen.add(h)
                code, covered, opaque, uncov, ops = 0, 0, 0, 0, []
                for ln in strip_comments(src):
                    r = classify(ln)
                    if r is None:
                        continue
                    code += 1
                    tag, sample = r
                    cls[tag] += 1
                    ops.append(tag)
                    if tag in ("UNCOVERED", "UNCOVERED_HARD"):
                        uncov += 1
                        if sample and len(uncovered_samples) < 25:
                            uncovered_samples.append(sample)
                    elif tag == "SET_OPAQUE":
                        opaque += 1; covered += 1
                    else:
                        covered += 1
                if code == 0:
                    continue
                frac = covered / code
                cov_hist.append(frac)
                if uncov == 0:
                    full += 1
                if frac >= 0.9:
                    mostly += 1
                if len(lower_samples) < SAMPLES and 6 <= code <= 16 and uncov == 0:
                    lower_samples.append((src, ops))

    U = len(cov_hist)
    total_lines = sum(cls.values())
    cov_lines = total_lines - cls["UNCOVERED"] - cls["UNCOVERED_HARD"]
    print(f"=== LOWERING FIDELITY (unique scripts: {U:,}) ===")
    print(f"line coverage      : {cov_lines/max(total_lines,1)*100:.0f}%  ({cov_lines:,}/{total_lines:,} code lines)")
    print(f"opaque-value lines : {cls['SET_OPAQUE']/max(total_lines,1)*100:.0f}%  (structure ok, literal not representable)")
    print(f"scripts FULLY lowerable (0 uncovered): {full/max(U,1)*100:.0f}%")
    print(f"scripts >=90% covered                : {mostly/max(U,1)*100:.0f}%")
    print(f"mean per-script coverage             : {sum(cov_hist)/max(U,1)*100:.0f}%")
    print(f"\n=== OP DISTRIBUTION ===")
    for t, c in cls.most_common():
        print(f"  {t:16s} {c:8,} ({c/max(total_lines,1)*100:4.1f}%)")
    print(f"\n=== UNCOVERED LINE SAMPLES ===")
    for s in uncovered_samples[:18]:
        print(f"  | {s[:90]}")
    print(f"\n=== SAMPLE LOWERINGS (fully covered) ===")
    for src, ops in lower_samples:
        print("  --- source ---")
        for ln in src.splitlines()[:16]:
            print(f"    {ln}")
        print(f"  --- ops: {' '.join(ops)}")


if __name__ == "__main__":
    main()
