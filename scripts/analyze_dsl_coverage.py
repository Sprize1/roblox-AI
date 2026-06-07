#!/usr/bin/env python3
"""Coverage analysis: what % of the scripts reduce to a small operation-DSL?

Decides whether the unified op-stream representation is viable. Streams structured_v3,
dedups scripts (many are copy-pasted free models), classifies each UNIQUE script into a
pattern bucket, and reports how much is DSL-coverable vs needs constructs a simple
CREATE/SET/ON/IF/WAIT DSL can't capture (require/remote/metatables/coroutines).
"""
import sys, json, re, hashlib
from collections import Counter

INP = sys.argv[1] if len(sys.argv) > 1 else "data/structured_v3.jsonl"

# --- patterns ---
P = {
    "touched":   re.compile(r'\.Touched\b'),
    "kill":      re.compile(r'BreakJoints|Health\s*=\s*0|Humanoid[^\n]*Health\s*=\s*Humanoid', re.I),
    "inst_new":  re.compile(r'Instance\.new'),
    "destroy":   re.compile(r':\s*[Dd]estroy\s*\(|:\s*[Rr]emove\s*\('),
    "clone":     re.compile(r':\s*[Cc]lone\s*\(|:\s*[Mm]akeJoints'),
    "tp_torso":  re.compile(r'(Torso|HumanoidRootPart)[^\n]*CFrame|CFrame[^\n]*=|\.Velocity\s*='),
    "tween":     re.compile(r'Transparency\s*=|TweenService|:\s*Tween|:\s*Play\s*\(|BrickColor\s*='),
    "loop":      re.compile(r'\bwhile\b|\bfor\b'),
    "propset":   re.compile(r'\.\w+\s*='),
    "ui":        re.compile(r'PlayerGui|ScreenGui|TextLabel|TextButton|\bFrame\b|ImageLabel|\.Text\s*='),
    # hard for a simple op-DSL:
    "remote":    re.compile(r'RemoteEvent|RemoteFunction|FireServer|FireClient|OnServerEvent|OnClientEvent|InvokeServer|BindableEvent'),
    "require":   re.compile(r'\brequire\s*\('),
    "meta":      re.compile(r'setmetatable|getmetatable|coroutine|pcall|task\.spawn|\bcoroutine\b'),
    "datastore": re.compile(r'DataStoreService|HttpService|MarketplaceService'),
}
HARD = ("remote", "require", "meta", "datastore")


def classify(src):
    hits = {k: bool(v.search(src)) for k, v in P.items()}
    if any(hits[h] for h in HARD):
        return "HARD_networked_module"
    if hits["ui"]:
        return "ui"
    if hits["touched"]:
        if hits["kill"]:
            return "touch_kill"
        if hits["tween"] or hits["tp_torso"]:
            return "touch_effect"
        return "touch_other"
    if hits["tp_torso"] and hits["loop"] is False and ("CFrame" in src):
        return "teleport"
    if hits["clone"] or hits["destroy"] or hits["inst_new"]:
        return "spawn_destroy"
    if hits["loop"] and (hits["propset"] or hits["tween"]):
        return "loop_effect"
    if hits["propset"]:
        return "simple_prop"
    return "complex_other"


def norm(s):
    return re.sub(r'\s+', ' ', s.strip())


def main():
    seen = {}                       # hash -> bucket (dedup)
    uniq_bucket = Counter()
    inst_bucket = Counter()
    total_scripts = attached = 0
    uniq_attached = set()
    hard_inst = 0
    line_counts = []

    with open(INP, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            for s in d.get("scripts", []):
                src = s.get("source", "")
                if not src or not src.strip():
                    continue
                total_scripts += 1
                if s.get("attach") is not None:
                    attached += 1
                h = hashlib.md5(norm(src).encode("utf-8", "ignore")).digest()
                if h not in seen:
                    seen[h] = classify(src)
                    uniq_bucket[seen[h]] += 1
                    line_counts.append(src.count("\n") + 1)
                b = seen[h]
                inst_bucket[b] += 1
                if b == "HARD_networked_module":
                    hard_inst += 1
                if s.get("attach") is not None:
                    uniq_attached.add(h)

    U = len(seen)
    COVERABLE = {"touch_kill", "touch_effect", "touch_other", "teleport",
                 "spawn_destroy", "loop_effect", "simple_prop", "ui"}
    cov_u = sum(uniq_bucket[b] for b in COVERABLE)
    cov_i = sum(inst_bucket[b] for b in COVERABLE)

    print(f"=== SCRIPTS ===")
    print(f"total instances : {total_scripts:,}   attached-to-part: {attached:,} ({attached/max(total_scripts,1)*100:.0f}%)")
    print(f"UNIQUE (dedup)  : {U:,}   -> dup factor {total_scripts/max(U,1):.1f}x")
    print(f"unique attached : {len(uniq_attached):,}")
    line_counts.sort()
    med = line_counts[len(line_counts)//2] if line_counts else 0
    short = sum(1 for c in line_counts if c <= 20)
    print(f"unique median lines: {med}   <=20 lines: {short/max(U,1)*100:.0f}%")
    print(f"\n=== DSL COVERAGE ===")
    print(f"coverable by simple op-DSL: UNIQUE {cov_u/max(U,1)*100:.0f}%   INSTANCES {cov_i/max(total_scripts,1)*100:.0f}%")
    print(f"HARD (remote/require/meta) : INSTANCES {hard_inst/max(total_scripts,1)*100:.0f}%")
    print(f"\n=== BUCKETS (unique / instances) ===")
    for b, _ in inst_bucket.most_common():
        print(f"  {b:24s} uniq {uniq_bucket[b]:6,} ({uniq_bucket[b]/max(U,1)*100:4.1f}%)   inst {inst_bucket[b]:7,} ({inst_bucket[b]/max(total_scripts,1)*100:4.1f}%)")


if __name__ == "__main__":
    main()
