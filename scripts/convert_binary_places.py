#!/usr/bin/env python3
"""
Batch-convert binary .rbxl ('<roblox!') places to .rbxlx (XML) so build_structured_v3
can ingest them through its reliable XML path (full rotation). The bundled Python binary
parser does NOT implement the real RBXL format, so we delegate to an external converter.

Recommended converters (you have Cargo):
  - rbx-util (Rojo / rbx-dom):
        cargo install --git https://github.com/rojo-rbx/rbx-dom rbx-util
        default cmd:  "rbx-util convert {in} {out}"
  - Lune:
        cargo install lune
        cmd:  "lune run scripts/convert.luau {in} {out}"   (see scripts/convert.luau)

Writes <name>.rbxlx NEXT TO each binary <name>.rbxl (originals untouched). build_structured_v3
skips a binary .rbxl when its .rbxlx twin exists.

Usage:
  python scripts/convert_binary_places.py <root> [<root2> ...] [--cmd "..."] [--workers 4] [--limit N]
"""
import argparse, os, shutil, shlex, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def is_binary(fp):
    try:
        with open(fp, "rb") as f:
            return f.read(8) == b"<roblox!"
    except OSError:
        return False


def find_binary(roots):
    out = []
    for root in roots:
        for dp, _, fs in os.walk(root):
            for fn in fs:
                if fn.lower().endswith(".rbxl"):
                    fp = os.path.join(dp, fn)
                    if is_binary(fp):
                        out.append(fp)
    return out


def convert_one(fp, tmpl):
    out = str(Path(fp).with_suffix(".rbxlx"))
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "skip", fp
    cmd = [a.replace("{in}", fp).replace("{out}", out) for a in tmpl]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
            return "ok", fp
        if os.path.exists(out) and os.path.getsize(out) == 0:
            os.remove(out)                                                       # clean empty partial
        msg = (r.stderr or r.stdout or f"rc={r.returncode}").strip()[:90]
        return f"fail:{msg}", fp
    except Exception as e:
        return f"exc:{type(e).__name__}:{str(e)[:60]}", fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="archive directories to scan")
    ap.add_argument("--cmd", default="rbx-util convert {in} {out}", help="converter command template")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    tmpl = shlex.split(a.cmd)
    if shutil.which(tmpl[0]) is None:
        print(f"[!] converter '{tmpl[0]}' not on PATH. Install one:")
        print("    rbx-util:  cargo install --git https://github.com/rojo-rbx/rbx-dom rbx-util")
        print('    Lune:      cargo install lune   (--cmd "lune run scripts/convert.luau {in} {out}")')
        sys.exit(1)

    files = find_binary(a.roots)
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} binary .rbxl found. Converting with: {a.cmd}")

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(convert_one, fp, tmpl) for fp in files]
        for i, fut in enumerate(as_completed(futs), 1):
            status, fp = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                if fail <= 15:
                    print(f"  ! {Path(fp).name[:40]}: {status}")
            if i % 50 == 0:
                print(f"  [{i}/{len(files)}] ok={ok} skip={skip} fail={fail}")

    print(f"Done: ok={ok} skip={skip} fail={fail}  (.rbxlx written next to originals)")


if __name__ == "__main__":
    main()
