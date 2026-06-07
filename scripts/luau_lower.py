#!/usr/bin/env python3
"""Statement-level Luau -> op-stream lowerer (the unified-representation core).

A real tokenizer + tolerant recursive-descent over the statement grammar (not line-based),
so block nesting and multi-statement lines are correct. `lower(src)` returns RECORDS
[(kind, text), ...] with explicit BLOCK_END markers; event handlers nest after `ON` so the
causal structure (trigger -> effects) is captured. The `text` keeps the statement's tokens
(incl. instance references like script.Parent.Torso = the geometry<->logic edges). Unparseable
bits -> ("OPAQUE", raw) so NOTHING is lost.

CLI streams the dataset and reports OP-STRUCTURE vs VALUE-FIDELITY + sample lowerings.
"""
import sys, json, re, hashlib
from collections import Counter

KW = {"and", "break", "do", "else", "elseif", "end", "false", "for", "function", "if", "in",
      "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while"}
STMT_START = {"if", "while", "for", "function", "local", "return", "repeat", "do", "break"}
BLOCK_END = {"end", "else", "elseif", "until"}
OPEN = {"(": ")", "{": "}", "[": "]"}
CLOSE = {")", "}", "]"}
CONT_END = {"+", "-", "*", "/", "..", "=", "==", "~=", "<", ">", "<=", ">=", "and", "or", "not",
            ",", ".", ":", "(", "{", "[", "%"}        # line ends mid-expression -> continue
VALUE_OPAQUE = {"SETX", "LOCALX", "CALLEXPR", "OPAQUE"}   # op known, but value/args not closed-form

_LONGOPEN = re.compile(r'\[(=*)\[')
_NUM = re.compile(r'0[xX][0-9a-fA-F]+|\d*\.?\d+([eE][+-]?\d+)?')
_NAME = re.compile(r'[A-Za-z_]\w*')
_OP = re.compile(r'\.\.\.|\.\.|==|~=|<=|>=|::|//|[-+*/%^#<>=(){}\[\];:,.]')
_SQUEEZE = re.compile(r'\s*([.:,;()\[\]])\s*')


def render(toks):
    return _SQUEEZE.sub(r'\1', " ".join(t[1] for t in toks)).strip()


def tokenize(src):
    toks, i, n, line = [], 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1; i += 1; continue
        if c in " \t\r":
            i += 1; continue
        if c == "-" and src.startswith("--", i):                 # comment
            m = _LONGOPEN.match(src, i + 2)
            if m:
                eq = m.group(1); end = src.find("]" + eq + "]", i)
                line += src.count("\n", i, end if end >= 0 else n)
                i = (end + len(eq) + 2) if end >= 0 else n
            else:
                j = src.find("\n", i); i = n if j < 0 else j
            continue
        if c == "[":
            m = _LONGOPEN.match(src, i)
            if m:                                                # long string [[ ]]
                eq = m.group(1); end = src.find("]" + eq + "]", i)
                val = src[i:(end+len(eq)+2) if end >= 0 else n]
                line += val.count("\n"); i = (end+len(eq)+2) if end >= 0 else n
                toks.append(("str", val, line)); continue
        if c in "\"'":                                            # quoted string
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            toks.append(("str", src[i:j+1], line)); i = j + 1; continue
        if c.isdigit() or (c == "." and i+1 < n and src[i+1].isdigit()):
            m = _NUM.match(src, i)
            toks.append(("num", m.group(0), line)); i = m.end(); continue
        if c.isalpha() or c == "_":
            m = _NAME.match(src, i); w = m.group(0)
            toks.append(("kw" if w in KW else "name", w, line)); i = m.end(); continue
        m = _OP.match(src, i)
        if m:
            toks.append(("op", m.group(0), line)); i = m.end(); continue
        i += 1                                                    # skip unknown char
    return toks


class P:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else ("eof", "", -1)

    def nx(self):
        tok = self.peek(); self.i += 1; return tok

    def at(self, *vals):
        return self.peek()[1] in vals

    def read_expr(self):
        """Collect one expression-statement's tokens up to a stmt boundary."""
        out, depth = [], 0
        while self.i < len(self.t):
            k, v, ln = self.peek()
            if depth == 0:
                if v == ";":
                    self.i += 1; break
                if k == "kw" and (v in STMT_START or v in BLOCK_END):
                    break
                if out and ln != out[-1][2]:                      # new line at depth 0
                    if out[-1][1] not in CONT_END and v not in CONT_END and not (k == "op" and v in "([{"):
                        break
            if v in OPEN:
                depth += 1
            elif v in CLOSE:
                depth -= 1
            out.append(self.nx())
        return out

    def read_until(self, stopword):
        """Consume + return tokens up to (excluding) stopword at depth 0; consume stopword."""
        out, depth = [], 0
        while self.i < len(self.t):
            k, v, _ = self.peek()
            if depth == 0 and v == stopword:
                self.nx(); break
            if v in OPEN: depth += 1
            elif v in CLOSE: depth -= 1
            out.append(self.nx())
        return out

    def read_func_header(self):
        out = [self.nx()]                                  # 'function' or 'local'
        if self.peek()[1] == "function":
            out.append(self.nx())
        depth = 0
        while self.i < len(self.t):
            v = self.peek()[1]
            out.append(self.nx())
            if v == "(":
                depth += 1
            elif v == ")":
                depth -= 1
                if depth == 0:
                    break
        return out

    def parse_block(self, ops):
        while self.i < len(self.t):
            k, v, _ = self.peek()
            if (k == "kw" and v in BLOCK_END) or k == "eof":
                return
            before = self.i
            self.parse_stat(ops)
            if self.i == before:                           # no progress -> force advance
                self.i += 1; ops.append(("OPAQUE", ""))

    def parse_stat(self, ops):
        v = self.peek()[1]
        try:
            if v == ";":
                self.nx(); return
            if v == "function" or (v == "local" and self._is_local_func()):
                hdr = self.read_func_header()
                ops.append(("FUNC", render(hdr))); self.parse_block(ops); self._end(ops); return
            if v == "local":
                self.nx(); toks = self.read_expr()
                rhs = ""
                for j, t in enumerate(toks):
                    if t[1] == "=":
                        rhs = " ".join(x[1] for x in toks[j+1:]); break
                if "Instance.new" in "".join(t[1] for t in toks):
                    ops.append(("CREATE", render(toks)))
                else:
                    ops.append(("LOCAL" if (not rhs or _repr_rhs(rhs)) else "LOCALX", render(toks)))
                return
            if v == "if":
                self.nx(); ops.append(("IF", render(self.read_until("then")))); self.parse_block(ops)
                while self.at("elseif"):
                    self.nx(); ops.append(("ELSEIF", render(self.read_until("then")))); self.parse_block(ops)
                if self.at("else"):
                    self.nx(); ops.append(("ELSE", "")); self.parse_block(ops)
                self._end(ops); return
            if v == "while":
                self.nx(); ops.append(("WHILE", render(self.read_until("do")))); self.parse_block(ops); self._end(ops); return
            if v == "for":
                self.nx(); ops.append(("FOR", render(self.read_until("do")))); self.parse_block(ops); self._end(ops); return
            if v == "repeat":
                self.nx(); ops.append(("REPEAT", "")); self.parse_block(ops)
                if self.at("until"):
                    self.nx(); self.read_expr()
                ops.append(("BLOCK_END", "")); return
            if v == "do":
                self.nx(); ops.append(("DO", "")); self.parse_block(ops); self._end(ops); return
            if v == "return":
                self.nx(); ops.append(("RETURN", render(self.read_expr()))); return
            if v == "break":
                self.nx(); ops.append(("BREAK", "")); return
            self._expr_stat(ops); return
        except Exception:
            ops.append(("OPAQUE", render(self.read_expr())))

    def _is_local_func(self):
        return self.i+1 < len(self.t) and self.t[self.i+1][1] == "function"

    def _end(self, ops):
        if self.at("end"):
            self.nx()
        ops.append(("BLOCK_END", ""))

    def _expr_stat(self, ops):
        toks = self.read_expr()
        if not toks:
            ops.append(("OPAQUE", "")); return
        text = render(toks)
        flat = "".join(t[1] for t in toks)
        d = 0                                              # assignment: top-level '=' not '=='
        for j, t in enumerate(toks):
            if t[1] in OPEN: d += 1
            elif t[1] in CLOSE: d -= 1
            elif d == 0 and t[1] == "=" and not (j+1 < len(toks) and toks[j+1][1] == "="):
                rhs = " ".join(x[1] for x in toks[j+1:])
                ops.append(("SET" if _repr_rhs(rhs) else "SETX", text)); return
        if "Instance.new" in flat:
            ops.append(("CREATE", text)); return
        if re.search(r':[Cc]onnect\(', flat):
            ops.append(("ON", text)); return
        if flat.startswith(("wait(", "task.wait(")):
            ops.append(("WAIT", text)); return
        if flat.startswith("print("):
            ops.append(("PRINT", text)); return
        if re.search(r':\w+\(', flat):
            ops.append(("CALL", text)); return
        if re.match(r'^[\w.]+\(', flat):
            ops.append(("CALLEXPR", text)); return
        ops.append(("OPAQUE", text))


REPR = re.compile(r'^\s*(-?\d+\.?\d*|"[^"]*"|\'[^\']*\'|true|false|nil|'
                  r'Vector3 \. new \([^()]*\)|CFrame \. new \([^()]*\)|UDim2 \. new \([^()]*\)|'
                  r'Color3 \. \w+ \([^()]*\)|BrickColor \. new \([^()]*\)|Enum [\w. ]+|'
                  r'[\w]+( \. \w+)*)\s*$')
HARD = re.compile(r'RemoteEvent|RemoteFunction|FireServer|FireClient|OnServerEvent|OnClientEvent|'
                  r'InvokeServer|BindableEvent|require|setmetatable|coroutine|pcall')


def _repr_rhs(rhs):
    return bool(REPR.match(rhs.strip()))


def lower(src):
    """src -> ([(kind, text), ...], n_ops, n_opaque)."""
    p = P(tokenize(src))
    ops = []
    p.parse_block(ops)
    n = len(ops)
    opq = sum(1 for k, _ in ops if k == "OPAQUE")
    return ops, n, opq


def main():
    INP = sys.argv[1] if len(sys.argv) > 1 else "data/structured_v3.jsonl"
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seen = set(); op_count = Counter()
    n_scripts = full = mostly = huge = tot_stmt = tot_opq = 0
    samples = []
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
                if len(src) > 40000 or src.count("\n") > 1500:
                    huge += 1; continue
                ops, n, opq = lower(src)
                if n == 0:
                    continue
                kinds = [k for k, _ in ops]
                n_scripts += 1; tot_stmt += n; tot_opq += opq
                vop = sum(1 for k in kinds if k in VALUE_OPAQUE)
                for k in kinds:
                    op_count[k] += 1
                if opq == 0:
                    full += 1
                if vop == 0:
                    mostly += 1
                if len(samples) < NS and 7 <= n <= 20 and opq == 0 and "ON" in kinds and ("CREATE" in kinds or "CALL" in kinds):
                    samples.append((src, ops))

    vop_tot = sum(op_count[o] for o in VALUE_OPAQUE)
    print(f"=== STATEMENT-LEVEL LOWERING (unique scripts: {n_scripts:,}; skipped huge: {huge:,}) ===")
    print(f"OP-STRUCTURE   : {(tot_stmt-tot_opq)/max(tot_stmt,1)*100:.1f}% of statements map to a known operation ({tot_opq/max(tot_stmt,1)*100:.1f}% OPAQUE)")
    print(f"VALUE-FIDELITY : {(tot_stmt-vop_tot)/max(tot_stmt,1)*100:.0f}% of ops have closed-form values ({vop_tot/max(tot_stmt,1)*100:.0f}% opaque sub-expr)")
    print(f"scripts fully op-structured : {full/max(n_scripts,1)*100:.0f}%   fully value-faithful: {mostly/max(n_scripts,1)*100:.0f}%")
    print(f"\n=== OP DISTRIBUTION ===")
    for o, c in op_count.most_common():
        print(f"  {o:10s} {c:8,} ({c/max(tot_stmt,1)*100:4.1f}%)")
    print(f"\n=== SAMPLE LOWERINGS ===")
    for src, ops in samples:
        print("  --- ops ---")
        for k, t in ops:
            print(f"    {k:9s} {t}")


if __name__ == "__main__":
    main()
