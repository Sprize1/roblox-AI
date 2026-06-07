#!/usr/bin/env python3
"""Cross-modal consistency metric for the op-stream models = the scaling signal.

Generates games from a trained model and measures REFERENTIAL VALIDITY: do the references
(`#id` used in ATTACH/SET/CALL/ON ops, i.e. the geometry<->logic edges) point to parts the
model actually CREATEd? A model that learns the binding produces valid refs; a model that
just memorizes text hallucinates ids. If validity ↗ with model size -> the scaling signal.

  python scripts/eval_opstream.py --size s   (then m, l)
  python scripts/eval_opstream.py --report    (print the scaling table)
"""
import sys, json, re, argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_opstream import GPT, TOKPATH

CREATE_ID = re.compile(r'^CREATE\s+\S+\s+#(\d+)')
REF_ID = re.compile(r'#(\d+)')


@torch.no_grad()
def generate(model, tok, dev, block, n_new, temp=0.8, topk=40):
    bos = tok.token_to_id("<bos>"); eos = tok.token_to_id("<eos>")
    ids = [bos] + tok.encode("[GAME ").ids
    x = torch.tensor([ids], device=dev)
    for _ in range(n_new):
        logits, _ = model(x[:, -block:])
        logits = logits[0, -1] / temp
        v, _ = torch.topk(logits, topk)
        logits[logits < v[-1]] = -float("inf")
        p = torch.softmax(logits, -1)
        nxt = torch.multinomial(p, 1).item()
        if nxt == eos:
            break
        x = torch.cat([x, torch.tensor([[nxt]], device=dev)], 1)
    return tok.decode(x[0].tolist())


ATTACH_AT = re.compile(r'ATTACH #(\d+)')
CREATE_AT = re.compile(r'CREATE \S+ #(\d+)')


@torch.no_grad()
def eval_binding(model, tok, dev, block, val_path, max_games=300):
    """Teacher-forced: at each real `ATTACH #` in val, does the model predict the CORRECT part
    id (exact = the true binding) or at least a CREATED id (valid = not hallucinated)?
    Dense over all ATTACH lines -> the cross-modal binding signal, no free-gen sparsity."""
    bos = tok.token_to_id("<bos>")
    tot = exact = valid = games = 0
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
            enc = tok.encode(text)
            ids, offs = enc.ids, enc.offsets
            for m in ATTACH_AT.finditer(text):
                r = m.group(1)
                created = set(c.group(1) for c in CREATE_AT.finditer(text[:m.start()]))
                if not created:
                    continue
                hashpos = m.start() + 7                       # the '#'
                tok_idx = next((j for j, (a, b) in enumerate(offs) if a >= hashpos), None)
                if not tok_idx:
                    continue
                prefix = ([bos] + ids[:tok_idx])[-block:]
                logits, _ = model(torch.tensor([prefix], device=dev))
                dm = re.search(r'\d+', tok.decode([logits[0, -1].argmax().item()]))
                tot += 1
                if dm:
                    exact += int(dm.group(0) == r)
                    valid += int(dm.group(0) in created)
            games += 1
            if games >= max_games:
                break
    return tot, exact, valid


def score(text):
    declared, refs, valid = set(), 0, 0
    attach, attach_ok = 0, 0
    for ln in text.splitlines():
        m = CREATE_ID.match(ln.strip())
        if m:
            declared.add(m.group(1)); continue
        ids = REF_ID.findall(ln)
        if ln.strip().startswith("ATTACH") and ids:
            attach += 1; attach_ok += int(ids[0] in declared)
        for r in ids:
            refs += 1; valid += int(r in declared)
    return {"parts": len(declared), "refs": refs, "valid": valid, "attach": attach, "attach_ok": attach_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="s")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-new", type=int, default=900)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--binding", action="store_true")
    ap.add_argument("--val", default="data/smoke_val.jsonl")
    a = ap.parse_args()

    def jload(path):
        return {json.loads(l)["size"]: json.loads(l) for l in open(path, encoding="utf-8")} if Path(path).exists() else {}

    if a.report:
        loss = jload("models/opstream_results.jsonl")
        ev = jload("models/opstream_eval.jsonl")
        bd = jload("models/opstream_binding.jsonl")
        print(f"{'size':>4} {'params_M':>9} {'val_loss':>9} {'gen_refvalid%':>13} {'bind_exact%':>12} {'bind_valid%':>12}")
        for s in ["s", "m", "l"]:
            if s in loss:
                L = loss[s]; E = ev.get(s, {}); B = bd.get(s, {})
                rv = f"{E['ref_valid']*100:.1f}" if E else "-"
                bx = f"{B['exact']*100:.1f}" if B else "-"
                bv = f"{B['valid']*100:.1f}" if B else "-"
                print(f"{s:>4} {L['params_M']:>9} {L['val_loss']:>9} {rv:>13} {bx:>12} {bv:>12}")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKPATH)
    cfg = json.load(open(f"models/opstream_{a.size}/config.json"))
    model = GPT(cfg["vocab"], cfg["block"], cfg["n_layer"], cfg["n_embd"], cfg["n_head"]).to(dev)
    model.load_state_dict(torch.load(f"models/opstream_{a.size}/model.pt", map_location=dev))
    model.eval()

    if a.binding:
        tot, exact, valid = eval_binding(model, tok, dev, cfg["block"], a.val)
        rec = {"size": a.size, "attach_n": tot, "exact": round(exact / max(tot, 1), 4), "valid": round(valid / max(tot, 1), 4)}
        with open("models/opstream_binding.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"BINDING {rec}")
        return

    agg = {"parts": 0, "refs": 0, "valid": 0, "attach": 0, "attach_ok": 0}
    sample0 = ""
    for i in range(a.n):
        g = generate(model, tok, dev, cfg["block"], a.n_new)
        if i == 0:
            sample0 = g
        s = score(g)
        for k in agg: agg[k] += s[k]
    rec = {"size": a.size,
           "ref_valid": round(agg["valid"] / max(agg["refs"], 1), 4),
           "attach_valid": round(agg["attach_ok"] / max(agg["attach"], 1), 4),
           "avg_parts": round(agg["parts"] / a.n, 1),
           "avg_refs": round(agg["refs"] / a.n, 1), "n": a.n}
    with open("models/opstream_eval.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    Path(f"models/opstream_{a.size}_sample.txt").write_text(sample0[:4000], encoding="utf-8")
    print(f"EVAL {rec}")
    print(f"sample -> models/opstream_{a.size}_sample.txt")


if __name__ == "__main__":
    main()
