#!/usr/bin/env python3
"""Teacher-forced name->id binding accuracy on synthetic games (the decisive test).

At each `workspace.<name>#<id>` reference, given the prefix up to the id, does the model predict
the correct id? Compares to the recency baseline (last-created id). exact >> recency = real
in-context binding learned. Run per size -> scaling curve on a DENSE binding signal.
"""
import sys, json, re, argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_opstream import GPT, TOKPATH

REF = re.compile(r'workspace\.[a-z]+#(\d+)')
CRE = re.compile(r'CREATE[^#\n]*#(\d+)')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="s")
    ap.add_argument("--val", default="data/synth_val.jsonl")
    ap.add_argument("--max-games", type=int, default=150)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKPATH)
    cfg = json.load(open(f"models/opstream_{a.size}/config.json"))
    model = GPT(cfg["vocab"], cfg["block"], cfg["n_layer"], cfg["n_embd"], cfg["n_head"]).to(dev)
    model.load_state_dict(torch.load(f"models/opstream_{a.size}/model.pt", map_location=dev))
    model.eval()
    bos = tok.token_to_id("<bos>")

    tot = exact = recency = ks = ng = 0
    with open(a.val, encoding="utf-8") as f:
        for line in f:
            if ng >= a.max_games:
                break
            ng += 1
            text = json.loads(line)["text"]
            enc = tok.encode(text); ids, offs = enc.ids, enc.offsets
            ks += len(CRE.findall(text))
            for m in REF.finditer(text):
                true = m.group(1)
                creates = [c.group(1) for c in CRE.finditer(text[:m.start()])]
                if not creates:
                    continue
                firstdigit = m.end() - len(true)
                tok_idx = next((k for k, (aa, bb) in enumerate(offs) if aa <= firstdigit < bb), None)
                if not tok_idx:
                    continue
                prefix = ([bos] + ids[:tok_idx])[-cfg["block"]:]
                logits, _ = model(torch.tensor([prefix], device=dev))
                dm = re.search(r'\d+', tok.decode([logits[0, -1].argmax().item()]))
                tot += 1
                if dm:
                    exact += int(dm.group(0) == true)
                recency += int(creates[-1] == true)

    rec = {"size": a.size, "refs": tot,
           "exact": round(exact / max(tot, 1), 4),
           "recency_base": round(recency / max(tot, 1), 4),
           "avg_K": round(ks / max(ng, 1), 1)}
    with open("models/synth_eval.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"SYNTH {rec}  (exact >> recency_base => real in-context binding)")


if __name__ == "__main__":
    main()
