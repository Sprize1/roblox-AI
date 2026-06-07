#!/usr/bin/env python3
"""From-scratch small GPT on the unified op-stream — the scaling-signal PoC.

Trains a fresh BPE tokenizer on the corpus, then a minimal decoder transformer at a chosen
size. Packs all games into block_size windows (standard LM pretraining). Saves model+config
and logs final train/val loss to models/opstream_results.jsonl so 3 sizes form a scaling curve.

Usage:
  python scripts/train_opstream.py --size s   # then m, then l
"""
import sys, json, math, time, argparse, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

SIZES = {                      # (n_layer, n_embd, n_head)
    "s": (4, 128, 4),
    "m": (6, 256, 8),
    "l": (8, 384, 8),
    "xl": (12, 512, 8),
}
TOKPATH = "data/opstream_tokenizer.json"


def build_tokenizer(train_path, vocab=8000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])

    def it():
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)["text"]
                except Exception:
                    continue
    tok.train_from_iterator(it(), trainer)
    tok.save(TOKPATH)
    print(f"tokenizer: vocab={tok.get_vocab_size()} -> {TOKPATH}")
    return tok


def load_ids(path, tok, bos, eos):
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line)["text"]
            except Exception:
                continue
            ids.append(bos)
            ids.extend(tok.encode(t).ids)
            ids.append(eos)
    return torch.tensor(ids, dtype=torch.long)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = h

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(self.ln1(x))
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, block, n_layer, n_embd, n_head):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def batches(data, block, bs, dev):
    n = (len(data) - 1) // block
    idx = torch.randperm(n)
    for s in range(0, n, bs):
        sel = idx[s:s + bs]
        x = torch.stack([data[i * block:(i + 1) * block] for i in sel])
        y = torch.stack([data[i * block + 1:(i + 1) * block + 1] for i in sel])
        yield x.to(dev), y.to(dev)


def evaluate(model, data, block, bs, dev, use_bf16, max_batches=40):
    model.eval(); tot, nb = 0.0, 0
    with torch.no_grad():
        for x, y in batches(data, block, bs, dev):
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(x, y)
            tot += loss.item(); nb += 1
            if nb >= max_batches: break
    model.train()
    return tot / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="s", choices=list(SIZES))
    ap.add_argument("--train", default="data/opstream_train.jsonl")
    ap.add_argument("--val", default="data/opstream_val.jsonl")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--save-steps", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKPATH) if Path(TOKPATH).exists() else build_tokenizer(a.train)
    bos = tok.token_to_id("<bos>"); eos = tok.token_to_id("<eos>")
    vocab = tok.get_vocab_size()

    train = load_ids(a.train, tok, bos, eos)
    val = load_ids(a.val, tok, bos, eos)
    print(f"size={a.size} dev={dev} bf16={use_bf16} vocab={vocab} train_tok={len(train):,} val_tok={len(val):,}")

    n_layer, n_embd, n_head = SIZES[a.size]
    model = GPT(vocab, a.block, n_layer, n_embd, n_head).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params={nparam/1e6:.1f}M  ({n_layer}L {n_embd}d {n_head}h)")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps_per_epoch = (len(train) - 1) // a.block // a.bs
    total_steps = min(a.max_steps, steps_per_epoch * a.epochs)
    print(f"steps/epoch~{steps_per_epoch}  total_steps={total_steps}")

    out = f"models/opstream_{a.size}"; os.makedirs(out, exist_ok=True)
    ckpt = f"{out}/ckpt.pt"
    step = 0
    if a.resume and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]
        print(f"RESUMED {a.size} at step {step}")
    t0 = time.time()
    done = step >= total_steps
    for ep in range(a.epochs):
        if done:
            break
        for x, y in batches(train, a.block, a.bs, dev):
            lr = a.lr * 0.5 * (1 + math.cos(math.pi * step / total_steps)) if step < total_steps else a.lr * 0.05
            for g in opt.param_groups: g["lr"] = lr
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(x, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
            if step % 100 == 0:
                print(f"  step{step}/{total_steps} loss {loss.item():.3f} lr {lr:.1e} {time.time()-t0:.0f}s")
            if step % a.save_steps == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt)
            if step >= total_steps:
                done = True; break
        if done: break

    vloss = evaluate(model, val, a.block, a.bs, dev, use_bf16)
    torch.save(model.state_dict(), f"{out}/model.pt")
    if os.path.exists(ckpt):
        os.remove(ckpt)
    json.dump({"size": a.size, "vocab": vocab, "block": a.block,
               "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head}, open(f"{out}/config.json", "w"))
    rec = {"size": a.size, "params_M": round(nparam/1e6, 2), "steps": step,
           "train_loss": round(loss.item(), 4), "val_loss": round(vloss, 4),
           "train_tok": len(train), "secs": round(time.time()-t0)}
    with open("models/opstream_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"DONE {rec} -> {out}")


if __name__ == "__main__":
    main()
