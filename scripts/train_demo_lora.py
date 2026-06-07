#!/usr/bin/env python3
"""
LoRA SFT demo — fine-tune a small code-LLM to generate Roblox games (serialized text)
from data/demo_small.jsonl. Proof-of-concept (text path, NOT the geometry-modality model).

Manual training loop (only needs transformers + peft + torch) so it's stable across versions
and runs unattended on a cheap cloud GPU. Trains a LoRA adapter on the full game documents.

Usage:
  # validate the loop locally (tiny model, CPU, 2 steps) — do this BEFORE spending cloud $:
  python scripts/train_demo_lora.py --smoke --data data/demo_small.jsonl --out models/demo_smoke

  # real run (on the cheap cloud GPU):
  python scripts/train_demo_lora.py --model Qwen/Qwen2.5-Coder-1.5B \
      --data data/demo_small.jsonl --out models/demo_lora --epochs 3
"""
import argparse, json, math, os, time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


class JsonlText(Dataset):
    def __init__(self, path, tok, ctx, max_samples=0):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    continue
                if t and t.strip():
                    self.rows.append(t)
                if max_samples and len(self.rows) >= max_samples:
                    break
        self.tok, self.ctx = tok, ctx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids = self.tok(self.rows[i], truncation=True, max_length=self.ctx - 1)["input_ids"]
        ids.append(self.tok.eos_token_id)               # learn to STOP after [/GAME]
        return torch.tensor(ids, dtype=torch.long)


def collate(batch, pad_id):
    m = max(len(x) for x in batch)
    inp = torch.full((len(batch), m), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), m), -100, dtype=torch.long)        # -100 = ignored in loss
    for i, x in enumerate(batch):
        inp[i, :len(x)] = x
        lab[i, :len(x)] = x
    return inp, lab


def save_ckpt(model, tok, opt, step, out):
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    torch.save({"opt": opt.state_dict(), "step": step}, os.path.join(out, "trainer_state.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--data", default="data/demo_small.jsonl")
    ap.add_argument("--out", default="models/demo_lora")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=50, help="checkpoint every N optimizer steps (resumable)")
    ap.add_argument("--attn", default="sdpa",
                    help="attn_implementation; use 'eager' on ROCm to avoid AOTriton kernel recompiles on variable-length batches")
    ap.add_argument("--resume", action="store_true", help="resume from adapter+optimizer state in --out")
    ap.add_argument("--smoke", action="store_true", help="tiny model + CPU + 2 steps to validate the loop")
    a = ap.parse_args()

    if a.smoke:
        a.model = "hf-internal-testing/tiny-random-LlamaForCausalLM"   # same proj names as Qwen
        a.ctx, a.epochs, a.batch, a.accum, a.max_samples = 256, 1, 2, 1, 4

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device={dev}  bf16={use_bf16}  model={a.model}")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        attn_implementation=a.attn)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear")
    resume_ok = a.resume and os.path.exists(os.path.join(a.out, "adapter_config.json"))
    if resume_ok:
        model = PeftModel.from_pretrained(model, a.out, is_trainable=True)
        print(f"resumed adapter from {a.out}")
    else:
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to(dev)

    ds = JsonlText(a.data, tok, a.ctx, a.max_samples)
    if len(ds) == 0:
        print(f"no samples in {a.data}"); return
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    print(f"samples={len(ds)}  optimizer-steps/epoch~={math.ceil(len(dl) / a.accum)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    step = 0
    if resume_ok and os.path.exists(os.path.join(a.out, "trainer_state.pt")):
        st = torch.load(os.path.join(a.out, "trainer_state.pt"), map_location=dev)
        opt.load_state_dict(st["opt"]); step = st.get("step", 0)
        print(f"resumed optimizer at step {step}")
    model.train()
    t0 = time.time()
    for ep in range(a.epochs):
        opt.zero_grad()
        for i, (inp, lab) in enumerate(dl):
            inp, lab = inp.to(dev), lab.to(dev)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(input_ids=inp, labels=lab).loss / a.accum
            loss.backward()
            if (i + 1) % a.accum == 0 or (i + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 5 == 0 or a.smoke:
                    print(f"ep{ep} step{step} loss {loss.item() * a.accum:.3f}  {time.time() - t0:.0f}s")
                if step % a.save_steps == 0:
                    save_ckpt(model, tok, opt, step, a.out)
                    print(f"  [checkpoint @ step {step}]")
        print(f"=== epoch {ep + 1}/{a.epochs} done ({time.time() - t0:.0f}s) ===")

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved LoRA adapter -> {a.out}")
    if a.smoke:
        print("SMOKE OK")


if __name__ == "__main__":
    main()
