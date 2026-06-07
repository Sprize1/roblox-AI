#!/usr/bin/env python3
"""Generate a Roblox game (serialized text) from the demo LoRA model — PoC verdict.

Fixes vs v1: stops at [/GAME], suppresses the base model's control/FIM special tokens
(so <|fim_*|>/<|file_sep|> can't leak), tuned sampling, UTF-8 file output (no cp1252 crash).

Usage:
  python scripts/generate_demo.py --prompt "[GAME: Lava Jump Obby]" --max-new 1200
"""
import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B")
ap.add_argument("--adapter", default="models/demo_lora")
ap.add_argument("--prompt", default="[GAME: Lava Jump Obby]\n[PARTS")
ap.add_argument("--max-new", type=int, default=1200)
ap.add_argument("--temp", type=float, default=0.75)
ap.add_argument("--rep-pen", type=float, default=1.15)
ap.add_argument("--no-repeat-ngram", type=int, default=0)
ap.add_argument("--stop", default="[/GAME]")
ap.add_argument("--out-text", default="models/demo_gen.txt")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.adapter)
base = AutoModelForCausalLM.from_pretrained(
    a.base, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
model = PeftModel.from_pretrained(base, a.adapter).to(dev).eval()

# Ban the base model's control/FIM tokens (keep bos/eos/pad) so they can't leak into output.
keep = {tok.eos_token_id, tok.pad_token_id, tok.bos_token_id}
ban = {i for i in tok.all_special_ids if i not in keep}
for t, i in tok.get_added_vocab().items():
    if i not in keep and ("<|" in t or "fim" in t.lower() or "file_sep" in t or "repo_name" in t):
        ban.add(i)
bad_words = [[i] for i in sorted(ban)] or None


class StopOnText(StoppingCriteria):
    def __init__(self, tokenizer, stop_str, prompt_len):
        self.tok, self.stop, self.plen = tokenizer, stop_str, prompt_len

    def __call__(self, input_ids, scores, **kw):
        return self.stop in self.tok.decode(input_ids[0][self.plen:], skip_special_tokens=True)


ids = tok(a.prompt, return_tensors="pt").to(dev)
plen = ids["input_ids"].shape[1]
with torch.no_grad():
    out = model.generate(
        **ids, max_new_tokens=a.max_new, do_sample=True, temperature=a.temp,
        top_p=0.9, repetition_penalty=a.rep_pen, no_repeat_ngram_size=a.no_repeat_ngram,
        bad_words_ids=bad_words,
        stopping_criteria=StoppingCriteriaList([StopOnText(tok, a.stop, plen)]),
        pad_token_id=tok.pad_token_id or tok.eos_token_id)

gen = tok.decode(out[0][plen:], skip_special_tokens=True)
with open(a.out_text, "w", encoding="utf-8") as f:
    f.write("=== PROMPT ===\n" + a.prompt + "\n=== GENERATED ===\n" + gen)
print(f"banned {len(bad_words or [])} control tokens | wrote {len(gen)} chars -> {a.out_text}")
