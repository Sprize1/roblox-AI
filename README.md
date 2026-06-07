# Roblox game generation, a research log

I spent a while trying to generate complete Roblox games (the 3D part layout plus the Luau logic
that goes with it) from a text prompt. This repo is the honest log of that: what I tried, what
worked, what didn't, and the walls I hit, with the actual numbers.

The bet I started from: the hard and valuable part isn't writing the scripts (any decent LLM
already writes fine Luau). It's getting the geometry and the logic to actually fit together at the
level of a whole game.

## What I found

Text LLMs are spatially blind. They happily output coordinates that look fine as numbers but are
absurd once you render them: parts floating, overlapping, unreachable. A 4-year-old spots it
instantly, the model never does, because it only ever sees tokens and never the 3D result.
Fine-tuning on serialized text does not fix that.

A 3D-native generator does much better. A small diffusion model over sets of box primitives
(positions and sizes, with attention across the parts) actually learns the real layout
distribution where the text model fails. Measured deep-overlap was 0.34 (real games sit around
0.33) versus 0.74 for the text model, and it kept part sizes realistic instead of cheating. This
was the one clearly positive result.

The real wall turned out to be data, and I can put numbers on it. Usable scraped games are big
(median around 2285 parts, ~1000 studs across) and there aren't many (~1774 uncopylocked total,
only ~141 with 256 parts or fewer). The geometry/logic coupling signal is sparse too: about 0.6
object references per script, and ~800 resolvable cross-part references in the whole corpus. That
is nowhere near enough to learn the binding.

Geometric checkers help, but the metric gets gamed. A deterministic SAT overlap/reachability
checker computes spatial truth the model can't. Using it to filter training samples gave a clean
+62% on synthetic data. On real data the naive "reduce overlap" target got gamed: the model just
shrank the parts. Exact is not the same as useful if the target is gameable. Lesson learned the
hard way, always check the dumb baseline first: an earlier "binding" metric looked great until I
realized a trivial copy-the-last-id heuristic scored 100% on it.

Getting more data legitimately means going through Roblox. Scraping free models to build a
training set runs against the Terms of Use (automated access, asset extraction) and the copyright
on creator content (free models are licensed for reuse inside Roblox, not as ML training data). So
the data that would make a "wow" demo possible isn't something you can get from the outside.

## How it went, in order

0. First from-scratch GPT on flat text. A plain decoder transformer trained on per-part absolute
   coordinates written as text. It just enumerates and repeats instead of composing anything
   coherent. I dropped it, but it's what pushed me toward everything after.

1. Two-stage LLM, layout then scripts. LoRA on Qwen2.5-Coder. The scripts came out valid and
   on-topic, but that's the easy half. The layout was monotone with no real course, and dropped
   into Studio it wasn't playable. Confirmed the scripts aren't the moat, the layout is.

2. Unified op-stream. The idea: a game is really a program/trace. The static workspace is t=0, and
   a runtime `Instance.new` is the same `CREATE` op at t>0; the missing axis is time/causality.
   `luau_lower.py` turns Luau into that op-stream (99.9% of statements map to an op, 79% with fully
   closed-form values). Training scaled on loss, but the cross-modal binding metric was degenerate
   and the real cross-part references were too sparse to learn.

3. Checker and grounding. `checker.py` does exact oriented-box overlap (separating axis theorem),
   penetration depth, reachability. `ground_experiment.py` runs a filter-and-retrain A/B.
   Synthetic gave +62%, real Roblox got gamed by part shrinkage.

4. 3D-native diffusion, the part that worked. `diffusion_layout.py` and `diffusion_whole.py`: a
   DDPM with a transformer denoiser over a set of box primitives (variable count via a presence
   channel), augmented with 90-degree rotations and mirrors. It matches the real spatial
   distribution with sizes intact. On whole games it stays coherent and isn't just memorizing the
   training set, but it's messy and limited to the small-game tail because of the data wall.

## Repo map

| Area | Scripts |
|---|---|
| Parsing (.rbxl/.rbxlx to structured) | `rbxlx_parser.py`, `build_structured_v3.py`, `convert_binary_places.py` |
| Unified op-stream | `luau_lower.py`, `serialize_game.py`, `build_opstream_dataset.py`, `analyze_dsl_coverage.py`, `lower_scripts.py` |
| From-scratch models | `train_opstream.py` (GPT), `diffusion_layout.py`, `diffusion_whole.py`, `point_cloud.py` |
| Geometric checker and metrics | `checker.py`, `eval_opstream.py`, `eval_synth.py` |
| Grounding / data experiments | `ground_experiment.py`, `make_synth.py`, `make_real_layout.py`, `make_demo_chunks.py`, `make_script_pairs.py` |
| Two-stage demo (early) | `train_demo_lora.py`, `generate_phase2.py`, `generate_game.py`, `generate_demo.py`, `make_demo_obby.py` |
| Studio injection (viz) | `make_inject_luau.py`, `make_whole_inject.py` |

## Data

No data, models, or assets are in here (see `.gitignore`). The structured datasets come from
scraped Roblox places, which are creator IP, so redistributing them would be exactly the copyright
problem this project concluded against. The code shows how the data was built; reproducing it
needs your own legally-sourced data.

## Where it landed

3D-native is the right direction for spatial coherence. It learns structure where text is blind,
and it does it without needing a ton of data. But generating a coherent, playable, whole game is
gated on data (games are big and few, the logic coupling is sparse) that you can't legitimately get
from outside Roblox. What's worth keeping here is the method and the clearly mapped walls, not a
finished generator.

## Thanks

To the archive maintainers who preserve Roblox history. Used here for research only, nothing is
redistributed:

- [RobloxRBXLArchive](https://github.com/LuaGunsX/RobloxRBXLArchive) by LuaGunsX
- [Biggest Uncopylocked Library](https://github.com/KH0DIN/Biggest_Uncopylocked_Roblox_Games_Library) by KH0DIN
- [Roblox Uncopylocked Games](https://github.com/IIIStatusIII/Roblox-Uncopylocked-Games) by IIIStatusIII
- [RBXLArchive-RRU](https://github.com/ZwDaNk/RBXLArchive-RRU) by ZwDaNk
- Extraction tooling: [Roblox RBXL Extractor](https://github.com/pinkythegawd/Roblox-rbxl-extractor) by pinkythegawd

## License

MIT, see `LICENSE`. Code only, no Roblox content.
