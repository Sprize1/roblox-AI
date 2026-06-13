# Roblox game generation, a research log

I'm trying to generate complete Roblox games (the 3D part layout plus the Luau logic that goes with
it) from scratch. This repo is the honest log of that: what I tried, what worked, what didn't, and
the walls I hit, with the actual numbers. It started as a post-mortem and turned back into active
work once I realized the wall I thought had stopped it was the wrong wall (see step 5). So this is a
work in progress, not a closed case.

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

I thought the wall was data, and for the logic half it is. Usable scraped games are big (median
around 2285 parts, ~1000 studs across) and there aren't many (~1774 uncopylocked total). The
geometry/logic coupling signal is genuinely sparse: about 0.6 object references per script, and
~800 resolvable cross-part references in the whole corpus, nowhere near enough to learn the binding.
But for the layout half I was wrong about the cause. I had only been training on the ~141 games
with 256 parts or fewer, and that cap was self-imposed by quadratic attention, not by what data
exists. Lifting it (step 5) unlocked most of the corpus I'd been throwing away.

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
   training set, but it was messy and stuck at the small-game tail (256 parts), which is what step 5
   is about.

5. Sub-quadratic attention, scaling to whole games (in progress). The 256-part cap wasn't the data,
   it was the quadratic attention. Swapping softmax for non-causal linear attention (O(n), about ten
   lines of pure PyTorch, no extra deps, and the right primitive for an unordered set anyway) lifted
   it. Training games went from ~141 to ~906, and generation went from ~155-part fragments to
   coherent full-size games and then to median-scale ones (~2267 parts, around the ~2285 dataset
   median), still novel and not memorizing. A learned per-token gate that suppresses the padding in
   the linear-attention sum tightened the layouts (deep-overlap 0.075 to 0.150). The whole thing
   runs end to end on one consumer GPU (RX 7600 XT, ROCm, about 7 of 16 GB). The honest open problem:
   scale and density pull against each other (more padding at larger context dilutes the gate), so
   the big layouts are structured and correctly oriented but still too loose to play. That is the
   current work: stronger gate, more model capacity, more training. (Aside: I'd assumed the fused
   sub-quadratic kernels in flash-linear-attention couldn't run on ROCm/Windows. They can, Triton
   works there via triton-windows; `fla_shim.py` is the small patch that gets it importing. I went
   with plain PyTorch linear attention here because a set model wants non-causal attention, but the
   FLA path is open for the causal op-stream model.)

## Repo map

| Area | Scripts |
|---|---|
| Parsing (.rbxl/.rbxlx to structured) | `rbxlx_parser.py`, `build_structured_v3.py`, `convert_binary_places.py` |
| Unified op-stream | `luau_lower.py`, `serialize_game.py`, `build_opstream_dataset.py`, `analyze_dsl_coverage.py`, `lower_scripts.py` |
| From-scratch models | `train_opstream.py` (GPT), `diffusion_layout.py`, `diffusion_whole.py` (linear-attention set diffusion, step 5), `point_cloud.py` |
| ROCm / sub-quadratic | `fla_shim.py` (imports flash-linear-attention on ROCm/Windows), `gen_whole.py` |
| Geometric checker and metrics | `checker.py`, `eval_opstream.py`, `eval_synth.py` |
| Grounding / data experiments | `ground_experiment.py`, `make_synth.py`, `make_real_layout.py`, `make_demo_chunks.py`, `make_script_pairs.py` |
| Two-stage demo (early) | `train_demo_lora.py`, `generate_phase2.py`, `generate_game.py`, `generate_demo.py`, `make_demo_obby.py` |
| Studio injection (viz) | `make_inject_luau.py`, `make_whole_inject.py` |

## Data

No data, models, or assets are in here (see `.gitignore`). The structured datasets come from
scraped Roblox places, which are creator IP, so redistributing them would be exactly the copyright
problem this project avoids. The code shows how the data was built; reproducing it needs your own
legally-sourced data.

## Where it is now

3D-native is the right direction for spatial coherence, and sub-quadratic attention turned the part
I'd written off, whole-game scale, into something tractable on a single consumer GPU. The model now
generates novel, full-size, median-scale layouts that are structured and correctly oriented, just
not yet dense or playable enough. So this went from a post-mortem back to active work. The open
fronts: coherence at scale (gate, capacity, diffusion steps), and then the logic coupling, which is
the one part still genuinely starved for data (sparse cross-part references) rather than compute.
Geometry first, binding next.

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
