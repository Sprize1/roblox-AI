# Training Specifications

## Model Configurations

| Config | Params | Layers | Hidden | VRAM (FP32) | VRAM (mixed) | GPU |
|--------|--------|--------|--------|-------------|--------------|-----|
| **tiny** | ~8M | 4 | 256 | 4 GB | 2 GB | CPU or any |
| **small** | ~45M | 6 | 512 | 8 GB | 4 GB | RTX 3060+ |
| **medium** | ~180M | 8 | 1024 | 16 GB | 8 GB | RTX 3090/4090 |
| **large** | ~600M | 12 | 2048 | 32 GB | 16 GB | A5000, dual 3090 |
| **xl** | ~2B | 16 | 3072 | 80 GB | 40 GB | A100 80GB |
| **default** | ~17B | 24 | 4096 | 480 GB | 240 GB | 8×A100 cluster |

## Training Time Estimates (small config, 375 samples)

| Hardware | Batch | Epochs | Time |
|----------|-------|--------|------|
| CPU (Ryzen 5950X) | 4 | 10 | ~2-4 hours |
| RTX 3060 12GB | 8 | 10 | ~20-40 min |
| RTX 4090 24GB | 16 | 10 | ~10-20 min |
| A100 80GB (cloud) | 64 | 100 | ~1-2 hours |

## Recommended First Training

```bash
# Step 1: Validate pipeline with tiny config (CPU, 2 min)
cargo run --bin roblox-ai -- train \
  --dataset data/full_dataset \
  --epochs 2 \
  --batch-size 4 \
  --config config/tiny.toml

# Step 2: Real training with small config (GPU, 30 min)
cargo run --bin roblox-ai -- train \
  --dataset data/full_dataset \
  --epochs 10 \
  --batch-size 8 \
  --config config/small.toml

# Step 3: Scale up (cloud GPU)
# Same command with config/large.toml on Lambda Labs / RunPod
```

## Cloud GPU Options

| Provider | GPU | VRAM | Price/hr |
|----------|-----|------|----------|
| Lambda Labs | A100 80GB | 80 GB | $1.10 |
| RunPod | A100 80GB | 80 GB | $1.69 |
| RunPod | RTX 4090 | 24 GB | $0.69 |
| Vast.ai | RTX 3090 | 24 GB | $0.30 |
| Lambda Labs | H100 80GB | 80 GB | $2.49 |

## AMD GPU (your local)

Your AMD GPU can be used via WGPU backend.
- Performance: ~60-80% of equivalent NVIDIA (Vulkan overhead)
- Supported: yes, via `--features wgpu`
- Recommendation: start with `small` config, batch_size=4

```bash
cargo build --release --features wgpu
cargo run --bin roblox-ai -- train --dataset data/full_dataset --epochs 10 --batch-size 4
```
