#!/usr/bin/env python
"""Train GR00T N1.7-3B on custom LeRobot dataset.

Usage:
    conda activate membench
    python scripts/train/train_gr00t_n1_7.py
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGES

# ==================== CONFIG ====================
DATASET_PATH = "/home/chensihang/membench/data/wa01_200seeds_v061"
MODEL_PATH = "/home/chensihang/pretrained_models/GR00T-N1.7-3B"
OUTPUT_DIR = Path("./outputs/gr00t_n1_7_wa01")

# Training hyperparams
BATCH_SIZE = 4  # per GPU, adjust based on VRAM
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 50
MAX_STEPS = 50000  # or set NUM_EPOCHS-based
LR = 1e-4
WARMUP_STEPS = 500
SAVE_EVERY = 5000
LOG_EVERY = 10
USE_WANDB = True

# Model tuning flags (start conservative)
TUNE_LLM = True       # Whether to fine-tune the Qwen3-VL backbone LLM
TUNE_VISUAL = True    # Whether to fine-tune the vision encoder
TUNE_PROJECTOR = True # Whether to fine-tune the vision-language projector
TUNE_DIFFUSION = True # Whether to fine-tune the diffusion action head
TUNE_VLLN = True      # Whether to fine-tune the vision-language LayerNorm

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ==================== SETUP ====================
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Load dataset metadata
print("=" * 60)
print("Loading dataset metadata...")
print("=" * 60)
ds_meta = LeRobotDatasetMetadata(DATASET_PATH, root=DATASET_PATH)
print(f"  Dataset: {DATASET_PATH}")
print(f"  Episodes: {ds_meta.num_episodes}")
print(f"  Frames: {ds_meta.num_frames}")
print(f"  FPS: {ds_meta.fps}")
print(f"  Features: {list(ds_meta.features.keys())}")
print(f"  State dim: {ds_meta.features[OBS_STATE]['shape'][0]}")
print(f"  Action dim: {ds_meta.features[ACTION]['shape'][0]}")
print(f"  Camera keys: {ds_meta.camera_keys}")

# 2. Build input/output features from dataset
features = {
    key: PolicyFeature(
        type=FeatureType.VISUAL if key.startswith(f"{OBS_IMAGES}.") or key.startswith("observation.images.")
        else FeatureType.STATE if key == OBS_STATE
        else FeatureType.ACTION if key == ACTION
        else FeatureType.STATE,
        shape=ds_meta.features[key]["shape"],
    )
    for key in ds_meta.features
    if key.startswith(f"{OBS_IMAGES}.") or key == OBS_STATE or key == ACTION
}
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

# 3. Create GrootConfig
print("\n" + "=" * 60)
print("Creating Groot config...")
print("=" * 60)
config = GrootConfig(
    # Base model
    base_model_path=MODEL_PATH,
    # Embodiment (from the model's embodiment_id.json: robocasa_panda_omron = 13)
    embodiment_tag="robocasa_panda_omron",
    # Dimensions
    max_state_dim=132,    # GR00T N1.7 max
    max_action_dim=132,   # GR00T N1.7 max
    chunk_size=40,         # action_horizon from model config
    n_action_steps=40,
    # Fine-tuning flags
    tune_llm=TUNE_LLM,
    tune_visual=TUNE_VISUAL,
    tune_projector=TUNE_PROJECTOR,
    tune_diffusion_model=TUNE_DIFFUSION,
    tune_vlln=TUNE_VLLN,
    # Training params
    optimizer_lr=LR,
    warmup_ratio=0.05,
    use_bf16=True,
    model_params_fp32=True,  # Keep params in fp32 for stable training
    # Dataset
    batch_size=BATCH_SIZE,
    max_steps=MAX_STEPS,
    save_steps=SAVE_EVERY,
    output_dir=str(output_dir),
    report_to="wandb" if USE_WANDB else "none",
    video_backend="pyav",
    resume=False,
    # Input/output features
    input_features=input_features,
    output_features=output_features,
    # Image size (matches GR00T N1.7 defaults)
    image_size=(256, 256),
    # Flash attention
    use_flash_attention=True,
    # Normalization
    normalization_mapping={
        "VISUAL": "identity",
        "STATE": "min_max",
        "ACTION": "min_max",
    },
)
print(f"  embodiment_tag: {config.embodiment_tag}")
print(f"  chunk_size: {config.chunk_size}")
print(f"  tune_llm: {config.tune_llm}")
print(f"  tune_visual: {config.tune_visual}")

# 4. Create policy from pretrained model
print("\n" + "=" * 60)
print("Loading GR00T N1.7-3B model...")
print("=" * 60)
policy = GrootPolicy(config)
policy.train()
policy.to(DEVICE)
print(f"  Model loaded on {DEVICE}")
print(f"  Total params: {sum(p.numel() for p in policy.parameters()):,}")
print(f"  Trainable params: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")

# 5. Create pre/post processors
print("\n" + "=" * 60)
print("Creating processors...")
print("=" * 60)
preprocessor, postprocessor = make_pre_post_processors(
    config, dataset_stats=ds_meta.stats
)
print("  Processors created")

# 6. Compute delta timestamps
delta_timestamps = {
    "observation.state": [0.0],  # current frame
    "action": [i / ds_meta.fps for i in range(config.chunk_size)],
}
# Add camera keys with delta timestamps
for cam_key in ds_meta.camera_keys:
    delta_timestamps[cam_key] = [0.0]

# 7. Create dataset
print("\n" + "=" * 60)
print("Creating dataset...")
print("=" * 60)
dataset = LeRobotDataset(
    DATASET_PATH,
    root=DATASET_PATH,
    delta_timestamps=delta_timestamps,
    video_backend="pyav",
    return_uint8=True,
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)
print(f"  Dataset size: {len(dataset)} frames")
print(f"  Batches per epoch: {len(dataloader)}")

# 8. Optimizer
print("\n" + "=" * 60)
print("Setting up optimizer...")
print("=" * 60)
optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=LR,
    betas=(0.95, 0.999),
    eps=1e-8,
    weight_decay=1e-5,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_STEPS, eta_min=LR * 0.1
)

# 9. WandB
if USE_WANDB:
    import wandb
    wandb.init(
        project="gr00t-n1-7-wa01",
        config={
            "model": "GR00T-N1.7-3B",
            "dataset": "wa01_200seeds_v061",
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "tune_llm": TUNE_LLM,
            "tune_visual": TUNE_VISUAL,
            "tune_projector": TUNE_PROJECTOR,
            "tune_diffusion": TUNE_DIFFUSION,
        },
    )

# ==================== TRAINING LOOP ====================
print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

global_step = 0
accum_loss = 0.0
best_ckpt = None

scaler = torch.amp.GradScaler("cuda", enabled=True)

while global_step < MAX_STEPS:
    for batch in dataloader:
        # Preprocess
        batch = preprocessor(batch)

        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(DEVICE, non_blocking=True)

        # Forward
        with torch.amp.autocast("cuda", dtype=DTYPE):
            loss, loss_dict = policy.forward(batch)

        # Scale loss for gradient accumulation
        loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()

        accum_loss += loss.item()

        if (global_step + 1) % GRAD_ACCUM_STEPS == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if global_step % LOG_EVERY == 0:
            avg_loss = accum_loss / max(LOG_EVERY, 1)
            lr_current = scheduler.get_last_lr()[0]
            print(f"  Step {global_step:>6d}/{MAX_STEPS} | Loss: {avg_loss:.4f} | LR: {lr_current:.2e}")
            if USE_WANDB:
                wandb.log({
                    "loss": avg_loss,
                    "lr": lr_current,
                    "step": global_step,
                }, step=global_step)
            accum_loss = 0.0

        # Sync scheduler step with optimizer step
        if (global_step + 1) % GRAD_ACCUM_STEPS == 0:
            scheduler.step()

        # Save checkpoint
        if (global_step + 1) % SAVE_EVERY == 0:
            ckpt_dir = output_dir / f"checkpoint-{global_step + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            preprocessor.save_pretrained(ckpt_dir)
            postprocessor.save_pretrained(ckpt_dir)
            # Save optimizer state
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
            }, ckpt_dir / "training_state.pt")
            print(f"  ✓ Checkpoint saved: {ckpt_dir}")

        global_step += 1
        if global_step >= MAX_STEPS:
            break

# Save final checkpoint
final_dir = output_dir / "checkpoint-final"
final_dir.mkdir(parents=True, exist_ok=True)
policy.save_pretrained(final_dir)
preprocessor.save_pretrained(final_dir)
postprocessor.save_pretrained(final_dir)
print(f"\n✓ Final checkpoint saved: {final_dir}")

if USE_WANDB:
    wandb.finish()

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)