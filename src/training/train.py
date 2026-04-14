"""
Phase 5 & 6 -- Core Training Loop
Fine-tunes Stable Diffusion v1.5 with LoRA on Brain MRI images.
Optimized for 6 GB VRAM (RTX 4050 Laptop).

Strategy: VAE + text_encoder in fp16 (frozen), UNet LoRA params in fp32.
Autocast only over the forward pass; optimizer works on fp32 LoRA params.
This avoids the 'Attempting to unscale FP16 gradients' error.

Run:
    python src/training/train.py
    python src/training/train.py --max-steps 10    # quick smoke test
    python src/training/train.py --epochs 20       # full training
"""

import os
import csv
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from safetensors.torch import save_file

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset import get_dataloaders

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR   = os.path.join(BASE_DIR, "output", "checkpoints")
PLOTS_DIR  = os.path.join(BASE_DIR, "output", "plots")
LOG_PATH   = os.path.join(BASE_DIR, "output", "training_log.csv")
os.makedirs(CKPT_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Using locally-cached SD v1.5 (already fully downloaded — no network needed).
_HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
MODEL_ID   = os.path.join(
    _HF_CACHE,
    "models--runwayml--stable-diffusion-v1-5",
    "snapshots",
    "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
)
LOCAL_ONLY = True   # always True — loading from local cache

# ── Training config ────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "epochs":            3,       # 3 epochs ≈ 3.5 hrs on RTX 4050 — good for assignment
    "batch_size":        1,
    "grad_accum_steps":  4,       # effective batch = 4
    "learning_rate":     1e-4,
    "weight_decay":      1e-2,
    "max_grad_norm":     1.0,
    "save_every_epochs": 1,       # checkpoint after every epoch
    "vae_scale":         0.18215,
}


def get_lora_state_dict(unet):
    """Extract only LoRA weights from the UNet for checkpoint saving."""
    return {k: v for k, v in unet.state_dict().items() if "lora" in k.lower()}


def save_checkpoint(unet, epoch: int, loss: float):
    path = os.path.join(CKPT_DIR, f"lora_epoch_{epoch:03d}.safetensors")
    save_file(get_lora_state_dict(unet), path)
    print(f"  [SAVED] lora_epoch_{epoch:03d}.safetensors  loss={loss:.4f}")


def plot_loss(losses: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(losses, color="#6C63FF", linewidth=1.5, alpha=0.8)
        # Rolling average overlay
        if len(losses) >= 20:
            import pandas as pd
            rolled = pd.Series(losses).rolling(20, min_periods=1).mean()
            plt.plot(rolled, color="#FF6584", linewidth=2, label="Rolling avg (20)")
            plt.legend()
        plt.title("Training Loss -- Brain MRI LoRA Fine-Tuning", fontsize=14, fontweight="bold")
        plt.xlabel("Training Step")
        plt.ylabel("MSE Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, "loss_curve.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  [PLOT] Loss curve -> {out}")
    except Exception as e:
        print(f"  [WARN] Could not save plot: {e}")


def load_models_for_training(device):
    """
    Load SD v1.5 components with the correct dtype strategy:
    - VAE:          fp16, frozen
    - Text encoder: fp16, frozen  (CLIP ViT-L/14, 768-dim)
    - UNet:         fp32 (so LoRA params stay in fp32 for stable training)
    """
    print("[Loading] Tokenizer + Text Encoder — CLIP ViT-L/14 (fp16, frozen)...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer",
                                              local_files_only=LOCAL_ONLY)
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16,
        local_files_only=LOCAL_ONLY
    ).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    print("[Loading] VAE (fp16, frozen)...")
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float16,
        local_files_only=LOCAL_ONLY
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    print("[Loading] UNet (fp32, LoRA will be injected)...")
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float32,
        local_files_only=LOCAL_ONLY
    ).to(device)
    unet.requires_grad_(False)

    # Inject LoRA -- only these adapters are trainable (fp32)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    # Gradient checkpointing on the underlying UNet to save VRAM
    unet.base_model.model.enable_gradient_checkpointing()

    print("[Loading] Noise Scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler",
                                                    local_files_only=LOCAL_ONLY)

    print("[OK] All models ready.\n")
    return vae, unet, text_encoder, tokenizer, noise_scheduler


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[START] Training on: {device}")
    print(f"  Batch size:     {cfg['batch_size']} (effective {cfg['batch_size'] * cfg['grad_accum_steps']})")
    print(f"  Epochs:         {cfg['epochs']}")
    print(f"  Learning rate:  {cfg['learning_rate']}\n")

    # Load models
    vae, unet, text_encoder, tokenizer, noise_scheduler = load_models_for_training(device)

    # DataLoaders
    train_loader, _ = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=cfg["batch_size"],
        num_workers=0,   # required on Windows
    )

    # Only LoRA params are trained -- they're in fp32
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer  = AdamW(trainable_params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    total_steps = (len(train_loader) // cfg["grad_accum_steps"]) * cfg["epochs"]
    scheduler  = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

    # autocast for forward pass (VAE + UNet inference in fp16, loss in fp32)
    scaler = torch.cuda.amp.GradScaler()

    # CSV log
    log_file   = open(LOG_PATH, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "global_step", "loss", "lr"])

    all_losses  = []
    global_step = 0
    best_loss   = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        unet.train()
        epoch_losses = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg['epochs']}", leave=True)

        for step_idx, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)   # fp32 from dataset
            input_ids    = batch["input_ids"].to(device)

            # Forward pass inside autocast
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # 1. Encode image to latent (VAE in fp16)
                with torch.no_grad():
                    latents = vae.encode(pixel_values.half()).latent_dist.sample()
                    latents = latents * cfg["vae_scale"]

                # 2. Sample noise + timestep
                noise     = torch.randn_like(latents)
                bsz       = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=device
                ).long()

                # 3. Forward diffusion: add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 4. Text embeddings (CLIP in fp16)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # 5. UNet noise prediction (LoRA params cast to fp16 inside autocast)
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

            # 6. Loss in fp32
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = loss / cfg["grad_accum_steps"]

            # 7. Scaled backprop
            scaler.scale(loss).backward()

            if (step_idx + 1) % cfg["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            loss_val = loss.item() * cfg["grad_accum_steps"]
            epoch_losses.append(loss_val)
            all_losses.append(loss_val)

            current_lr = optimizer.param_groups[0]["lr"]
            log_writer.writerow([epoch, global_step, f"{loss_val:.6f}", f"{current_lr:.2e}"])
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{current_lr:.1e}",
                              "step": global_step})

            if cfg.get("max_steps") and global_step >= cfg["max_steps"]:
                print(f"\n  [STOP] Reached max_steps={cfg['max_steps']}.")
                break

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(f"  Epoch {epoch:02d} avg loss: {avg_loss:.4f}")

        # Checkpoint
        if epoch % cfg["save_every_epochs"] == 0 or epoch == cfg["epochs"]:
            save_checkpoint(unet, epoch, avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(CKPT_DIR, "lora_best.safetensors")
                save_file(get_lora_state_dict(unet), best_path)
                print(f"  [BEST] New best saved (loss={best_loss:.4f})")

        if cfg.get("max_steps") and global_step >= cfg["max_steps"]:
            break

    log_file.close()
    plot_loss(all_losses)

    print(f"\n[DONE] Training complete!")
    print(f"  Best loss:   {best_loss:.4f}")
    print(f"  Checkpoints: {CKPT_DIR}")
    print(f"  Log file:    {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA on Brain MRI dataset")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--max-steps",   type=int,   default=None,
                        help="Stop after N optimizer steps (for smoke testing)")
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG}
    cfg["epochs"]        = args.epochs
    cfg["batch_size"]    = args.batch_size
    cfg["learning_rate"] = args.lr
    if args.max_steps:
        cfg["max_steps"] = args.max_steps

    train(cfg)
