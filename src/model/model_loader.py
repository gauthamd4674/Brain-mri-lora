"""
Phase 3 & 4 -- Load Stable Diffusion components + inject LoRA into UNet.
Designed for 6 GB VRAM (RTX 4050 Laptop) with fp16 + gradient checkpointing.
"""

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# ── Model ID ───────────────────────────────────────────────────────────────
# stabilityai/stable-diffusion-2-1-base: publicly available, no license gate, 512x512 native
MODEL_ID = "stabilityai/stable-diffusion-2-1-base"


def get_lora_config() -> LoraConfig:
    """
    LoRA targeting all 4 attention projections in every UNet cross-attention layer.
    r=4 keeps it lightweight for 6 GB VRAM; increase to 8 if you have headroom.
    """
    return LoraConfig(
        r=4,                # rank -- lower = fewer params = faster & less VRAM
        lora_alpha=8,       # scaling: alpha/r = 2 (standard ratio)
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",   # all attention projections
        ],
        lora_dropout=0.05,
        bias="none",
    )


def load_models(device: str = "cuda", dtype=torch.float16):
    """
    Downloads (or loads from cache) SD v1.5 components.
    Freezes everything except LoRA adapters.

    Returns:
        vae, unet, text_encoder, tokenizer, noise_scheduler
    """
    print(f"📥 Loading Stable Diffusion v1.5 components to [{device}] ({dtype}) ...")

    # ── Text Encoder + Tokenizer (CLIP) ───────────────────────────────────
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder.requires_grad_(False)   # fully frozen

    # ── VAE ───────────────────────────────────────────────────────────────
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False)            # fully frozen

    # ── UNet (LoRA injected) ───────────────────────────────────────────────
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=dtype
    ).to(device)
    unet.requires_grad_(False)           # freeze base weights

    # Inject LoRA adapters -- only these weights are trainable
    lora_cfg = get_lora_config()
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()   # shows how few params we're training

    # Enable gradient checkpointing on the underlying UNet model to save VRAM
    unet.base_model.model.enable_gradient_checkpointing()

    # ── Noise Scheduler (DDPM) ─────────────────────────────────────────────
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    print("✅ All models loaded successfully!\n")
    return vae, unet, text_encoder, tokenizer, noise_scheduler


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
