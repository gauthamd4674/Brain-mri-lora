"""
Phase 7 — Inference: generate Brain MRI images from text prompts.
Loads base SD v1.5 (local cache) + LoRA weights and runs DPM++ denoising.

Usage:
    python src/inference/generate.py
    python src/inference/generate.py --prompt "Brain MRI showing glioma tumor" --steps 30
    python src/inference/generate.py --all-classes    # generate one per class
"""

import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR    = os.path.join(BASE_DIR, "output", "checkpoints")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output", "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Using locally-cached SD v1.5 (already downloaded — no network needed).
_HF_CACHE  = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
MODEL_ID   = os.path.join(
    _HF_CACHE,
    "models--runwayml--stable-diffusion-v1-5",
    "snapshots",
    "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
)
LOCAL_ONLY  = True

# ── Default prompts per class ──────────────────────────────────────────────
CLASS_PROMPTS = {
    "glioma":      "Brain MRI scan showing glioma tumor, high-grade malignant growth, axial view, T2-weighted",
    "meningioma":  "Brain MRI showing meningioma, benign tumor on brain membrane, coronal view, contrast enhanced",
    "pituitary":   "Brain MRI with pituitary tumor near the base of the skull, sagittal view, T1-weighted",
    "notumor":     "Normal brain MRI scan, healthy brain tissue, no tumor detected, axial view, T2-weighted",
}


def _load_lora_into_unet(unet, lora_path: str):
    """
    Load PEFT-trained LoRA weights (safetensors) into a plain pipeline UNet.
    PEFT saves keys as 'base_model.model.<layer>.lora_A/B.weight'.
    We strip the prefix and inject directly into the UNet state dict.
    """
    from safetensors.torch import load_file
    from peft import LoraConfig, get_peft_model

    # Inject LoRA adapter structure first (same config as training)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,   # no dropout at inference
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)

    # Load the saved weights and apply them
    lora_sd = load_file(lora_path)
    missing, unexpected = unet.load_state_dict(lora_sd, strict=False)
    print(f"  LoRA loaded: {len(lora_sd)} tensors | "
          f"missing={len(missing)} | unexpected={len(unexpected)}")
    return unet


def load_pipeline(lora_path: str = None, device: str = "cuda"):
    """Load SD v1.5 pipeline (local cache) with optional LoRA weights."""
    print("Loading SD v1.5 pipeline from local cache...")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=LOCAL_ONLY,
    ).to(device)

    # Use DPM++ solver — faster than DDIM, same quality in fewer steps
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA weights: {lora_path}")
        pipe.unet = _load_lora_into_unet(pipe.unet, lora_path)
        pipe.unet = pipe.unet.to(device, dtype=torch.float16)
        print("LoRA applied successfully!")
    else:
        print("No LoRA weights — using base SD only" if not lora_path
              else f"LoRA not found at {lora_path} — using base SD only")

    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str = "blurry, low quality, artifacts, distorted, cartoon, painting",
    steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> Image.Image:
    """Run inference and return PIL image."""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt          = prompt,
        negative_prompt = negative_prompt,
        num_inference_steps = steps,
        guidance_scale  = guidance_scale,
        generator       = generator,
        height          = 512,
        width           = 512,
    )
    return result.images[0]


def save_image(img: Image.Image, label: str = "generated") -> str:
    """Save image with timestamp and return path."""
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{label}_{ts}.png")
    img.save(path)
    print(f"  [SAVED] -> {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Brain MRI images")
    parser.add_argument("--prompt",      type=str, default=CLASS_PROMPTS["glioma"],
                        help="Text prompt for generation")
    parser.add_argument("--lora-path",   type=str, default=None,
                        help="Path to LoRA safetensors (leave empty to auto-find best)")
    parser.add_argument("--steps",       type=int, default=30)
    parser.add_argument("--guidance",    type=float, default=7.5)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--all-classes", action="store_true",
                        help="Generate one image per class using default prompts")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Auto-find best LoRA checkpoint if not specified
    lora_path = args.lora_path
    if lora_path is None:
        best = os.path.join(CKPT_DIR, "lora_best.safetensors")
        lora_path = best if os.path.exists(best) else None

    pipe = load_pipeline(lora_path=lora_path, device=device)

    if args.all_classes:
        print("\n[Generating] one image per class...\n")
        for label, prompt in CLASS_PROMPTS.items():
            print(f"  -> {label}: {prompt[:60]}...")
            img = generate_image(pipe, prompt, steps=args.steps,
                                 guidance_scale=args.guidance, seed=args.seed)
            save_image(img, label=label)
    else:
        print(f"\n[Generating] {args.prompt[:80]}...")
        img = generate_image(pipe, args.prompt, steps=args.steps,
                             guidance_scale=args.guidance, seed=args.seed)
        save_image(img, label="custom")

    print("\n[Done] Check output/generated/")
