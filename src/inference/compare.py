"""
Phase 7 — Side-by-side comparison: Base SD vs LoRA fine-tuned
Generates a grid showing base model output vs LoRA output for each MRI class.

Usage:
    python src/inference/compare.py
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.inference.generate import load_pipeline, generate_image, CLASS_PROMPTS

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR   = os.path.join(BASE_DIR, "output", "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_comparison_grid(steps: int = 30, seed: int = 42):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-find best LoRA
    lora_path = os.path.join(CKPT_DIR, "lora_best.safetensors")
    if not os.path.exists(lora_path):
        print("⚠️  No LoRA checkpoint found. Run training first.")
        print("    Generating base-only comparison instead...")
        lora_path = None

    classes = list(CLASS_PROMPTS.keys())
    n_cols  = len(classes)

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 10))
    fig.patch.set_facecolor("#0D0D0D")
    fig.suptitle("Brain MRI Generation — Base SD vs LoRA Fine-Tuned",
                 fontsize=16, color="white", fontweight="bold", y=0.98)

    row_labels = ["Base SD v1.5", "LoRA Fine-Tuned"]
    configs    = [(None, "Base SD"), (lora_path, "LoRA")]

    for row_idx, (lp, row_label) in enumerate(configs):
        pipe = load_pipeline(lora_path=lp, device=device)
        print(f"\n🎨 Generating [{row_label}] images...")

        for col_idx, cls in enumerate(classes):
            prompt = CLASS_PROMPTS[cls]
            print(f"  → {cls}")
            img = generate_image(pipe, prompt, steps=steps, seed=seed)

            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.set_facecolor("#0D0D0D")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row_idx == 0:
                ax.set_title(cls.capitalize(), color="white", fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], color="#6C63FF", fontsize=11,
                              fontweight="bold", rotation=90, labelpad=10)

        del pipe
        torch.cuda.empty_cache()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_DIR, "comparison_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close()
    print(f"\n✅ Comparison grid saved → {out_path}")
    return out_path


if __name__ == "__main__":
    make_comparison_grid(steps=30, seed=42)
