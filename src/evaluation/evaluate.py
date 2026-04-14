"""
Phase 8 — Evaluation
Computes CLIP similarity score between generated images and their text prompts.
Generates a 4x4 qualitative grid of MRI images per class.

Usage:
    python src/evaluation/evaluate.py
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.inference.generate import load_pipeline, generate_image, CLASS_PROMPTS

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR   = os.path.join(BASE_DIR, "output", "checkpoints")
EVAL_DIR   = os.path.join(BASE_DIR, "output", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def compute_clip_similarity(images: list, prompts: list, device: str = "cuda") -> list:
    """
    Compute cosine similarity between each image and its prompt using CLIP.
    Returns list of float scores in [0, 1].
    """
    print("[CLIP] Computing similarity scores...")
    model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    scores = []
    for img, prompt in zip(images, prompts):
        inputs = processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        score = outputs.logits_per_image.sigmoid().item()
        scores.append(round(score, 4))

    del model
    torch.cuda.empty_cache()
    return scores


def generate_qualitative_grid(pipe, n_per_class: int = 4, steps: int = 25):
    """Generate n_per_class images per MRI class and save a grid."""
    classes = list(CLASS_PROMPTS.keys())
    n_rows  = len(classes)

    fig, axes = plt.subplots(n_rows, n_per_class, figsize=(n_per_class * 3.5, n_rows * 3.5))
    fig.patch.set_facecolor("#0A0A0A")
    fig.suptitle("Qualitative Evaluation — Brain MRI LoRA Generation",
                 fontsize=14, color="white", fontweight="bold")

    all_images  = []
    all_prompts = []

    for row_idx, cls in enumerate(classes):
        prompt = CLASS_PROMPTS[cls]
        print(f"\n  Generating {n_per_class}× [{cls}]...")

        for col_idx in range(n_per_class):
            seed = 42 + col_idx * 7
            img  = generate_image(pipe, prompt, steps=steps, seed=seed)

            all_images.append(img)
            all_prompts.append(prompt)

            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.set_facecolor("#0A0A0A")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col_idx == 0:
                ax.set_ylabel(cls.capitalize(), color="#A78BFA", fontsize=11,
                              fontweight="bold", rotation=90, labelpad=10)

    plt.tight_layout()
    out_path = os.path.join(EVAL_DIR, "qualitative_grid.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0A0A0A")
    plt.close()
    print(f"\n  [EVAL] Qualitative grid -> {out_path}")

    return all_images, all_prompts


def save_scores_report(classes: list, scores: list):
    """Save CLIP scores to CSV and print summary."""
    # Average per class (4 images per class)
    n = len(scores) // len(classes)
    class_avgs = []
    for i, cls in enumerate(classes):
        cls_scores = scores[i*n:(i+1)*n]
        class_avgs.append({"class": cls, "avg_clip_score": round(sum(cls_scores)/len(cls_scores), 4),
                            "min": min(cls_scores), "max": max(cls_scores)})

    df = pd.DataFrame(class_avgs)
    csv_path = os.path.join(EVAL_DIR, "clip_scores.csv")
    df.to_csv(csv_path, index=False)

    print("\n[CLIP] Similarity Scores:")
    print(df.to_string(index=False))
    print(f"\n  Overall avg: {df['avg_clip_score'].mean():.4f}")
    print(f"  Saved -> {csv_path}")
    return df


def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-find LoRA
    lora_path = os.path.join(CKPT_DIR, "lora_best.safetensors")
    if not os.path.exists(lora_path):
        print("[WARN] lora_best.safetensors not found. Run training first.")
        lora_path = None

    pipe = load_pipeline(lora_path=lora_path, device=device)

    print("\n[Evaluation] Running...\n")
    images, prompts = generate_qualitative_grid(pipe, n_per_class=4, steps=25)

    scores = compute_clip_similarity(images, prompts, device=device)
    save_scores_report(list(CLASS_PROMPTS.keys()), scores)

    print("\n[Done] Evaluation complete! Check output/evaluation/")


if __name__ == "__main__":
    run_evaluation()
