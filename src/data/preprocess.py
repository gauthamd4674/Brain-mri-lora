# -*- coding: utf-8 -*-
"""
Phase 2 -- Dataset Preprocessing
Reads images from Brain MRI Dataset folder, builds metadata.csv,
resizes to 512x512, converts to RGB, and saves to dataset/processed/
"""

import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR      = os.path.join(BASE_DIR, "Brain MRI Dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed", "images")
META_PATH    = os.path.join(BASE_DIR, "dataset", "metadata.csv")

# ── Caption templates per class ────────────────────────────────────────────
CAPTIONS = {
    "glioma":      "Brain MRI scan showing glioma tumor, high-grade malignant growth",
    "meningioma":  "Brain MRI showing meningioma, benign tumor on brain membrane",
    "pituitary":   "Brain MRI with pituitary tumor near the base of the skull",
    "notumor":     "Normal brain MRI scan with no tumor detected, healthy tissue",
}

TARGET_SIZE = (512, 512)


def preprocess_dataset(dry_run: bool = False, max_per_class: int = None):
    """
    Walks Training + Testing folders, resizes images to 512×512 RGB,
    saves to processed/images/, and writes metadata.csv.

    Args:
        dry_run: If True, only validate — don't write files.
        max_per_class: Limit images per class (useful for quick tests).
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    records = []
    splits  = ["Training", "Testing"]
    classes = ["glioma", "meningioma", "notumor", "pituitary"]

    for split in splits:
        for cls in classes:
            src_folder = os.path.join(RAW_DIR, split, cls)
            if not os.path.exists(src_folder):
                print(f"  [WARN] Skipping missing folder: {src_folder}")
                continue

            images = [f for f in os.listdir(src_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if max_per_class:
                images = images[:max_per_class]

            print(f"\n Processing [{split}] {cls} - {len(images)} images")
            for fname in tqdm(images, desc=f"{cls}"):
                src_path  = os.path.join(src_folder, fname)
                # Unique name to avoid collisions across splits
                out_name  = f"{split}_{cls}_{fname}"
                out_path  = os.path.join(PROCESSED_DIR, out_name)

                if not dry_run:
                    img = Image.open(src_path).convert("RGB")
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)
                    img.save(out_path, "JPEG", quality=95)

                records.append({
                    "image_name": out_name,
                    "label":      cls,
                    "split":      split.lower(),
                    "caption":    CAPTIONS[cls],
                })

    df = pd.DataFrame(records)
    if not dry_run:
        df.to_csv(META_PATH, index=False)
        print(f"\n[OK] Saved {len(df)} records -> {META_PATH}")
        print(f"[OK] Images saved -> {PROCESSED_DIR}")
    else:
        print(f"\n[OK] Dry-run complete - would process {len(df)} images")

    # Summary
    print("\nClass distribution:")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0).to_string())
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Brain MRI Dataset")
    parser.add_argument("--dry-run",       action="store_true", help="Validate without writing files")
    parser.add_argument("--max-per-class", type=int, default=None, help="Limit images per class")
    args = parser.parse_args()

    preprocess_dataset(dry_run=args.dry_run, max_per_class=args.max_per_class)
