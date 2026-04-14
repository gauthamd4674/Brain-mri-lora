"""
Phase 2 — PyTorch Dataset class for Brain MRI LoRA training.
Returns dict: { pixel_values, input_ids, label } for each sample.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed", "images")
META_PATH     = os.path.join(BASE_DIR, "dataset", "metadata.csv")

# ── Stable Diffusion normalization ─────────────────────────────────────────
SD_MEAN = [0.5, 0.5, 0.5]
SD_STD  = [0.5, 0.5, 0.5]   # normalises to [-1, 1]


class BrainMRIDataset(Dataset):
    """
    Dataset for fine-tuning Stable Diffusion with LoRA on Brain MRI images.

    Args:
        split:         'train' or 'test'
        tokenizer:     CLIP tokenizer instance (from CLIPTokenizer.from_pretrained)
        max_length:    max token length for captions (77 for SD)
        image_size:    target size (default 512)
        augment:       apply light augmentation during training
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        max_length: int = 77,
        image_size: int = 512,
        augment: bool = True,
    ):
        self.split      = split
        self.tokenizer  = tokenizer
        self.max_length = max_length

        # Load + filter metadata
        # CSV stores 'training'/'testing'; normalize to 'train'/'test'
        df = pd.read_csv(META_PATH)
        df["split"] = df["split"].str.replace("training", "train").str.replace("testing", "test")
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(f"No images found for split='{split}' in {META_PATH}. "
                               "Run preprocess.py first.")

        # ── Transforms ────────────────────────────────────────────────────
        if augment and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize(SD_MEAN, SD_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(SD_MEAN, SD_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load image ────────────────────────────────────────────────────
        img_path = os.path.join(PROCESSED_DIR, row["image_name"])
        image    = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)   # [3, 512, 512], range [-1, 1]

        # ── Tokenize caption ──────────────────────────────────────────────
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                row["caption"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.squeeze(0)   # [77]
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids":    input_ids,
            "label":        row["label"],
            "caption":      row["caption"],
        }


def get_dataloaders(tokenizer, batch_size: int = 2, num_workers: int = 0):
    """
    Convenience function: returns (train_loader, test_loader).
    num_workers=0 is safer on Windows.
    """
    from torch.utils.data import DataLoader

    train_ds = BrainMRIDataset(split="train", tokenizer=tokenizer, augment=True)
    test_ds  = BrainMRIDataset(split="test",  tokenizer=tokenizer, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[OK] Train: {len(train_ds)} images | Test: {len(test_ds)} images")
    return train_loader, test_loader
