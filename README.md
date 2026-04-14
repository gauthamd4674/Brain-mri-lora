# 🧠 Brain MRI Generator — Stable Diffusion + LoRA Fine-Tuning

> Fine-tuning Stable Diffusion v1.5 with LoRA on a Brain Tumor MRI dataset to generate synthetic, class-conditioned MRI images. Includes a 4-tab Gradio demo app.

---

## 📌 Project Overview

This project implements a complete **text-to-MRI generation pipeline** using:
- **Stable Diffusion v1.5** as the base generative model
- **LoRA (Low-Rank Adaptation)** for efficient fine-tuning on 6 GB VRAM
- **Brain Tumor MRI Dataset** (4 classes: Glioma, Meningioma, Pituitary, No Tumor)
- **Gradio** for an interactive 4-tab demo interface

---

## 🗂️ Project Structure

```
├── app.py                          # Gradio demo app (4 tabs)
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── dataset.py              # PyTorch Dataset + DataLoaders
│   │   └── preprocess.py          # Image preprocessing + metadata CSV
│   ├── model/
│   │   └── model_loader.py        # SD v1.5 loader + LoRA injection
│   ├── training/
│   │   └── train.py               # Full training loop (AMP + grad accumulation)
│   ├── inference/
│   │   └── generate.py            # Inference pipeline (DPM++ scheduler)
│   └── evaluation/
│       └── evaluate.py            # CLIP similarity scoring + qualitative grid
├── output/
│   ├── checkpoints/               # LoRA weights (.safetensors)
│   ├── plots/                     # Loss curves
│   ├── evaluation/                # CLIP scores + image grid
│   └── training_log.csv           # Per-step loss log
└── dataset/
    ├── metadata.csv               # Image paths + captions + labels
    └── processed/images/          # Preprocessed 512×512 MRI images
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install PyTorch with CUDA first (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then install remaining packages
pip install -r requirements.txt
```

### 2. Prepare Dataset
Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle and place it in the `Brain MRI Dataset/` folder, then run:
```bash
python src/data/preprocess.py
```

### 3. Train the Model
```bash
# Full training (3 epochs, ~3.5 hrs on RTX 4050)
python src/training/train.py

# Quick smoke test (10 steps)
python src/training/train.py --max-steps 10
```

### 4. Generate Images
```bash
# Generate one image per class using the best LoRA checkpoint
python src/inference/generate.py --all-classes --steps 25
```

### 5. Run Evaluation
```bash
python src/evaluation/evaluate.py
```

### 6. Launch the Demo App
```bash
python app.py
```
Then open **http://127.0.0.1:7860** in your browser.

---

## 🖥️ Demo App Tabs

| Tab | Description |
|-----|-------------|
| ✏️ **Free Prompt** | Generate an MRI from any custom text description |
| 🔬 **Disease Conditioning** | Pick a condition + severity, prompt is auto-built |
| 🎨 **Style Control** | Choose MRI sequence (T1/T2/FLAIR) + brain region |
| 📊 **Base vs LoRA** | Side-by-side comparison before/after fine-tuning |

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Stable Diffusion v1.5 |
| LoRA rank (r) | 4 |
| LoRA alpha | 8 |
| Target modules | `to_q, to_k, to_v, to_out.0` |
| Trainable params | ~797K / 860M (0.09%) |
| Batch size | 1 (effective 4 with grad accumulation) |
| Learning rate | 1e-4 (cosine decay) |
| Epochs | 3 |
| Precision | fp16 (VAE + CLIP) / fp32 (LoRA params) |
| GPU | NVIDIA RTX 4050 Laptop (6 GB VRAM) |

---

## 📊 Results

- **Training loss** converges from ~0.5 → ~0.08 over 3 epochs
- **CLIP similarity score** measured against class-specific prompts
- Generated images show consistent MRI-like appearance with class conditioning

### Sample Outputs
| Class | Generated MRI |
|-------|--------------|
| Glioma | `output/generated/glioma_*.png` |
| Meningioma | `output/generated/meningioma_*.png` |
| Pituitary | `output/generated/pituitary_*.png` |
| No Tumor | `output/generated/notumor_*.png` |

---

## 📦 Sharing the LoRA Weights

The trained LoRA weights (`lora_best.safetensors`, ~3 MB) are not included in this repo due to size limits. You can:
- Upload to [HuggingFace Hub](https://huggingface.co/) and reference here
- Share via Google Drive

---

## 🏗️ Architecture

```
Text Prompt
    │
    ▼
CLIP Text Encoder (frozen, fp16)
    │  768-dim embeddings
    ▼
UNet (fp32 base + fp32 LoRA adapters)  ◄── Your fine-tuned weights
    │  Iterative denoising (30 steps, DPM++ scheduler)
    ▼
VAE Decoder (frozen, fp16)
    │
    ▼
512×512 Brain MRI Image
```

---

## 📋 Requirements

- Python 3.10+
- NVIDIA GPU with 6+ GB VRAM (CUDA 12.x)
- ~10 GB disk space for model cache

---

## 📄 License

This project is for academic/educational use. The base model (SD v1.5) is subject to the [CreativeML Open RAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
