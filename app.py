"""
Phase 9 & 10 — Gradio Demo App
Brain MRI Generation with Stable Diffusion + LoRA

Features:
  Tab 1 — Free-text prompt generation
  Tab 2 — Disease conditioning (dropdown → auto-prompt)
  Tab 3 — Style control (MRI sequence + region)
  Tab 4 — Base vs LoRA comparison

Run:
    python app.py
"""

import os
import sys
import torch
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.inference.generate import load_pipeline, generate_image, CLASS_PROMPTS

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR  = os.path.join(BASE_DIR, "output", "checkpoints")
LORA_PATH = os.path.join(CKPT_DIR, "lora_best.safetensors")

# ── Load pipeline once at startup ─────────────────────────────────────────
print("\n[Init] Brain MRI Generator...")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
LORA_FILE = LORA_PATH if os.path.exists(LORA_PATH) else None

if LORA_FILE:
    print(f"[OK] LoRA weights found: {LORA_FILE}")
else:
    print("[WARN] No LoRA weights found -- using base SD only. Run training first!")

PIPE = load_pipeline(lora_path=LORA_FILE, device=DEVICE)
print("[OK] Model ready!\n")

# ── Disease conditioning options ───────────────────────────────────────────
DISEASE_MAP = {
    "Glioma":          "Brain MRI scan showing {severity} glioma tumor, malignant growth, axial view, T2-weighted MRI",
    "Meningioma":      "Brain MRI showing {severity} meningioma, benign tumor on brain membrane, contrast-enhanced, coronal view",
    "Pituitary Tumor": "Brain MRI with {severity} pituitary tumor near base of skull, sagittal view, T1-weighted MRI",
    "Normal (No Tumor)": "Normal brain MRI scan, healthy brain tissue, no pathology detected, axial view, T2-weighted",
}

SEVERITY_OPTIONS = ["mild", "moderate", "severe"]

# ── Style control options ──────────────────────────────────────────────────
MRI_SEQUENCES = {
    "T1-Weighted":   "T1-weighted MRI, short repetition time, grey matter bright",
    "T2-Weighted":   "T2-weighted MRI, long repetition time, CSF bright, white matter dark",
    "FLAIR":         "FLAIR MRI sequence, fluid-attenuated inversion recovery, lesions hyperintense",
    "High Contrast":  "high contrast enhanced brain MRI, gadolinium contrast agent",
    "Low Dose":       "low-dose brain MRI scan, reduced noise protocol",
}

BRAIN_REGIONS = {
    "Full Brain":      "full brain axial view",
    "Frontal Lobe":    "frontal lobe region, coronal view",
    "Temporal Lobe":   "temporal lobe region, axial view",
    "Cerebellum":      "cerebellum and posterior fossa, sagittal view",
    "Brain Stem":      "brain stem and basal ganglia, axial view",
}

NEG_PROMPT = "blurry, low quality, artifacts, distorted, cartoon, painting, color photo, text, watermark"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Free Prompt Generation
# ─────────────────────────────────────────────────────────────────────────────
def tab1_generate(prompt, negative_prompt, steps, guidance, seed):
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt."
    try:
        img = generate_image(PIPE, prompt, negative_prompt=negative_prompt,
                             steps=int(steps), guidance_scale=float(guidance), seed=int(seed))
        return img, f"✅ Generated with seed {seed}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Disease Conditioning
# ─────────────────────────────────────────────────────────────────────────────
def tab2_generate(disease, severity, steps, seed):
    template = DISEASE_MAP.get(disease, "Brain MRI scan")
    # Normal has no severity
    if disease == "Normal (No Tumor)":
        prompt = template.replace("{severity} ", "")
    else:
        prompt = template.format(severity=severity)

    try:
        img = generate_image(PIPE, prompt, negative_prompt=NEG_PROMPT,
                             steps=int(steps), guidance_scale=7.5, seed=int(seed))
        return img, f"🔬 Prompt used:\n{prompt}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Style Control
# ─────────────────────────────────────────────────────────────────────────────
def tab3_generate(sequence, region, extra_details, steps, seed):
    seq_desc    = MRI_SEQUENCES.get(sequence, sequence)
    region_desc = BRAIN_REGIONS.get(region, region)
    prompt = f"Brain MRI, {seq_desc}, {region_desc}"
    if extra_details.strip():
        prompt += f", {extra_details.strip()}"

    try:
        img = generate_image(PIPE, prompt, negative_prompt=NEG_PROMPT,
                             steps=int(steps), guidance_scale=7.5, seed=int(seed))
        return img, f"🎨 Prompt used:\n{prompt}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Base vs LoRA Comparison (single class)
# ─────────────────────────────────────────────────────────────────────────────
def tab4_compare(disease_class, steps, seed):
    prompt = CLASS_PROMPTS.get(disease_class.lower().replace(" ", "").replace("(notumor)", "notumor"),
                                CLASS_PROMPTS["glioma"])

    # Generate with base (no LoRA)
    base_pipe = load_pipeline(lora_path=None, device=DEVICE)
    base_img  = generate_image(base_pipe, prompt, steps=int(steps), seed=int(seed))
    del base_pipe
    torch.cuda.empty_cache()

    # Generate with LoRA
    lora_img = generate_image(PIPE, prompt, steps=int(steps), seed=int(seed))

    return base_img, lora_img, f"📝 Prompt: {prompt}"


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
:root {
    --primary:   #7C3AED;
    --secondary: #6D28D9;
    --accent:    #A78BFA;
    --bg:        #09090B;
    --surface:   #18181B;
    --border:    #27272A;
    --text:      #F4F4F5;
    --muted:     #71717A;
}

body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; }

.tab-nav button {
    background: var(--surface) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    background: var(--primary) !important;
    color: white !important;
    border-color: var(--primary) !important;
}

.generate-btn {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.5) !important;
}

label { color: var(--accent) !important; font-weight: 600 !important; }

.panel { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }

footer { display: none !important; }
"""

HEADER_HTML = """
<div style="text-align:center; padding: 20px 0 10px 0;">
  <h1 style="font-size:2.2rem; font-weight:800;
             background: linear-gradient(135deg, #A78BFA, #7C3AED, #6D28D9);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             margin-bottom: 6px;">&#x1F9E0; Brain MRI Generator</h1>
  <p style="color:#71717A; font-size:0.95rem; margin:0;">
    Stable Diffusion v1.5 fine-tuned with LoRA on Brain Tumor MRI Dataset
  </p>
</div>
"""

with gr.Blocks(css=CSS, title="Brain MRI Generator") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── Tab 1: Free Prompt ────────────────────────────────────────────
        with gr.TabItem("✏️ Free Prompt"):
            gr.Markdown("Generate a Brain MRI from any custom text description.")
            with gr.Row():
                with gr.Column(scale=1):
                    t1_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder='e.g. "Brain MRI showing glioma tumor, axial view, T2-weighted"',
                        lines=3,
                        value=CLASS_PROMPTS["glioma"],
                    )
                    t1_neg = gr.Textbox(
                        label="Negative Prompt",
                        value=NEG_PROMPT,
                        lines=2,
                    )
                    with gr.Row():
                        t1_steps    = gr.Slider(10, 60, value=30, step=5, label="Inference Steps")
                        t1_guidance = gr.Slider(3, 15, value=7.5, step=0.5, label="Guidance Scale")
                    t1_seed = gr.Number(value=42, label="Seed", precision=0)
                    t1_btn  = gr.Button("🧠 Generate MRI", elem_classes="generate-btn")

                with gr.Column(scale=1):
                    t1_out    = gr.Image(label="Generated MRI", type="pil", height=520)
                    t1_status = gr.Textbox(label="Status", interactive=False)

            t1_btn.click(tab1_generate,
                         inputs=[t1_prompt, t1_neg, t1_steps, t1_guidance, t1_seed],
                         outputs=[t1_out, t1_status])

        # ── Tab 2: Disease Conditioning ───────────────────────────────────
        with gr.TabItem("🔬 Disease Conditioning"):
            gr.Markdown("Select a brain condition — the prompt is built automatically.")
            with gr.Row():
                with gr.Column(scale=1):
                    t2_disease   = gr.Dropdown(label="Condition", choices=list(DISEASE_MAP.keys()),
                                               value="Glioma")
                    t2_severity  = gr.Radio(label="Severity", choices=SEVERITY_OPTIONS, value="moderate")
                    with gr.Row():
                        t2_steps = gr.Slider(10, 60, value=30, step=5, label="Steps")
                        t2_seed  = gr.Number(value=42, label="Seed", precision=0)
                    t2_btn = gr.Button("🔬 Generate", elem_classes="generate-btn")

                with gr.Column(scale=1):
                    t2_out    = gr.Image(label="Conditioned MRI", type="pil", height=520)
                    t2_status = gr.Textbox(label="Prompt Used", interactive=False, lines=3)

            t2_btn.click(tab2_generate,
                         inputs=[t2_disease, t2_severity, t2_steps, t2_seed],
                         outputs=[t2_out, t2_status])

        # ── Tab 3: Style Control ──────────────────────────────────────────
        with gr.TabItem("🎨 Style Control"):
            gr.Markdown("Control MRI sequence type and brain region.")
            with gr.Row():
                with gr.Column(scale=1):
                    t3_seq    = gr.Dropdown(label="MRI Sequence", choices=list(MRI_SEQUENCES.keys()),
                                            value="T2-Weighted")
                    t3_region = gr.Dropdown(label="Brain Region", choices=list(BRAIN_REGIONS.keys()),
                                            value="Full Brain")
                    t3_extra  = gr.Textbox(label="Additional Details (optional)",
                                           placeholder='e.g. "showing hyperintense lesion"', lines=2)
                    with gr.Row():
                        t3_steps = gr.Slider(10, 60, value=30, step=5, label="Steps")
                        t3_seed  = gr.Number(value=42, label="Seed", precision=0)
                    t3_btn = gr.Button("🎨 Generate Styled MRI", elem_classes="generate-btn")

                with gr.Column(scale=1):
                    t3_out    = gr.Image(label="Styled MRI", type="pil", height=520)
                    t3_status = gr.Textbox(label="Prompt Used", interactive=False, lines=3)

            t3_btn.click(tab3_generate,
                         inputs=[t3_seq, t3_region, t3_extra, t3_steps, t3_seed],
                         outputs=[t3_out, t3_status])

        # ── Tab 4: Comparison ─────────────────────────────────────────────
        with gr.TabItem("📊 Base vs LoRA"):
            gr.Markdown("Compare base Stable Diffusion output vs LoRA fine-tuned output side by side.")
            with gr.Row():
                with gr.Column(scale=1):
                    t4_class = gr.Dropdown(
                        label="MRI Class to Compare",
                        choices=["Glioma", "Meningioma", "Pituitary", "Normal (No Tumor)"],
                        value="Glioma",
                    )
                    with gr.Row():
                        t4_steps = gr.Slider(10, 50, value=25, step=5, label="Steps")
                        t4_seed  = gr.Number(value=42, label="Seed", precision=0)
                    t4_btn = gr.Button("📊 Compare", elem_classes="generate-btn")
                    t4_status = gr.Textbox(label="Prompt", interactive=False, lines=3)

            with gr.Row():
                t4_base = gr.Image(label="Base SD v1.5 (no fine-tuning)", type="pil", height=480)
                t4_lora = gr.Image(label="LoRA Fine-Tuned", type="pil", height=480)

            t4_btn.click(tab4_compare,
                         inputs=[t4_class, t4_steps, t4_seed],
                         outputs=[t4_base, t4_lora, t4_status])

    gr.HTML("""
    <div style="text-align:center; margin-top:16px; color:#52525B; font-size:0.82rem;">
      Stable Diffusion v1.5 + LoRA | Brain Tumor MRI Dataset | RTX 4050 Laptop GPU
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
