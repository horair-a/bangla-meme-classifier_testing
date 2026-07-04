"""
BCMD-IACF: Bengali Cyberbullying Meme Classifier
Streamlit research demo for multimodal Bengali meme classification.
Classes: Sexual | Political | Troll | Non-Bully
"""

import os
import io
import warnings
import numpy as np
import streamlit as st
from PIL import Image
from load_from_hub import get_checkpoint_path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BCMD-IACF Classifier",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}

.main-header {
    background: linear-gradient(135deg, #101828 0%, #172554 48%, #0f3460 100%);
    border-radius: 22px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.3rem;
    color: white;
    box-shadow: 0 10px 30px rgba(15, 52, 96, 0.20);
}
.main-header h1 {
    margin: 0;
    font-size: 2.05rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.main-header p {
    margin: 0.6rem 0 0;
    opacity: 0.88;
    font-size: 1rem;
    line-height: 1.55;
}

.section-card {
    background: #ffffff;
    border: 1px solid #e7eaf0;
    border-radius: 18px;
    padding: 1.25rem 1.35rem;
    margin: 0.8rem 0 1rem;
    box-shadow: 0 4px 18px rgba(16, 24, 40, 0.05);
}

.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    min-height: 105px;
}
.metric-card .label {
    color: #64748b;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.metric-card .value {
    margin-top: 0.35rem;
    color: #0f172a;
    font-size: 1.2rem;
    font-weight: 800;
}

.pipeline-step {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.75rem;
}
.pipeline-step b {
    color: #0f3460;
}

.class-card {
    border-radius: 16px;
    padding: 1.05rem 1.15rem;
    margin: 0.55rem 0;
    border: 1px solid #e5e7eb;
    background: #ffffff;
}
.class-card h4 {
    margin: 0 0 0.3rem;
}
.class-sexual { border-left: 6px solid #e63946; }
.class-political { border-left: 6px solid #f4a261; }
.class-troll { border-left: 6px solid #2a9d8f; }
.class-nonbully { border-left: 6px solid #457b9d; }

.result-card {
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0 1rem;
    border-left: 6px solid;
    box-shadow: 0 5px 22px rgba(16, 24, 40, 0.06);
}
.pred-Sexual    { background:#fff0f3; border-color:#e63946; }
.pred-Political { background:#fff8e1; border-color:#f4a261; }
.pred-Troll     { background:#e8f5e9; border-color:#2a9d8f; }
.pred-Non-Bully { background:#e8eaf6; border-color:#457b9d; }

.label-badge {
    display:inline-block;
    padding:6px 18px;
    border-radius:22px;
    font-weight:700;
    font-size:1rem;
    color:white;
}
.badge-Sexual    { background:#e63946; }
.badge-Political { background:#f4a261; }
.badge-Troll     { background:#2a9d8f; }
.badge-Non-Bully { background:#457b9d; }

.conf-row {
    display:flex;
    align-items:center;
    gap:12px;
    margin:10px 0;
}
.conf-name {
    width:120px;
    font-size:0.95rem;
}
.conf-value {
    width:58px;
    text-align:right;
    font-size:0.9rem;
    font-weight:700;
}
.conf-bar-bg {
    background:#e9ecef;
    border-radius:999px;
    height:12px;
    flex:1;
    overflow:hidden;
}
.conf-bar-fill {
    height:12px;
    border-radius:999px;
    transition: width 0.6s ease;
}

.ocr-box {
    background:#f8f9fa;
    border:1px solid #dee2e6;
    border-radius:12px;
    padding:1rem 1.15rem;
    font-family: monospace;
    font-size:0.96rem;
    white-space: pre-wrap;
    word-break: break-word;
}

.small-note {
    color: #667085;
    font-size: 0.9rem;
    line-height: 1.55;
}

hr {
    margin-top: 1.4rem;
    margin-bottom: 1.4rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Sexual", "Political", "Troll", "Non-Bully"]
LABEL2ID     = {c: i for i, c in enumerate(CLASS_NAMES)}
ID2LABEL     = {i: c for i, c in enumerate(CLASS_NAMES)}
MAX_LEN      = 128
IMG_SIZE     = 224

BADGE_COLORS = {
    "Sexual":    "#e63946",
    "Political": "#f4a261",
    "Troll":     "#2a9d8f",
    "Non-Bully": "#457b9d",
}
BAR_COLORS = {
    "Sexual":    "#e63946",
    "Political": "#f4a261",
    "Troll":     "#2a9d8f",
    "Non-Bully": "#457b9d",
}


# ─────────────────────────────────────────────────────────────────────────────
# IACF Model definition  (must match training notebook)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(ckpt_path: str):
    """Load IACF model + BanglaBERT tokenizer. Cached across reruns."""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel, ViTModel

    def masked_mean_pool(features, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    class IACF(nn.Module):
        def __init__(self, proj_dim=256, num_heads=4,
                     dropout_attn=0.1, dropout_proj=0.1, dropout_cls=0.3,
                     gate_bias_init=1.5, num_classes=4):
            super().__init__()
            d = proj_dim
            d2 = d * 2

            self.text_encoder  = AutoModel.from_pretrained("csebuetnlp/banglabert")
            self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")

            self.proj_text  = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d))
            self.proj_image = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d))
            self.drop_proj  = nn.Dropout(dropout_proj)

            self.cross_attn_t2v = nn.MultiheadAttention(
                embed_dim=d, num_heads=num_heads, dropout=dropout_attn, batch_first=True)
            self.cross_attn_v2t = nn.MultiheadAttention(
                embed_dim=d, num_heads=num_heads, dropout=dropout_attn, batch_first=True)

            self.incon_mlp = nn.Sequential(
                nn.Linear(d * 3, d2),
                nn.ReLU(),
                nn.Dropout(dropout_attn),
                nn.Linear(d2, d2),
            )
            self.drop_branch = nn.Dropout(dropout_proj)

            self.gate_linear = nn.Linear(d2 * 2, d2)
            nn.init.constant_(self.gate_linear.bias, gate_bias_init)

            self.drop_cls   = nn.Dropout(dropout_cls)
            self.classifier = nn.Linear(d2 + d + d, num_classes)
            self.aux_agree  = nn.Linear(d2, num_classes)
            self.aux_incon  = nn.Linear(d2, num_classes)
            self._d = d

        def forward(self, input_ids, attention_mask, token_type_ids, pixel_values):
            text_out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            H = text_out.last_hidden_state

            image_out = self.image_encoder(pixel_values=pixel_values)
            V = image_out.last_hidden_state[:, 1:, :]

            H_p = self.drop_proj(self.proj_text(H))
            V_p = self.drop_proj(self.proj_image(V))

            mu_t = masked_mean_pool(H_p, attention_mask)
            mu_v = V_p.mean(dim=1)

            text_kpm = (attention_mask == 0)
            H_cross, _ = self.cross_attn_t2v(
                query=H_p, key=V_p, value=V_p, need_weights=False)
            V_cross, _ = self.cross_attn_v2t(
                query=V_p, key=H_p, value=H_p,
                key_padding_mask=text_kpm, need_weights=False)

            h_agree = masked_mean_pool(H_cross, attention_mask)
            v_agree = V_cross.mean(dim=1)
            f_agree = self.drop_branch(torch.cat([h_agree, v_agree], dim=-1))

            R_t = H_p - H_cross
            R_v = V_p - V_cross
            h_r = masked_mean_pool(R_t, attention_mask)
            v_r = R_v.mean(dim=1)
            d_c = torch.abs(mu_t - mu_v)
            f_incon = self.drop_branch(self.incon_mlp(torch.cat([h_r, v_r, d_c], dim=-1)))

            g = torch.sigmoid(self.gate_linear(torch.cat([f_agree, f_incon], dim=-1)))
            f_fused = g * f_agree + (1.0 - g) * f_incon
            f_final = self.drop_cls(torch.cat([f_fused, mu_t, mu_v], dim=-1))
            return self.classifier(f_final)

    import torch

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    model = IACF(
        proj_dim       = cfg["proj_dim"],
        num_heads      = cfg["num_heads"],
        dropout_attn   = cfg["dropout_attn"],
        dropout_proj   = cfg["dropout_proj"],
        dropout_cls    = cfg["dropout_cls"],
        gate_bias_init = cfg["gate_bias_init"],
        num_classes    = cfg["num_classes"],
    ).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
    return model, tokenizer, device


@st.cache_resource(show_spinner=False)
def load_vit_processor():
    from transformers import ViTImageProcessor
    return ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


# ─────────────────────────────────────────────────────────────────────────────
# OCR helper: Tesseract
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def ocr_image_from_bytes(image_bytes: bytes) -> str:
    """Extract Bangla + English text using Tesseract OCR.

    Requires packages.txt with:
    tesseract-ocr
    tesseract-ocr-ben
    tesseract-ocr-eng
    """
    import pytesseract
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Light preprocessing for meme text
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = gray.filter(ImageFilter.SHARPEN)

    # Upscale small images for OCR
    w, h = gray.size
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        gray = gray.resize((int(w * scale), int(h * scale)))

    config1 = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(gray, lang="ben+eng", config=config1).strip()

    if not text:
        config2 = "--oem 3 --psm 11"
        text = pytesseract.image_to_string(gray, lang="ben+eng", config=config2).strip()

    return " ".join(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize_bangla(text: str) -> str:
    """BanglaBERT normalisation (same as training)."""
    try:
        from normalizer import normalize as bn_normalize
        norm_kw = dict(
            unicode_norm="NFKC",
            punct_replacement=None,
            url_replacement=None,
            emoji_replacement=None,
            apply_unicode_norm_last=True,
        )
        out = bn_normalize(text, **norm_kw)
        return out if out.strip() else "[EMPTY]"
    except Exception:
        return text if text.strip() else "[EMPTY]"


def predict(pil_img: Image.Image, text: str, model, tokenizer, processor, device):
    import torch

    text_norm = normalize_bangla(text)
    enc = tokenizer(
        text_norm,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    pixel_values = processor(
        images=pil_img.convert("RGB"),
        return_tensors="pt",
    )["pixel_values"]

    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(enc["input_ids"])

    with torch.no_grad():
        logits = model(
            input_ids       = enc["input_ids"].to(device),
            attention_mask  = enc["attention_mask"].to(device),
            token_type_ids  = token_type_ids.to(device),
            pixel_values    = pixel_values.to(device),
        )

    logits_np = logits.cpu().numpy()[0]
    e = np.exp(logits_np - logits_np.max())
    probs = e / e.sum()
    pred_id = int(np.argmax(probs))
    return ID2LABEL[pred_id], probs


# ─────────────────────────────────────────────────────────────────────────────
# Small UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def header(title: str, subtitle: str):
    st.markdown(f"""
<div class="main-header">
  <h1>{title}</h1>
  <p>{subtitle}</p>
</div>
""", unsafe_allow_html=True)


def metric_card(label: str, value: str):
    st.markdown(f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
</div>
""", unsafe_allow_html=True)


def confidence_bars(probs, pred_label):
    for cls in CLASS_NAMES:
        p = float(probs[LABEL2ID[cls]]) * 100
        bar_color = BAR_COLORS[cls]
        is_pred = "⭐ " if cls == pred_label else ""
        weight = "800" if cls == pred_label else "500"
        st.markdown(f"""
<div class="conf-row">
  <span class="conf-name" style="font-weight:{weight};">{is_pred}{cls}</span>
  <div class="conf-bar-bg">
    <div class="conf-bar-fill" style="width:{p:.1f}%; background:{bar_color};"></div>
  </div>
  <span class="conf-value">{p:.1f}%</span>
</div>
""", unsafe_allow_html=True)


def interpretation_text(pred_label: str) -> str:
    descriptions = {
        "Sexual": (
            "The meme is classified as sexual cyberbullying content based on the "
            "combined visual and textual cues."
        ),
        "Political": (
            "The meme is classified as political cyberbullying content based on "
            "political references or targeted political messaging in the image/text."
        ),
        "Troll": (
            "The meme is classified as troll content, indicating mocking, sarcastic, "
            "provocative, or ridiculing intent."
        ),
        "Non-Bully": (
            "The meme is classified as non-bullying content, indicating no strong "
            "cyberbullying signal from the combined modalities."
        ),
    }
    return descriptions[pred_label]


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    header(
        "BCMD-IACF: Bengali Cyberbullying Meme Classifier",
        "A multimodal Streamlit prototype for classifying Bengali memes using BanglaBERT, ViT, and the proposed IACF model.",
    )

    st.markdown("""
<div class="section-card">
BCMD-IACF is a research demonstration system for multiclass multimodal cyberbullying
classification in Bengali memes. The system analyzes both the uploaded meme image
and the OCR-extracted visible text, then predicts one of four categories:
<b>Sexual</b>, <b>Political</b>, <b>Troll</b>, or <b>Non-Bully</b>.
</div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Text Encoder", "BanglaBERT")
    with c2:
        metric_card("Image Encoder", "ViT")
    with c3:
        metric_card("Task Type", "Multiclass")
    with c4:
        metric_card("Classes", "4")

    st.markdown("### System Pipeline")
    st.markdown("""
<div class="section-card">
<b>Meme Image</b> → <b>OCR Text Extraction</b> → <b>BanglaBERT + ViT Encoding</b>
→ <b>IACF Multimodal Fusion</b> → <b>Class Prediction</b>
</div>
""", unsafe_allow_html=True)

    st.info("Use the **Classifier** page from the sidebar to upload a meme and run prediction.")


def page_classifier():
    header(
        "Classifier",
        "Upload a Bengali meme, extract visible text using OCR, and classify it into one of four cyberbullying categories.",
    )

    # Load checkpoint + model only on classifier page
    with st.spinner("⏳ Downloading checkpoint from Hugging Face..."):
        try:
            ckpt_path = get_checkpoint_path()
        except Exception as e:
            st.error(f"❌ Failed to load checkpoint from Hugging Face: {e}")
            st.stop()

    with st.spinner("⏳ Loading IACF model (BanglaBERT + ViT) — first run may take around one minute…"):
        try:
            model, tokenizer, device = load_model_and_tokenizer(ckpt_path)
            processor = load_vit_processor()
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()

    st.subheader("📤 Upload Bengali Meme")
    uploaded_img = st.file_uploader(
        "Choose a meme image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_img is None:
        st.markdown("""
<div style="border:2px dashed #ced4da; border-radius:16px; padding:3rem;
            text-align:center; color:#6c757d; margin-top:0.5rem;">
  📎 Drag and drop a meme image here, or click to browse<br>
  <small>JPG · PNG · WEBP</small>
</div>
""", unsafe_allow_html=True)
        st.stop()

    image_bytes = uploaded_img.getvalue()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    col_img, col_ocr = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### 🖼️ Uploaded Meme")
        st.image(pil_img, caption="Uploaded meme", use_column_width=True)

    with col_ocr:
        st.markdown("#### 📝 OCR Extracted Visible Text")

        with st.spinner("🔤 Running OCR with Tesseract…"):
            try:
                extracted_text = ocr_image_from_bytes(image_bytes)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                st.stop()

        if extracted_text:
            st.markdown(
                f'<div class="ocr-box">{extracted_text}</div>',
                unsafe_allow_html=True,
            )
            extracted_text = st.text_area(
                "Edit OCR output if needed:",
                value=extracted_text,
                height=110,
            )
        else:
            st.warning("OCR did not detect text. Try a clearer/larger meme image.")
            extracted_text = st.text_area(
                "Enter visible meme text manually if OCR fails:",
                value="",
                height=110,
            )

    st.divider()
    run_btn = st.button("🚀 Classify Meme", type="primary", use_container_width=True)

    if run_btn:
        if not extracted_text.strip():
            st.warning("⚠️ No text found. The model will run with empty text input.")

        with st.spinner("🔮 Running IACF model…"):
            try:
                pred_label, probs = predict(pil_img, extracted_text, model, tokenizer, processor, device)
            except Exception as e:
                st.error(f"❌ Inference error: {e}")
                st.stop()

        st.markdown("---")
        st.subheader("📊 Classification Results")

        conf = float(probs[LABEL2ID[pred_label]]) * 100

        st.markdown(f"""
<div class="result-card pred-{pred_label}">
  <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
    <span style="font-size:1rem; color:#475467; font-weight:700;">Prediction:</span>
    <span class="label-badge badge-{pred_label}">{pred_label}</span>
    <span style="font-size:0.95rem; color:#667085;">Confidence: <b>{conf:.1f}%</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### Confidence Scores")
        confidence_bars(probs, pred_label)

        st.info(f"**Interpretation:** {interpretation_text(pred_label)}")


def page_model_overview():
    header(
        "Model Overview",
        "Architecture and component-level overview of the proposed IACF model.",
    )

    st.markdown("""
<div class="section-card">
The proposed <b>IACF</b> model performs multimodal cyberbullying classification
by jointly analyzing the meme image and the visible text extracted from the meme.
BanglaBERT encodes the textual modality, while ViT encodes the visual modality.
Cross-modal attention captures image-text interaction, and the IACF module combines
agreement and incongruity information before final classification.
</div>
""", unsafe_allow_html=True)

    img_path = "assets/abu_ho.jpeg"
    if os.path.exists(img_path):
        st.image(
            img_path,
            caption="Architecture of the proposed IACF model",
            use_column_width=True,
        )
    else:
        st.warning("Architecture image not found. Please upload it as assets/abu_ho.jpeg.")

    st.markdown("### Main Components")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class="section-card">
<b>Text Stream</b><br>
OCR-extracted visible meme text is tokenized and encoded using BanglaBERT.
</div>
""", unsafe_allow_html=True)
        st.markdown("""
<div class="section-card">
<b>Agreement Branch</b><br>
This branch captures aligned and mutually supportive information between image and text.
</div>
""", unsafe_allow_html=True)
        st.markdown("""
<div class="section-card">
<b>Classifier</b><br>
The fused multimodal representation is mapped into four output classes.
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="section-card">
<b>Image Stream</b><br>
The uploaded meme image is encoded using a Vision Transformer to obtain visual patch features.
</div>
""", unsafe_allow_html=True)
        st.markdown("""
<div class="section-card">
<b>Incongruity Branch</b><br>
This branch captures residual or mismatched information where one modality does not fully explain the other.
</div>
""", unsafe_allow_html=True)
        st.markdown("""
<div class="section-card">
<b>Adaptive Fusion</b><br>
Agreement and incongruity features are combined before final prediction.
</div>
""", unsafe_allow_html=True)


def page_dataset_classes():
    header(
        "Dataset / Classes",
        "Overview of the Bengali Cyberbullying Meme Dataset and class definitions.",
    )

    st.markdown("""
<div class="section-card">
The <b>Bengali Cyberbullying Meme Dataset (BCMD)</b> was developed from publicly
accessible Bengali meme content collected from Facebook and Instagram between
November 2025 and April 2026. From 4,348 initially collected raw memes, the final
dataset contains <b>4,315 multimodal meme samples</b> after filtering, cleaning,
duplicate checking, annotation, and validation.
<br><br>
Each sample includes a meme image and its associated visible text. Since Bengali
meme text often appears with stylized fonts, slang, spelling variation, compression
artifacts, and code-mixed writing, the textual content was manually extracted to
reduce OCR-related errors. The final dataset is annotated into four classes:
<b>Sexual</b>, <b>Political</b>, <b>Troll</b>, and <b>Non-Bully</b>.
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Dataset Size", "4,315")
    with c2:
        metric_card("Task", "Single-label")
    with c3:
        metric_card("Modalities", "Image + Text")

    st.markdown("### Class Definitions")

    st.markdown("""
<div class="class-card class-sexual">
<h4>🔞 Sexual</h4>
Sexual or adult-oriented cyberbullying content.
</div>
<div class="class-card class-political">
<h4>🏛️ Political</h4>
Political attack, propaganda, satire, or politically targeted bullying.
</div>
<div class="class-card class-troll">
<h4>😈 Troll</h4>
Mocking, sarcastic, provocative, or ridiculing meme content.
</div>
<div class="class-card class-nonbully">
<h4>✅ Non-Bully</h4>
Benign meme content without clear bullying or harmful intent.
</div>
""", unsafe_allow_html=True)


def page_how_it_works():
    header(
        "How It Works",
        "Step-by-step processing pipeline of the deployed classifier.",
    )

    steps = [
        ("1. Upload Meme", "A Bengali meme image is uploaded through the classifier interface."),
        ("2. OCR Text Extraction", "Tesseract OCR extracts visible Bangla/English text from the meme image."),
        ("3. Text Review", "The extracted text is displayed and can be edited before classification."),
        ("4. Text Encoding", "BanglaBERT converts the visible meme text into contextual textual features."),
        ("5. Image Encoding", "ViT converts the meme image into visual patch representations."),
        ("6. Multimodal Fusion", "The IACF model combines image and text features using agreement and incongruity information."),
        ("7. Classification", "The fused representation is classified into Sexual, Political, Troll, or Non-Bully."),
        ("8. Confidence Display", "The predicted label and confidence scores for all four classes are shown."),
    ]

    for title, desc in steps:
        st.markdown(f"""
<div class="pipeline-step">
<b>{title}</b><br>
<span class="small-note">{desc}</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 BCMD-IACF")
    st.caption("Bengali Cyberbullying Meme Classifier")

    page = st.radio(
        "Navigation",
        ["Home", "Classifier", "Model Overview", "Dataset / Classes", "How It Works"],
    )

    st.divider()
    st.markdown("""
**System Info**
- Model: IACF v7
- Text Encoder: BanglaBERT
- Image Encoder: ViT
- OCR: Tesseract Bengali + English
- Checkpoint: Hugging Face Hub
    """)


if page == "Home":
    page_home()
elif page == "Classifier":
    page_classifier()
elif page == "Model Overview":
    page_model_overview()
elif page == "Dataset / Classes":
    page_dataset_classes()
elif page == "How It Works":
    page_how_it_works()
