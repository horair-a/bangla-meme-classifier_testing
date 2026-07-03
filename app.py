"""
Bangla Meme Classifier — IACF (BanglaBERT + ViT)
4 classes: Sexual | Political | Troll | Non-Bully
Deploy free on Streamlit Cloud via GitHub.
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
    page_title="Bangla Meme Classifier",
    page_icon="🔍",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.header-box h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.header-box p  { margin: 0.4rem 0 0; opacity: 0.8; font-size: 0.95rem; }

.result-card {
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
    border-left: 5px solid;
}
.pred-Sexual    { background:#fff0f3; border-color:#e63946; }
.pred-Political { background:#fff8e1; border-color:#f4a261; }
.pred-Troll     { background:#e8f5e9; border-color:#2a9d8f; }
.pred-Non-Bully { background:#e8eaf6; border-color:#457b9d; }

.label-badge {
    display:inline-block;
    padding:4px 14px;
    border-radius:20px;
    font-weight:600;
    font-size:1rem;
    color:white;
}
.badge-Sexual    { background:#e63946; }
.badge-Political { background:#f4a261; }
.badge-Troll     { background:#2a9d8f; }
.badge-Non-Bully { background:#457b9d; }

.conf-bar-bg {
    background:#e9ecef; border-radius:8px; height:10px;
    margin: 4px 0 12px;
}
.conf-bar-fill {
    height:10px; border-radius:8px;
    transition: width 0.6s ease;
}
.ocr-box {
    background:#f8f9fa; border:1px solid #dee2e6;
    border-radius:10px; padding:1rem 1.2rem;
    font-family: monospace; font-size:0.95rem;
    white-space: pre-wrap; word-break: break-word;
}
.info-chip {
    display:inline-block; background:#e9ecef;
    border-radius:20px; padding:3px 12px; font-size:0.82rem;
    margin: 2px 3px;
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
# IACF Model definition  (must match training notebook byte-for-byte)
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
            d = proj_dim; d2 = d * 2

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
                nn.Linear(d*3, d2), nn.ReLU(), nn.Dropout(dropout_attn), nn.Linear(d2, d2))
            self.drop_branch = nn.Dropout(dropout_proj)

            self.gate_linear = nn.Linear(d2*2, d2)
            nn.init.constant_(self.gate_linear.bias, gate_bias_init)

            self.drop_cls   = nn.Dropout(dropout_cls)
            self.classifier = nn.Linear(d2 + d + d, num_classes)
            self.aux_agree  = nn.Linear(d2, num_classes)
            self.aux_incon  = nn.Linear(d2, num_classes)
            self._d = d

        def forward(self, input_ids, attention_mask, token_type_ids, pixel_values):
            text_out  = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids)
            H = text_out.last_hidden_state                      # (B,128,768)

            image_out = self.image_encoder(pixel_values=pixel_values)
            V = image_out.last_hidden_state[:, 1:, :]           # (B,196,768)

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

            R_t = H_p - H_cross; R_v = V_p - V_cross
            h_r = masked_mean_pool(R_t, attention_mask)
            v_r = R_v.mean(dim=1)
            d_c = torch.abs(mu_t - mu_v)
            f_incon = self.drop_branch(self.incon_mlp(torch.cat([h_r, v_r, d_c], dim=-1)))

            g       = torch.sigmoid(self.gate_linear(torch.cat([f_agree, f_incon], dim=-1)))
            f_fused = g * f_agree + (1.0 - g) * f_incon
            f_final = self.drop_cls(torch.cat([f_fused, mu_t, mu_v], dim=-1))
            return self.classifier(f_final)

    import torch
    device = torch.device("cpu")   # CPU-only on free Streamlit Cloud

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]

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


@st.cache_resource(show_spinner=False)
def load_ocr_reader():
    """EasyOCR reader for Bengali (bn) + English (en)."""
    import easyocr
    return easyocr.Reader(["bn", "en"], gpu=False, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def ocr_image(pil_img: Image.Image) -> str:
    """Extract Bangla + English text from meme via EasyOCR."""
    reader = load_ocr_reader()
    img_np = np.array(pil_img.convert("RGB"))
    results = reader.readtext(img_np, detail=0, paragraph=True)
    return " ".join(results).strip()


def normalize_bangla(text: str) -> str:
    """BanglaBERT normalisation (same as training)."""
    try:
        from normalizer import normalize as bn_normalize
        NORM_KW = dict(unicode_norm="NFKC", punct_replacement=None,
                       url_replacement=None, emoji_replacement=None,
                       apply_unicode_norm_last=True)
        out = bn_normalize(text, **NORM_KW)
        return out if out.strip() else "[EMPTY]"
    except Exception:
        return text if text.strip() else "[EMPTY]"


def predict(pil_img: Image.Image, text: str, model, tokenizer, processor, device):
    import torch
    text_norm = normalize_bangla(text)
    enc = tokenizer(text_norm, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    pixel_values = processor(images=pil_img.convert("RGB"),
                             return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        logits = model(
            input_ids       = enc["input_ids"].to(device),
            attention_mask  = enc["attention_mask"].to(device),
            token_type_ids  = enc["token_type_ids"].to(device),
            pixel_values    = pixel_values.to(device),
        )  # (1, 4)

    logits_np = logits.cpu().numpy()[0]
    e = np.exp(logits_np - logits_np.max())
    probs = e / e.sum()
    pred_id = int(np.argmax(probs))
    return ID2LABEL[pred_id], probs


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>🔍 Bangla Meme Classifier</h1>
  <p>IACF · BanglaBERT + ViT · 4 classes: Sexual · Political · Troll · Non-Bully</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: setup ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    st.markdown("""
**Step 1 — Checkpoint**

The IACF checkpoint is loaded automatically from Hugging Face.
    """)

    st.divider()
    st.markdown("""
**Text Input**  
For stable deployment, manual text input is recommended.  
OCR is optional and may be slow on free Streamlit Cloud.
    """)
    manual_text_mode = st.checkbox("Enter text manually (skip OCR)", value=True)

    st.divider()
    st.markdown("""
**About**  
- Model: IACF v7 (BanglaBERT + ViT)  
- Checkpoint: Hugging Face Hub  
- App: Streamlit Cloud  
    """)

# ── Load checkpoint + model ──────────────────────────────────────────────────
with st.spinner("⏳ Downloading checkpoint from Hugging Face..."):
    try:
        ckpt_path = get_checkpoint_path()
        st.success("✅ Checkpoint loaded from Hugging Face.")
    except Exception as e:
        st.error(f"❌ Failed to load checkpoint from Hugging Face: {e}")
        st.stop()

# Load model (cached after first load)
with st.spinner("⏳ Loading IACF model (BanglaBERT + ViT) — this takes ~60 s on first run…"):
    try:
        model, tokenizer, device = load_model_and_tokenizer(ckpt_path)
        processor = load_vit_processor()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# ── Upload meme ─────────────────────────────────────────────────────────────
st.subheader("📤 Upload a Meme")
uploaded_img = st.file_uploader(
    "Choose a meme image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_img is None:
    st.markdown("""
<div style="border:2px dashed #ced4da; border-radius:12px; padding:3rem;
            text-align:center; color:#6c757d; margin-top:0.5rem;">
  📎 Drag and drop a meme image here, or click to browse<br>
  <small>JPG · PNG · WEBP</small>
</div>
""", unsafe_allow_html=True)
    st.stop()

pil_img = Image.open(uploaded_img).convert("RGB")

col_img, col_ocr = st.columns([1, 1], gap="medium")

with col_img:
    st.image(pil_img, caption="Uploaded meme", use_column_width=True)

with col_ocr:
    st.markdown("#### 📝 Visible Meme Text")

    if manual_text_mode:
        extracted_text = st.text_area(
            "Paste meme text here:",
            height=160,
            placeholder="মিমের দৃশ্যমান বাংলা/ইংরেজি লেখা এখানে লিখুন...",
        )

    else:
        st.info("OCR is optional. If it fails on Streamlit Cloud, use manual text mode.")

        run_ocr = st.button("🔤 Try OCR Extraction", use_container_width=True)

        if run_ocr:
            with st.spinner("Running OCR on the uploaded meme..."):
                try:
                    extracted_text = ocr_image(pil_img)

                    if extracted_text.strip():
                        st.success("OCR completed.")
                        st.markdown(
                            f'<div class="ocr-box">{extracted_text}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning("OCR did not detect any text.")
                        extracted_text = ""

                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    extracted_text = ""
        else:
            extracted_text = ""

        extracted_text = st.text_area(
            "Edit OCR text or enter text manually:",
            value=extracted_text,
            height=120,
            placeholder="OCR output will appear here, or you can type manually...",
        )

# ── Classify button ─────────────────────────────────────────────────────────
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

    # ── Results ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Classification Results")

    color   = BADGE_COLORS[pred_label]
    conf    = float(probs[LABEL2ID[pred_label]]) * 100

    st.markdown(f"""
<div class="result-card pred-{pred_label}">
  <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
    <span style="font-size:1rem; color:#495057; font-weight:600;">Prediction:</span>
    <span class="label-badge badge-{pred_label}">{pred_label}</span>
    <span style="font-size:0.9rem; color:#6c757d;">Confidence: <b>{conf:.1f}%</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Confidence Scores")
    for cls in CLASS_NAMES:
        p = float(probs[LABEL2ID[cls]]) * 100
        bar_color = BAR_COLORS[cls]
        is_pred = "⭐ " if cls == pred_label else ""
        st.markdown(f"""
<div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
  <span style="width:110px; font-size:0.9rem; font-weight:{'700' if cls==pred_label else '400'};">
    {is_pred}{cls}
  </span>
  <div class="conf-bar-bg" style="flex:1;">
    <div class="conf-bar-fill" style="width:{p:.1f}%; background:{bar_color};"></div>
  </div>
  <span style="width:52px; text-align:right; font-size:0.88rem; font-weight:600;">
    {p:.1f}%
  </span>
</div>
""", unsafe_allow_html=True)

    # Probabilities table
    st.markdown("#### Raw Probabilities")
    cols = st.columns(4)
    for i, cls in enumerate(CLASS_NAMES):
        with cols[i]:
            p = float(probs[i])
            st.metric(
                label=cls,
                value=f"{p*100:.2f}%",
                delta="← predicted" if cls == pred_label else None,
            )

    # Class descriptions
    descriptions = {
        "Sexual":    "🔞 Contains sexually explicit or adult content targeting individuals.",
        "Political": "🏛️ Contains political commentary, propaganda, or partisan messaging.",
        "Troll":     "😈 Designed to provoke, mock, or ridicule without political/sexual content.",
        "Non-Bully": "✅ Does not contain bullying, hate, or aggressive content.",
    }
    st.info(f"**{pred_label}** — {descriptions[pred_label]}")
