# 🔍 Bangla Meme Classifier

A **Streamlit** web app for classifying Bangla memes using the **IACF** model  
(BanglaBERT + ViT multimodal fusion) into 4 categories:

| Class | Description |
|-------|-------------|
| 🔞 **Sexual** | Sexually explicit or adult content |
| 🏛️ **Political** | Political commentary / propaganda |
| 😈 **Troll** | Provocation / mockery content |
| ✅ **Non-Bully** | Benign, no aggressive content |

---

## 🚀 Deploy FREE on Streamlit Cloud

### 1 · Fork / push this repo to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2 · Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub
2. Click **New app**
3. Select your repo, branch `main`, main file `app.py`
4. Click **Deploy** — it's completely free!

### 3 · Use the app

- Open the deployed URL
- **Upload your `.pt` checkpoint** in the sidebar (trained by the IACF notebook)
- Upload a Bangla meme image
- Click **Classify Meme** → see results!

---

## ⚡ Faster startup: Auto-load checkpoint from Hugging Face Hub

Instead of uploading the checkpoint every session, store it on HuggingFace:

1. Create a **free private repo** at [huggingface.co/new](https://huggingface.co/new)
2. Upload your `.pt` checkpoint file there
3. In Streamlit Cloud → **App Settings → Secrets**, add:
   ```toml
   HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
   ```
4. Edit `load_from_hub.py`:
   ```python
   REPO_ID  = "your-username/your-repo"
   FILENAME = "iacf_v7_seed42.pt"
   ```
5. In `app.py`, replace the sidebar uploader block with:
   ```python
   from load_from_hub import get_checkpoint_path
   ckpt_path = get_checkpoint_path()
   ```

---

## 🏗️ Architecture

```
Meme Image ──► ViT (google/vit-base-patch16-224)
                │
                ▼
            Cross-Attention Fusion (IACF)
                │
                ▼
Meme Text ──► BanglaBERT (csebuetnlp/banglabert)
                │
                ▼
         4-class Classifier
    Sexual | Political | Troll | Non-Bully
```

**Text extraction**: EasyOCR (Bengali + English, no API needed, runs locally)

---

## 📦 Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **RAM**: BanglaBERT + ViT needs ~3–4 GB RAM. Streamlit Cloud free tier provides 1 GB,  
> which may be tight. If you hit memory errors:
> - Use `torch.no_grad()` (already done ✅)
> - Quantize the model with `torch.quantization.quantize_dynamic`
> - Or upgrade to Streamlit Cloud Starter ($0 during beta)

---

## 📁 File structure

```
├── app.py               # Main Streamlit application
├── load_from_hub.py     # Optional: auto-download checkpoint from HF Hub
├── requirements.txt     # Python dependencies
├── .streamlit/
│   └── config.toml      # Streamlit theme & server config
├── .gitignore
└── README.md
```
