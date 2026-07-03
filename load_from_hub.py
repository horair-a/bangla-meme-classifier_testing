"""
Optional helper — download checkpoint from Hugging Face Hub on startup.

Use this instead of the sidebar uploader when you want fully-automatic
deployment without manual uploading every session.

Steps:
  1. Create a FREE private HuggingFace repo at https://huggingface.co/new
  2. Upload your .pt checkpoint there
  3. Set HF_TOKEN in Streamlit Cloud secrets (Settings → Secrets):
         HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxx"
  4. Set REPO_ID and FILENAME below
  5. Import `get_checkpoint_path` in app.py and call it instead of the sidebar uploader

Usage in app.py:
    from load_from_hub import get_checkpoint_path
    ckpt_path = get_checkpoint_path()
"""

import os
import streamlit as st

REPO_ID = "Muhammadhoraira/bangla-meme-iacf-checkpoint"
FILENAME = "iacf_v7_seed42.pt"
LOCAL_PATH = "/tmp/iacf_checkpoint.pt"


@st.cache_resource(show_spinner=False)
def get_checkpoint_path() -> str:
    """Download checkpoint from HF Hub (cached). Returns local path."""
    if os.path.exists(LOCAL_PATH):
        return LOCAL_PATH

    try:
        from huggingface_hub import hf_hub_download
        token = st.secrets.get("HF_TOKEN", None)

        with st.spinner("⬇️ Downloading model checkpoint from Hugging Face…"):
            path = hf_hub_download(
                repo_id   = REPO_ID,
                filename  = FILENAME,
                token     = token,
                local_dir = "/tmp",
            )
        # Rename to expected path
        if path != LOCAL_PATH:
            os.rename(path, LOCAL_PATH)

        st.success("✅ Checkpoint downloaded!")
        return LOCAL_PATH

    except Exception as e:
        st.error(f"❌ Could not download checkpoint: {e}")
        st.stop()
