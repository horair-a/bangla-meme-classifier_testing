import os
import streamlit as st
from huggingface_hub import hf_hub_download


REPO_ID = "Muhammadhoraira/bangla-meme-iacf-checkpoint"
FILENAME = "iacf_v7_seed42.pt"


@st.cache_resource(show_spinner=False)
def get_checkpoint_path():
    """
    Download the IACF checkpoint from Hugging Face Hub and return
    the real local cached file path.
    """

    token = st.secrets.get("HF_TOKEN", None)

    if token is None:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it in Streamlit App Settings → Secrets."
        )

    ckpt_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=token,
        repo_type="model",
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    return ckpt_path
