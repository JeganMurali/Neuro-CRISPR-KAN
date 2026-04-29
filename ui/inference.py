"""
Cached model loader + single-pair prediction for the Streamlit UI.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import streamlit as st

from configs.config import config
from utils.helpers import get_device
from data.encoding import NullTensorEncoder, ZeroPadEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


@st.cache_resource(show_spinner="Loading Neuro-CRISPR-KAN (DNABERT-2 + LoRA + KAN)…")
def load_model(variant: str = "null_tensor"):
    """variant in {'null_tensor','zero_pad'}"""
    device = get_device()
    if variant == "zero_pad":
        config.cnn.input_channels = 4
        ckpt_path = os.path.join(CKPT_DIR, "ablation_zeropad", "best_model.pt")
    else:
        config.cnn.input_channels = 5
        ckpt_path = os.path.join(CKPT_DIR, "best_model.pt")

    from models.neuro_crispr_kan import NeuroCRISPRKAN
    model = NeuroCRISPRKAN(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


def predict_single(sgrna: str, dna: str, has_deletion: bool, variant: str = "null_tensor"):
    """Return dict with risk_prob and risk_logit for a single pair."""
    model, device = load_model(variant)
    sgrna = sgrna.upper().strip()
    dna = dna.upper().strip()

    if variant == "zero_pad":
        enc = ZeroPadEncoder(target_length=config.data.sequence_length)
    else:
        enc = NullTensorEncoder(target_length=config.data.sequence_length)

    pair = enc.encode_pair(sgrna, dna, has_deletion=has_deletion)  # (2, L, C)
    x = torch.from_numpy(pair).unsqueeze(0).to(device)             # (1, 2, L, C)

    with torch.no_grad():
        out = model(x, [sgrna], [dna])
    prob = float(torch.sigmoid(out["risk_logit"]).cpu().item())
    return {"risk_prob": prob, "risk_logit": float(out["risk_logit"].cpu().item())}


def count_mismatches(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i])


def seed_mismatches(a: str, b: str, seed_start: int = 10, seed_end: int = 20) -> int:
    """Mismatches in seed region (PAM-proximal)."""
    n = min(len(a), len(b), seed_end)
    return sum(1 for i in range(seed_start, n) if a[i] != b[i])


def pam_intact(dna: str) -> bool:
    return len(dna) >= 3 and dna[-2:].upper() == "GG"


def predict_both_encoders(sgrna: str, dna: str, has_deletion: bool):
    """Run inference with BOTH null_tensor and zero_pad — proves paper's claim live."""
    nt = predict_single(sgrna, dna, has_deletion, variant="null_tensor")
    zp = predict_single(sgrna, dna, has_deletion, variant="zero_pad")
    return {
        "null_tensor": nt,
        "zero_pad": zp,
        "delta": nt["risk_prob"] - zp["risk_prob"],
    }


def saliency_per_position(sgrna: str, dna: str, has_deletion: bool,
                          variant: str = "null_tensor"):
    """
    Gradient-based saliency: |∂(risk_logit)/∂(encoded_input)| summed over channels,
    averaged across the 2 strands. Returns array of shape (seq_len,) — one
    importance value per position.
    """
    model, device = load_model(variant)
    sgrna = sgrna.upper().strip()
    dna = dna.upper().strip()
    if variant == "zero_pad":
        enc = ZeroPadEncoder(target_length=config.data.sequence_length)
    else:
        enc = NullTensorEncoder(target_length=config.data.sequence_length)

    pair = enc.encode_pair(sgrna, dna, has_deletion=has_deletion)
    x = torch.from_numpy(pair).unsqueeze(0).to(device).requires_grad_(True)

    out = model(x, [sgrna], [dna])
    logit = out["risk_logit"].sum()
    grad = torch.autograd.grad(logit, x, create_graph=False)[0]   # (1, 2, L, C)
    sal = grad.detach().abs().sum(dim=-1).mean(dim=1).squeeze(0)  # (L,)
    sal = sal.cpu().numpy()
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal


def dnabert_token_importance(sgrna: str, dna: str, has_deletion: bool,
                             variant: str = "null_tensor"):
    """
    Gradient-based BPE token importance from DNABERT-2.
    Derivative of the projected CLS feature w.r.t. token embeddings → per-token importance.
    (DNABERT-2's MosaicBERT does not expose attention weights; this is the
     standard substitute used in Captum-style explainability work.)

    Returns (tokens: List[str], importance: np.ndarray of shape (T,)).
    """
    model, device = load_model(variant)
    ts = model.transformer_stream
    inputs = ts.tokenize_sequences([sgrna], [dna], device)

    # MosaicBERT's embeddings forward requires token_type_ids and the encoder
    # expects an unpadded layout — re-running them by hand is brittle. Instead,
    # register a hook on word_embeddings to capture & retain the embedding
    # tensor's gradient during a normal forward pass.
    base = ts.base_model
    inner = base.base_model if hasattr(base, "base_model") else base
    while hasattr(inner, "bert"):
        inner = inner.bert
    word_emb = inner.embeddings.word_embeddings

    captured = {}
    def _hook(_module, _inp, out):
        new_out = out.detach().clone().requires_grad_(True)
        captured["emb"] = new_out
        return new_out
    handle = word_emb.register_forward_hook(_hook)
    try:
        outputs = base(**inputs)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        cls = hidden[:, 0]
        proj = ts.projection(cls)
        score = proj.norm()
        emb = captured["emb"]
        grad = torch.autograd.grad(score, emb, create_graph=False)[0]
    finally:
        handle.remove()
    importance = grad.detach().abs().sum(dim=-1).squeeze(0).cpu().numpy()    # (T,)
    ids = inputs["input_ids"][0].cpu().tolist()
    mask = inputs["attention_mask"][0].cpu().tolist()
    keep = [i for i, m in enumerate(mask) if m == 1]
    tokens = [ts.tokenizer.decode([ids[i]]).strip() or "·" for i in keep]
    importance = importance[keep]
    if importance.max() > 0:
        importance = importance / importance.max()
    return tokens, importance
