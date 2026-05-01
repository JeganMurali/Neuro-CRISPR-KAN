"""
FastAPI backend for the Neuro-CRISPR-KAN web frontend.

Wraps the existing inference helpers (ui/inference.py) and the Llama RAG
auditor (rag/rag_llm.py). The main page (web/) calls these endpoints
instead of mocking with setTimeout.

Run:
    cd Neuro-CRISPR-KAN
    uvicorn backend.server:app --reload --port 8000

Endpoints:
    GET  /            health check
    POST /api/predict full prediction (risk, encoder Δ, saliency, tokens)
    POST /api/audit   Llama 3.1 8B clinical audit (lazy-loaded)
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make sure the repo root is on sys.path so we can import ui/, rag/, etc.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
PRED_LOG = os.path.join(LOG_DIR, "predictions.jsonl")
AUDIT_LOG = os.path.join(LOG_DIR, "audits.jsonl")
SERVER_LOG = os.path.join(LOG_DIR, "server.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(SERVER_LOG)],
)
log = logging.getLogger("neurocrispr.backend")


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append one JSON line to a log file. Best-effort — never crashes the API."""
    try:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("jsonl log write failed (%s): %s", path, e)

app = FastAPI(title="Neuro-CRISPR-KAN API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # demo only — tighten before public deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/response schemas ─────────────────────────────────
SEQ_LEN = 23
VALID_BASES = set("ATGC")

# Biological priors applied post-hoc to the model output.
# These reflect *strict* Cas9 mechanism that the model learns only softly
# from data. Documented and exposed in the response so the UI can show
# the user what was applied.
PAM_FILTER_CAP = 0.05         # broken-PAM upper bound on cleavage prob
DELETION_POS = 12             # canonical ΔF508 start position (0-indexed)
DELETION_LEN = 3


class PredictRequest(BaseModel):
    sgrna: str = Field(..., min_length=SEQ_LEN, max_length=SEQ_LEN)
    dna: str = Field(..., min_length=SEQ_LEN, max_length=SEQ_LEN)
    has_deletion: bool = False


def _validate_seq(name: str, seq: str) -> str:
    """Normalize and validate. Reject N (model wasn't trained on wildcards)."""
    s = seq.upper().strip()
    if len(s) != SEQ_LEN:
        raise HTTPException(status_code=422,
            detail=f"{name} must be exactly {SEQ_LEN} nt (got {len(s)})")
    bad = set(s) - VALID_BASES
    if bad:
        msg = (f"{name} contains invalid base{'s' if len(bad)>1 else ''} "
               f"{sorted(bad)!s}. Use only A/T/G/C — replace N with the actual base.")
        raise HTTPException(status_code=422, detail=msg)
    return s


class AuditRequest(BaseModel):
    sgrna: str
    dna: str
    risk_prob: float
    mismatches: int
    seed_mismatches: int
    has_deletion: bool = False
    pam_intact: bool = True
    chromatin_score: float = 0.5
    use_llm: bool = True   # set False to use template fallback (fast)


# ── Lazy model loading ───────────────────────────────────────
@app.on_event("startup")
def _warm_neuro_model() -> None:
    """Pre-load weights AND warm CUDA kernels with a dummy forward pass
    so the first user request is fast (not 4s)."""
    # Suppress streamlit warnings when running headless
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    try:
        from ui.inference import load_model, predict_single
        log.info("Pre-loading Neuro-CRISPR-KAN (null_tensor)…")
        load_model("null_tensor")
        log.info("Warming CUDA kernels with dummy forward pass…")
        predict_single("GAGTCCGAGCAGAAGAAGAATGG",
                       "GAGTCCGAGCAGAAGAAGAATGG",
                       has_deletion=False, variant="null_tensor")
        log.info("Model + kernels ready.")
    except Exception as e:
        log.warning("Model warm-up failed (will retry on first request): %s", e)


@app.get("/")
def health():
    return {"status": "ok", "service": "neuro-crispr-kan", "version": "1.0"}


@app.get("/api/logs")
def get_logs(kind: str = "predictions", n: int = 20):
    """Return the last N entries from a log file. kind in {predictions, audits}."""
    path = PRED_LOG if kind == "predictions" else AUDIT_LOG
    if not os.path.exists(path):
        return {"path": path, "entries": [], "count": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        recent = [json.loads(l) for l in lines[-n:] if l.strip()]
        return {"path": path, "entries": recent, "count": len(lines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"log read: {e}")


# ── /api/predict ─────────────────────────────────────────────
@app.post("/api/predict")
def predict(body: PredictRequest):
    from ui.inference import (
        predict_single,
        saliency_per_position,
        dnabert_token_importance,
        count_mismatches,
        seed_mismatches as _seed_mm,
        pam_intact as _pam,
    )

    sg = _validate_seq("sgrna", body.sgrna)
    dna = _validate_seq("dna", body.dna)

    # Deletion-aware encoding: when has_deletion=True, shrink the DNA by
    # DELETION_LEN bases at DELETION_POS so the NullTensor encoder's null-
    # vector path activates (it only triggers when len(seq) < target_length).
    # sgRNA stays 23 nt — the deletion only exists on the genomic strand.
    if body.has_deletion:
        dna_for_model = dna[:DELETION_POS] + dna[DELETION_POS + DELETION_LEN:]
    else:
        dna_for_model = dna

    t0 = time.time()
    try:
        nt = predict_single(sg, dna_for_model, body.has_deletion, variant="null_tensor")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "cuda" in str(e).lower():
            try:
                import torch; torch.cuda.empty_cache()
            except Exception: pass
            log.error("CUDA OOM on predict_single")
            raise HTTPException(status_code=503,
                detail="GPU out of memory — please retry. Free GPU memory or restart the server.")
        log.exception("predict_single failed")
        raise HTTPException(status_code=500, detail=f"predict_single: {e}")
    except Exception as e:
        log.exception("predict_single failed")
        raise HTTPException(status_code=500, detail=f"predict_single: {e}")

    # Encoder ablation — best-effort. Skip if zero_pad checkpoint is missing.
    encoder_delta = {"null_tensor": nt["risk_prob"], "zero_pad": None, "delta": None}
    try:
        zp = predict_single(sg, dna_for_model, body.has_deletion, variant="zero_pad")
        encoder_delta = {
            "null_tensor": nt["risk_prob"],
            "zero_pad": zp["risk_prob"],
            "delta": nt["risk_prob"] - zp["risk_prob"],
        }
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("zero_pad encoder unavailable: %s", e)

    # Saliency — best-effort
    saliency = []
    try:
        saliency = saliency_per_position(sg, dna_for_model, body.has_deletion).tolist()
    except Exception as e:
        log.warning("saliency failed: %s", e)

    # DNABERT-2 token importance — best-effort
    tokens, token_imp = [], []
    try:
        tokens, imp_arr = dnabert_token_importance(sg, dna_for_model, body.has_deletion)
        token_imp = imp_arr.tolist()
    except Exception as e:
        log.warning("token importance failed: %s", e)

    elapsed = time.time() - t0

    # ── PAM hard filter ──────────────────────────────────────
    # Cas9 strictly requires NGG. The model learns this only softly from
    # data — broken-PAM negatives still get MOD scores. We apply a hard
    # cap and surface both values so the UI / paper can show transparency.
    raw_prob = float(nt["risk_prob"])
    pam_ok = _pam(dna)
    if not pam_ok:
        risk_prob = min(raw_prob, PAM_FILTER_CAP)
        pam_filter_applied = True
    else:
        risk_prob = raw_prob
        pam_filter_applied = False

    response = {
        "risk_prob": risk_prob,
        "risk_prob_raw": raw_prob,
        "pam_filter_applied": pam_filter_applied,
        "risk_logit": float(nt["risk_logit"]),
        "encoder_delta": encoder_delta,
        "saliency": saliency,
        "tokens": tokens,
        "token_importance": token_imp,
        "mismatches": count_mismatches(sg, dna),
        "seed_mismatches": _seed_mm(sg, dna),
        "pam_intact": pam_ok,
        "sgRNA": sg,
        "dna": dna,
        "hasDel": body.has_deletion,
        "elapsed": round(elapsed, 2),
    }

    # Persist for post-hoc verification
    _append_jsonl(PRED_LOG, {
        "endpoint": "predict",
        "input": {"sgrna": sg, "dna": dna, "has_deletion": body.has_deletion},
        "output": {
            "risk_prob": response["risk_prob"],
            "risk_prob_raw": response["risk_prob_raw"],
            "pam_filter_applied": response["pam_filter_applied"],
            "risk_logit": response["risk_logit"],
            "encoder_delta": response["encoder_delta"],
            "mismatches": response["mismatches"],
            "seed_mismatches": response["seed_mismatches"],
            "pam_intact": response["pam_intact"],
            "saliency": response["saliency"],
            "tokens": response["tokens"],
            "token_importance": response["token_importance"],
        },
        "elapsed_s": response["elapsed"],
    })
    log.info("PREDICT  sg=%s dna=%s has_del=%s -> risk=%.4f mm=%d seed_mm=%d pam=%s in %.2fs",
             sg[:8] + "…", dna[:8] + "…", body.has_deletion,
             response["risk_prob"], response["mismatches"], response["seed_mismatches"],
             response["pam_intact"], response["elapsed"])

    return response


# ── /api/audit ───────────────────────────────────────────────
@app.post("/api/audit")
def audit(body: AuditRequest):
    """Returns a Llama 3.1 8B clinical audit (or a template fallback)."""
    t0 = time.time()

    if not body.use_llm:
        from rag.rag_llm import generate_template_audit
        text = generate_template_audit(
            risk_score=body.risk_prob,
            num_mismatches=body.mismatches,
            seed_mismatches=body.seed_mismatches,
            has_deletion=body.has_deletion,
            chromatin_score=body.chromatin_score,
            pam_intact=body.pam_intact,
        )
        resp = {
            "mode": "template",
            "verdict": text,
            "retrieved": [],
            "risk_level": "high" if body.risk_prob > 0.7 else "moderate" if body.risk_prob > 0.3 else "low",
            "elapsed": round(time.time() - t0, 2),
        }
        _append_jsonl(AUDIT_LOG, {
            "endpoint": "audit", "mode": resp["mode"],
            "input": {"sgrna": body.sgrna, "dna": body.dna, "risk_prob": body.risk_prob,
                      "mismatches": body.mismatches, "seed_mismatches": body.seed_mismatches,
                      "has_deletion": body.has_deletion, "pam_intact": body.pam_intact},
            "output": resp, "elapsed_s": resp["elapsed"],
        })
        log.info("AUDIT    mode=%s risk=%.3f -> %s in %.2fs", resp["mode"], body.risk_prob, resp["risk_level"], resp["elapsed"])
        return resp

    try:
        from rag.rag_llm import generate_llm_audit
        out = generate_llm_audit(
            sgrna=body.sgrna,
            dna=body.dna,
            risk_score=body.risk_prob,
            num_mismatches=body.mismatches,
            seed_mismatches=body.seed_mismatches,
            has_deletion=body.has_deletion,
            chromatin_score=body.chromatin_score,
            pam_intact=body.pam_intact,
        )
        out["mode"] = "llama-3.1-8b"
        out["elapsed"] = round(time.time() - t0, 2)
        _append_jsonl(AUDIT_LOG, {
            "endpoint": "audit", "mode": out["mode"],
            "input": {"sgrna": body.sgrna, "dna": body.dna, "risk_prob": body.risk_prob,
                      "mismatches": body.mismatches, "seed_mismatches": body.seed_mismatches,
                      "has_deletion": body.has_deletion, "pam_intact": body.pam_intact},
            "output": out, "elapsed_s": out["elapsed"],
        })
        log.info("AUDIT    mode=%s risk=%.3f -> %s in %.2fs",
                 out["mode"], body.risk_prob, out.get("risk_level", "?"), out["elapsed"])
        return out
    except Exception as e:
        log.exception("Llama audit failed; falling back to template")
        from rag.rag_llm import generate_template_audit
        text = generate_template_audit(
            risk_score=body.risk_prob,
            num_mismatches=body.mismatches,
            seed_mismatches=body.seed_mismatches,
            has_deletion=body.has_deletion,
            chromatin_score=body.chromatin_score,
            pam_intact=body.pam_intact,
        )
        resp = {
            "mode": "template-fallback",
            "verdict": text,
            "retrieved": [],
            "risk_level": "high" if body.risk_prob > 0.7 else "moderate" if body.risk_prob > 0.3 else "low",
            "error": str(e),
            "elapsed": round(time.time() - t0, 2),
        }
        _append_jsonl(AUDIT_LOG, {
            "endpoint": "audit", "mode": resp["mode"],
            "input": {"sgrna": body.sgrna, "dna": body.dna, "risk_prob": body.risk_prob,
                      "mismatches": body.mismatches, "seed_mismatches": body.seed_mismatches,
                      "has_deletion": body.has_deletion, "pam_intact": body.pam_intact},
            "output": resp, "elapsed_s": resp["elapsed"],
        })
        log.info("AUDIT    mode=%s (fallback) risk=%.3f in %.2fs", resp["mode"], body.risk_prob, resp["elapsed"])
        return resp
