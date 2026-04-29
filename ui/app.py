"""
Neuro-CRISPR-KAN — Streamlit Dashboard
======================================
6 pages: Predict | Sample Explorer | Performance | Ablation | RAG Audit | Architecture

Run:
    streamlit run ui/app.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from ui.inference import (
    predict_single, count_mismatches, seed_mismatches, pam_intact,
    predict_both_encoders, saliency_per_position, dnabert_token_importance,
)


@st.cache_resource(show_spinner="Loading Llama 3.1 8B (4-bit)…")
def _get_llama():
    from rag.rag_llm import get_llama_auditor
    aud = get_llama_auditor()
    aud.initialize()
    return aud

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
FIGS = os.path.join(ROOT, "figures")

# ------------------------------------------------------------------ page
st.set_page_config(
    page_title="Neuro-CRISPR-KAN | Off-Target Safety Audit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ theme
st.markdown("""
<style>
:root { --accent:#6C8CFF; --accent2:#A66CFF; --bg:#0b0f1a; --card:#141a2b; --muted:#8a93a8; }
html, body, [class*="css"], .stApp { background: var(--bg) !important; color: #e6e9f2 !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b0f1a 0%, #131a2c 100%) !important; }
.hero {
  background: radial-gradient(1200px 400px at 10% 0%, rgba(108,140,255,.25), transparent 60%),
              radial-gradient(900px 300px at 90% 0%, rgba(166,108,255,.22), transparent 60%);
  border: 1px solid #1f263b; border-radius: 18px; padding: 28px 32px; margin-bottom: 22px;
}
.hero h1 { font-size: 2.2rem; margin: 0; background: linear-gradient(90deg,#6C8CFF,#A66CFF,#FF8FB2);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
.hero p { color: #aab2c8; margin: 6px 0 0; font-size: 1rem; }
.badge { display:inline-block; padding:4px 10px; margin-right:6px; border:1px solid #2a324a;
         border-radius:999px; font-size:.78rem; color:#cbd2e6; background:#121829; }
.card { background: var(--card); border:1px solid #1f263b; border-radius:14px; padding:18px; }
.kv { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px dashed #1f263b; }
.kv:last-child { border-bottom:none; }
.kv span:first-child { color:#aab2c8; }
.kv span:last-child { color:#fff; font-weight:600; }
.audit-box { background:#0a0e18; color:#cbd2e6; font-family:'JetBrains Mono','Courier New',monospace;
             padding:18px; border-radius:10px; font-size:.86rem; white-space:pre-wrap; border:1px solid #1f263b; }
.stButton>button { background: linear-gradient(90deg,#6C8CFF,#A66CFF); color:white; border:0;
                   border-radius:10px; padding:.6rem 1rem; font-weight:600; }
.stButton>button:hover { filter: brightness(1.08); }
.small { color:#8a93a8; font-size:.85rem; }
hr { border-color:#1f263b !important; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_data
def load_threshold_sweep():
    p = os.path.join(CKPT, "threshold_sweep.json")
    return json.load(open(p)) if os.path.exists(p) else None

@st.cache_data
def load_history(rel):
    p = os.path.join(CKPT, rel)
    return json.load(open(p)) if os.path.exists(p) else None

@st.cache_data
def load_predictions(rel):
    p = os.path.join(CKPT, rel)
    if not os.path.exists(p): return None
    d = np.load(p)
    return {"y": d["y"], "p": d["p"], "has_deletion": d["has_deletion"]}

@st.cache_data
def load_dataset():
    p = os.path.join(ROOT, "data/generated/crispr_dataset.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def risk_gauge(prob: float, threshold: float = 0.5):
    color = "#2ecc71" if prob < 0.4 else "#f1c40f" if prob < 0.7 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 42, "color": "#fff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8a93a8"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(46,204,113,.18)"},
                {"range": [40, 70], "color": "rgba(241,196,15,.18)"},
                {"range": [70, 100], "color": "rgba(231,76,60,.22)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 3},
                          "thickness": 0.85, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#cbd2e6"})
    return fig


# ------------------------------------------------------------------ hero
st.markdown("""
<div class="hero">
  <h1>🧬 Neuro-CRISPR-KAN</h1>
  <p>Hybrid CNN + DNABERT-2 + Kolmogorov-Arnold Network for Off-Target Risk Prediction
     in Cystic Fibrosis (CFTR ΔF508).</p>
  <div style="margin-top:14px;">
    <span class="badge">IEEE ICAUC 2026</span>
    <span class="badge">DNABERT-2 + LoRA</span>
    <span class="badge">Null Tensor Encoding</span>
    <span class="badge">B-Spline KAN Head</span>
    <span class="badge">RAG Safety Audit</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "page",
        ["🔬 Predict", "🧫 Sample Explorer", "📊 Performance",
         "🧪 Ablation", "📋 RAG Audit", "🏗️ Architecture"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown('<div class="small">CNN(5ch) ⊕ DNABERT-2+LoRA → Gated Fusion → KAN</div>',
                unsafe_allow_html=True)
    sweep = load_threshold_sweep()
    if sweep:
        st.markdown("**Test set**")
        st.markdown(f'<div class="kv"><span>AUROC</span><span>{sweep["test_auroc"]:.3f}</span></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="kv"><span>Best F1 t*</span><span>{sweep["best_threshold_f1"]:.2f}</span></div>',
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="small">Final-year project · KSRCT · 2026</div>',
                unsafe_allow_html=True)


# ================================================================== PREDICT
if page == "🔬 Predict":
    st.subheader("Off-Target Risk Prediction")
    c1, c2 = st.columns(2)
    with c1:
        sgrna = st.text_input("sgRNA (23 bp, ends in NGG)",
                              value="ATCGATCGATCGATCGATCGAGG", max_chars=23)
    with c2:
        dna   = st.text_input("DNA target (23 bp)",
                              value="ATCGATCGATCGATCGATCGAGG", max_chars=23)

    c3, c4, c5 = st.columns(3)
    with c3:
        has_del = st.checkbox("ΔF508 deletion present", value=False)
    with c4:
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    with c5:
        variant = st.selectbox("Encoder", ["null_tensor", "zero_pad"], index=0)

    tg1, tg2 = st.columns(2)
    with tg1:
        st.toggle("🦙 Use Llama 3.1 8B for the safety verdict (slower, much better)",
                  key="use_llm_predict", value=False,
                  help="Off = instant template. On = real LLM via 4-bit quant on A100 (~5 s/call).")
    with tg2:
        st.toggle("🔬 Show model explainability (encoder Δ + CNN saliency + DNABERT-2 token importance)",
                  key="show_xai", value=True,
                  help="Adds 3 panels: side-by-side encoder comparison, gradient-based per-position importance from the CNN stream, and per-token importance from DNABERT-2.")

    if st.button("🚀 Run inference", use_container_width=True):
        if len(sgrna) != 23 or len(dna) != 23:
            st.error("Both sequences must be exactly 23 bp.")
        else:
            with st.spinner("Forward pass through CNN + DNABERT-2 + KAN…"):
                res = predict_single(sgrna, dna, has_del, variant=variant)
            prob = res["risk_prob"]
            mm = count_mismatches(sgrna, dna)
            sm = seed_mismatches(sgrna, dna)
            pam = pam_intact(dna)
            decision = "OFF-TARGET" if prob > threshold else "SAFE"

            st.markdown("---")
            g1, g2 = st.columns([1.1, 1])
            with g1:
                st.plotly_chart(risk_gauge(prob, threshold), use_container_width=True)
            with g2:
                st.markdown(f"""
                <div class="card">
                  <div class="kv"><span>Decision</span><span>{'🛑 '+decision if decision=='OFF-TARGET' else '✅ '+decision}</span></div>
                  <div class="kv"><span>Risk score</span><span>{prob:.4f}</span></div>
                  <div class="kv"><span>Logit</span><span>{res['risk_logit']:+.3f}</span></div>
                  <div class="kv"><span>Threshold</span><span>{threshold:.2f}</span></div>
                  <div class="kv"><span>Total mismatches</span><span>{mm} / 23</span></div>
                  <div class="kv"><span>Seed mismatches (10–20)</span><span>{sm}</span></div>
                  <div class="kv"><span>PAM (NGG)</span><span>{'✅ intact' if pam else '❌ disrupted'}</span></div>
                  <div class="kv"><span>Encoder</span><span>{variant}</span></div>
                </div>
                """, unsafe_allow_html=True)

            # Mismatch track
            st.markdown("**Position-wise alignment**")
            track = []
            for i in range(min(len(sgrna), len(dna))):
                track.append({"pos": i+1, "sgRNA": sgrna[i], "DNA": dna[i],
                              "match": int(sgrna[i] == dna[i])})
            tdf = pd.DataFrame(track)
            fig = go.Figure(go.Bar(
                x=tdf["pos"], y=[1]*len(tdf),
                marker_color=["#2ecc71" if m else "#e74c3c" for m in tdf["match"]],
                text=[f"{s}/{d}" for s, d in zip(tdf["sgRNA"], tdf["DNA"])],
                textposition="inside", textfont={"color":"#fff", "size":11},
                hovertemplate="pos %{x}<br>sgRNA/DNA: %{text}<extra></extra>",
            ))
            fig.update_layout(height=130, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, yaxis={"visible": False},
                              xaxis={"title": "Position (1=PAM-distal · 23=PAM)",
                                     "color": "#cbd2e6"})
            st.plotly_chart(fig, use_container_width=True)

            # ---------------------------- Explainability (Tier 1 features)
            if st.session_state.get("show_xai", True):
                st.markdown("---")
                st.markdown("### 🔬 Explainability")
                tabA, tabB, tabC = st.tabs([
                    "Encoder Δ (Null Tensor vs Zero-Pad)",
                    "CNN saliency (per-position)",
                    "DNABERT-2 token importance",
                ])

                # --- Encoder comparison ---
                with tabA:
                    try:
                        with st.spinner("Running both encoders…"):
                            both = predict_both_encoders(sgrna, dna, has_del)
                        nt_p = both["null_tensor"]["risk_prob"]
                        zp_p = both["zero_pad"]["risk_prob"]
                        delta = both["delta"]
                        eA, eB, eC = st.columns([1, 1, 1])
                        with eA:
                            st.markdown("**Null Tensor (5-ch)**")
                            st.plotly_chart(risk_gauge(nt_p, threshold), use_container_width=True)
                        with eB:
                            st.markdown("**Zero-Pad (4-ch)**")
                            st.plotly_chart(risk_gauge(zp_p, threshold), use_container_width=True)
                        with eC:
                            sign = "🔺" if delta > 0 else ("🔻" if delta < 0 else "•")
                            st.markdown(f"""
                            <div class="card" style="margin-top:18px;">
                              <div class="kv"><span>Null Tensor risk</span><span>{nt_p:.4f}</span></div>
                              <div class="kv"><span>Zero-Pad risk</span><span>{zp_p:.4f}</span></div>
                              <div class="kv"><span>Δ (NT − ZP)</span><span>{sign} {delta:+.4f}</span></div>
                              <div class="kv"><span>ΔF508 in input?</span><span>{'yes' if has_del else 'no'}</span></div>
                            </div>
                            <div class="small" style="margin-top:8px;">
                              The 5-channel encoder reserves an explicit GAP channel so the network sees
                              deletions as a signal rather than absence-of-base. Δ is largest on samples
                              with ΔF508 — that's the paper's central claim, computed live.
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Encoder comparison unavailable: {e}")

                # --- CNN saliency ---
                with tabB:
                    try:
                        with st.spinner("Computing gradient saliency…"):
                            sal = saliency_per_position(sgrna, dna, has_del, variant=variant)
                        sal = sal[:23]
                        positions = list(range(1, len(sal) + 1))
                        colors = ["#e74c3c" if sgrna[i] != dna[i] else "#6C8CFF"
                                  for i in range(min(23, len(sgrna), len(dna)))]
                        sfig = go.Figure(go.Bar(
                            x=positions, y=sal, marker_color=colors,
                            text=[f"{s}/{d}" for s, d in zip(sgrna[:23], dna[:23])],
                            textposition="outside", textfont={"color": "#cbd2e6", "size": 10},
                            hovertemplate="pos %{x}<br>saliency %{y:.3f}<extra></extra>",
                        ))
                        sfig.update_layout(
                            height=300, margin=dict(l=10, r=10, t=20, b=40),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={"color": "#cbd2e6"},
                            xaxis_title="Position (1=PAM-distal · 23=PAM)",
                            yaxis_title="|∂risk / ∂input|  (normalized)",
                            yaxis_range=[0, 1.15],
                        )
                        st.plotly_chart(sfig, use_container_width=True)
                        top3 = np.argsort(-sal)[:3] + 1
                        st.markdown(
                            f'<div class="small">Top-3 most influential positions: '
                            f'<b>{top3.tolist()}</b>. Red bars mark mismatched positions; '
                            f'tall red bars = mismatches the model is reacting strongly to.</div>',
                            unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Saliency unavailable: {e}")

                # --- DNABERT-2 token importance ---
                with tabC:
                    try:
                        with st.spinner("Computing DNABERT-2 token importance…"):
                            toks, imp = dnabert_token_importance(sgrna, dna, has_del, variant=variant)
                        special = {"[CLS]", "[SEP]", "[PAD]", "[MASK]"}
                        keep = [(t, v) for t, v in zip(toks, imp) if t not in special]
                        if keep:
                            tk_labels = [t for t, _ in keep]
                            tk_vals = [float(v) for _, v in keep]
                            tfig = go.Figure(go.Bar(
                                x=tk_vals[::-1], y=tk_labels[::-1], orientation="h",
                                marker_color="#A66CFF",
                                text=[f"{v:.2f}" for v in tk_vals[::-1]],
                                textposition="outside",
                            ))
                            tfig.update_layout(
                                height=max(280, 28 * len(tk_labels)),
                                margin=dict(l=10, r=40, t=20, b=20),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font={"color": "#cbd2e6"},
                                xaxis_title="|∂CLS-projection / ∂token-embedding| (normalized)",
                                yaxis_title="DNABERT-2 BPE token",
                                xaxis_range=[0, 1.15],
                            )
                            st.plotly_chart(tfig, use_container_width=True)
                            st.markdown(
                                '<div class="small">DNABERT-2 (MosaicBERT) does not expose attention '
                                'maps, so we use gradient-based BPE token importance — the standard '
                                'Captum-style substitute. Tokens are the model\'s own subword units, '
                                'not raw bases.</div>', unsafe_allow_html=True)
                        else:
                            st.info("No non-special tokens to display.")
                    except Exception as e:
                        st.warning(f"Token importance unavailable: {e}")

            # Safety audit
            use_llm = st.session_state.get("use_llm_predict", False)
            label = "🦙 Llama-3.1-8B audit" if use_llm else "📝 Template audit"
            st.markdown(f"**{label}**")
            try:
                if use_llm:
                    aud = _get_llama()
                    with st.spinner("Llama 3.1 8B generating safety verdict…"):
                        res = aud.generate(sgrna, dna, prob, mm, sm, has_del, 0.5, pam)
                    audit = res["verdict"]
                    st.caption("Retrieved context: " + ", ".join(res["retrieved"]))
                else:
                    from rag.rag_llm import generate_template_audit
                    audit = generate_template_audit(prob, mm, sm, has_del, 0.5, pam)
            except Exception as e:
                audit = f"(audit unavailable: {e})"
            st.markdown(f'<div class="audit-box">{audit}</div>', unsafe_allow_html=True)

# ================================================================== SAMPLE EXPLORER
elif page == "🧫 Sample Explorer":
    st.subheader("Browse predictions on the held-out test set")
    preds = load_predictions("test_predictions.npz")
    df = load_dataset()
    if preds is None:
        st.warning("Run full_train.py first — test_predictions.npz not found.")
    else:
        y, p, hd = preds["y"], preds["p"], preds["has_deletion"]
        n = len(y)
        st.markdown(f"<span class='small'>{n} test samples · {int(y.sum())} off-target · "
                    f"{int(hd.sum())} with ΔF508</span>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            f_label = st.selectbox("Label", ["all", "off-target only", "safe only"])
        with col2:
            f_del = st.selectbox("Deletion", ["all", "with ΔF508", "no deletion"])
        with col3:
            sort = st.selectbox("Sort by", ["risk ↓", "risk ↑", "index"])

        idx = np.arange(n)
        if f_label == "off-target only": idx = idx[y[idx] == 1]
        elif f_label == "safe only":     idx = idx[y[idx] == 0]
        if f_del == "with ΔF508":        idx = idx[hd[idx] == 1]
        elif f_del == "no deletion":     idx = idx[hd[idx] == 0]
        if sort == "risk ↓":             idx = idx[np.argsort(-p[idx])]
        elif sort == "risk ↑":           idx = idx[np.argsort(p[idx])]

        view = pd.DataFrame({
            "idx": idx[:200],
            "label": ["off-target" if v else "safe" for v in y[idx[:200]]],
            "risk": np.round(p[idx[:200]], 4),
            "ΔF508": ["yes" if v else "no" for v in hd[idx[:200]]],
        })
        st.dataframe(view, use_container_width=True, height=440, hide_index=True)

        # Risk histogram split by label
        st.markdown("**Risk distribution by ground truth**")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=p[y == 0], name="Safe", opacity=.7,
                                   marker_color="#2ecc71", nbinsx=40))
        fig.add_trace(go.Histogram(x=p[y == 1], name="Off-target", opacity=.7,
                                   marker_color="#e74c3c", nbinsx=40))
        fig.update_layout(barmode="overlay", height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color": "#cbd2e6"},
                          xaxis_title="Predicted risk", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)


# ================================================================== PERFORMANCE
elif page == "📊 Performance":
    st.subheader("Trained model performance — held-out test set")
    sweep = load_threshold_sweep()
    if not sweep:
        st.warning("Run threshold_tune.py first.")
    else:
        d05 = sweep["test_at_default"]
        dst = sweep["test_at_best_threshold"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test AUROC", f"{sweep['test_auroc']:.3f}")
        m2.metric("Best F1 t*", f"{sweep['best_threshold_f1']:.2f}")
        m3.metric("Recall @ t=0.50", f"{d05['recall']:.3f}")
        m4.metric("Precision @ t*", f"{dst['precision']:.3f}")

        st.markdown("**Two operating points**")
        cA, cB = st.columns(2)
        with cA:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**🛟 Safety mode (t = 0.50)**")
            for k in ["accuracy","precision","recall","f1","mcc","recall_deletion","recall_no_deletion"]:
                st.markdown(f'<div class="kv"><span>{k}</span><span>{d05[k]:.3f}</span></div>',
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with cB:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**🔍 Audit mode (t = {sweep['best_threshold_f1']:.2f})**")
            for k in ["accuracy","precision","recall","f1","mcc","recall_deletion","recall_no_deletion"]:
                st.markdown(f'<div class="kv"><span>{k}</span><span>{dst[k]:.3f}</span></div>',
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Threshold sweep plot
        sw = pd.DataFrame(sweep["sweep"])
        st.markdown("**Threshold sweep on validation set**")
        fig = go.Figure()
        for col, color in [("f1", "#6C8CFF"), ("precision", "#2ecc71"),
                           ("recall", "#e74c3c"), ("mcc", "#A66CFF")]:
            fig.add_trace(go.Scatter(x=sw["threshold"], y=sw[col], name=col,
                                     mode="lines", line={"color": color, "width": 2}))
        fig.add_vline(x=sweep["best_threshold_f1"], line_dash="dash", line_color="#fff",
                      annotation_text=f"best F1 t*={sweep['best_threshold_f1']:.2f}")
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font={"color":"#cbd2e6"},
                          xaxis_title="Threshold", yaxis_title="Score",
                          legend={"orientation":"h"})
        st.plotly_chart(fig, use_container_width=True)

        # Training curves figure (use generated png if present)
        png = os.path.join(FIGS, "fig1_training_curves.png")
        if os.path.exists(png):
            st.markdown("**Training history**")
            st.image(png, use_container_width=True)


# ================================================================== ABLATION
elif page == "🧪 Ablation":
    st.subheader("Null Tensor vs. Zero-Padding (paper's central claim)")

    cA, cB = st.columns(2)
    with cA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Null Tensor (5 channels)**")
        st.code("A → [1,0,0,0,0]\nT → [0,1,0,0,0]\nG → [0,0,1,0,0]\n"
                "C → [0,0,0,1,0]\nGAP → [0,0,0,0,1]   ← explicit deletion signal",
                language="text")
        st.markdown("</div>", unsafe_allow_html=True)
    with cB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Zero-Pad (4 channels) — baseline**")
        st.code("A → [1,0,0,0]\nT → [0,1,0,0]\nG → [0,0,1,0]\n"
                "C → [0,0,0,1]\nGAP → [0,0,0,0]   ← lost as noise",
                language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    # Real ablation numbers from histories + test_predictions
    h1 = load_history("history.json")
    h2 = load_history("ablation_zeropad/history.json")
    p1 = load_predictions("test_predictions.npz")
    p2 = load_predictions("ablation_zeropad/test_predictions.npz")

    if h1 and h2 and p1 and p2:
        from sklearn.metrics import roc_auc_score, recall_score, f1_score, accuracy_score

        def m(p, t=0.5):
            y, pr, hd = p["y"], p["p"], p["has_deletion"]
            yp = (pr > t).astype(int)
            out = {
                "AUROC": roc_auc_score(y, pr),
                "Accuracy": accuracy_score(y, yp),
                "F1": f1_score(y, yp, zero_division=0),
                "Recall": recall_score(y, yp, zero_division=0),
                "Recall (ΔF508)": recall_score(y[hd==1], yp[hd==1], zero_division=0),
                "Recall (no del)": recall_score(y[hd==0], yp[hd==0], zero_division=0),
            }
            return out

        m1, m2 = m(p1), m(p2)
        st.markdown("**Test-set comparison (t = 0.50)**")
        rows = []
        for k in m1:
            rows.append({"Metric": k,
                         "Null Tensor": f"{m1[k]:.3f}",
                         "Zero-Pad":    f"{m2[k]:.3f}",
                         "Δ":           f"{m1[k]-m2[k]:+.3f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # bar comparison
        keys = list(m1.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Null Tensor", x=keys, y=[m1[k] for k in keys],
                             marker_color="#6C8CFF",
                             text=[f"{m1[k]:.2f}" for k in keys], textposition="outside"))
        fig.add_trace(go.Bar(name="Zero-Pad", x=keys, y=[m2[k] for k in keys],
                             marker_color="#A66CFF",
                             text=[f"{m2[k]:.2f}" for k in keys], textposition="outside"))
        fig.update_layout(height=420, barmode="group",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color":"#cbd2e6"}, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)
    else:
        png = os.path.join(FIGS, "fig4_deletion_stratified_recall.png")
        if os.path.exists(png):
            st.image(png, use_container_width=True)
        else:
            st.info("Train both encoders to populate this page.")


# ================================================================== RAG AUDIT
elif page == "📋 RAG Audit":
    st.subheader("Retrieval-Augmented Safety Audit")
    st.markdown('<span class="small">Combines model risk score with biomedical guidance '
                'to produce an examiner-friendly clinical note.</span>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        rs = st.slider("Predicted risk score", 0.0, 1.0, 0.78, 0.01)
        mm = st.slider("Total mismatches", 0, 23, 3)
    with c2:
        sm = st.slider("Seed-region mismatches", 0, 10, 1)
        chrom = st.slider("Chromatin accessibility", 0.0, 1.0, 0.6, 0.01)
    has_del = st.checkbox("ΔF508 deletion present", value=True)
    pam = st.checkbox("PAM (NGG) intact", value=True)

    cA, cB = st.columns(2)
    sgrna_demo = cA.text_input("sgRNA", "ATCGATCGATCGATCGATCGAGG", max_chars=23)
    dna_demo   = cB.text_input("DNA",   "ATCGATCGATCGATCGATCGAGG", max_chars=23)

    use_llm = st.toggle("🦙 Use Llama 3.1 8B (real LLM)", value=True,
                        help="On = local Llama 3.1 8B (4-bit, A100). Off = template fallback.")

    if st.button("Generate audit report", use_container_width=True):
        try:
            if use_llm:
                aud = _get_llama()
                with st.spinner("Llama 3.1 8B retrieving context + generating verdict…"):
                    res = aud.generate(sgrna_demo, dna_demo, rs, mm, sm, has_del, chrom, pam)
                report = res["verdict"]
                st.caption("📚 Retrieved KB chunks: " + ", ".join(res["retrieved"]))
            else:
                from rag.rag_llm import generate_template_audit
                report = generate_template_audit(rs, mm, sm, has_del, chrom, pam)
        except Exception as e:
            report = f"(audit unavailable: {e})"
        st.markdown(f'<div class="audit-box">{report}</div>', unsafe_allow_html=True)


# ================================================================== ARCHITECTURE
elif page == "🏗️ Architecture":
    st.subheader("Model architecture & training recipe")

    st.markdown("""
<div class="card">
<b>Inputs</b><br>
&nbsp;&nbsp;• Encoded pair tensor — shape <code>(B, 2, 23, 5)</code> via Null Tensor encoding<br>
&nbsp;&nbsp;• Raw sgRNA / DNA strings — fed to DNABERT-2 tokenizer<br><br>
<b>Two streams</b><br>
&nbsp;&nbsp;• <b>1D-CNN stream</b> — multi-kernel residual blocks (k = 3, 5, 7), 64 filters each<br>
&nbsp;&nbsp;&nbsp;&nbsp;projects to 128-d feature vector<br>
&nbsp;&nbsp;• <b>Transformer stream</b> — DNABERT-2 (117 M params, frozen) + LoRA adapter<br>
&nbsp;&nbsp;&nbsp;&nbsp;(rank 8, target = Wqkv) → 128-d projection<br><br>
<b>Gated fusion</b> — learned scalar gate σ(W·[c‖t]) blends streams into 256-d feature<br><br>
<b>KAN head</b> — Kolmogorov-Arnold Network with B-spline activations on edges<br>
&nbsp;&nbsp;hidden = [128, 64], spline order = 3, knots = 8, L1 = 0.01<br><br>
<b>Loss</b> — 0.7·Focal(γ=2) + 0.25·SplineL1<br>
<b>Optimizer</b> — Adam, multi-LR groups (LoRA 1e-5 · others 1e-4), CosineAnneal T_max=50<br>
<b>Trainable</b> — ~3.0 M of 120 M total (2.5 %)
</div>
    """, unsafe_allow_html=True)

    png = os.path.join(FIGS, "fig7_risk_distribution.png")
    if os.path.exists(png):
        st.markdown("**Calibration on test set**")
        st.image(png, use_container_width=True)
