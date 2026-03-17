"""
Streamlit Dashboard
===================
Interactive UI for Neuro-CRISPR-KAN Safety Audit System.

Features:
- Input sgRNA-DNA pair for prediction
- Real-time risk score visualization
- Attention heatmap display
- Safety audit report generation
- Model comparison charts
- Dataset statistics

Run: streamlit run ui/app.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Streamlit must be imported at top
import streamlit as st

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Neuro-CRISPR-KAN | Safety Audit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# CUSTOM CSS
# ================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-high { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }
    .risk-moderate { background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%); }
    .risk-low { background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%); }
    .audit-box {
        background: #0d1117;
        color: #c9d1d9;
        font-family: 'Courier New', monospace;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 0.85rem;
        white-space: pre-wrap;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # ================================================================
    # HEADER
    # ================================================================
    st.markdown('<p class="main-header">🧬 Neuro-CRISPR-KAN</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Hybrid CNN-Transformer Architecture for '
        'Off-Target Prediction in Cystic Fibrosis</p>',
        unsafe_allow_html=True
    )

    # ================================================================
    # SIDEBAR
    # ================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")

        page = st.radio(
            "Navigation",
            ["🔬 Predict", "📊 Model Performance", "🧪 Ablation Study", "📋 Dataset Info"],
        )

        st.markdown("---")
        st.markdown("**Model Info**")
        st.markdown("- Architecture: CNN + DNABERT-2 + KAN")
        st.markdown("- Encoding: Null Tensor")
        st.markdown("- Loss: Focal + Spline L1")
        st.markdown("- Paper: IEEE ICAUC 2026")

    # ================================================================
    # PREDICTION PAGE
    # ================================================================
    if page == "🔬 Predict":
        st.header("Off-Target Risk Prediction")

        col1, col2 = st.columns(2)

        with col1:
            sgrna_input = st.text_input(
                "sgRNA Sequence (23bp)",
                value="ATCGATCGATCGATCGATCGNGG",
                max_chars=23,
                help="Enter a 23-nucleotide sgRNA sequence ending with NGG PAM"
            )

        with col2:
            dna_input = st.text_input(
                "DNA Target Sequence",
                value="ATCGATCGATCGATCGATCGNGG",
                max_chars=23,
                help="Enter the DNA target site sequence"
            )

        col3, col4, col5 = st.columns(3)

        with col3:
            has_deletion = st.checkbox("ΔF508 Deletion Present", value=False)
        with col4:
            chromatin = st.slider("Chromatin Accessibility", 0.0, 1.0, 0.5, 0.01)
        with col5:
            threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.01)

        if st.button("🚀 Predict Off-Target Risk", type="primary", use_container_width=True):
            with st.spinner("Running Neuro-CRISPR-KAN inference..."):
                # TODO: Replace with actual model inference
                # For demo, generate synthetic prediction
                risk_score = _demo_predict(sgrna_input, dna_input, has_deletion, chromatin)

                # Display results
                risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"
                risk_class = f"risk-{risk_level.lower()}"

                st.markdown("---")

                # Metric cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Risk Score", f"{risk_score:.4f}")
                with m2:
                    st.metric("Risk Level", risk_level)
                with m3:
                    mismatches = _count_mismatches(sgrna_input, dna_input)
                    st.metric("Mismatches", mismatches)
                with m4:
                    pam_ok = dna_input[-2:] == "GG" if len(dna_input) >= 2 else False
                    st.metric("PAM Status", "Intact ✅" if pam_ok else "Disrupted ❌")

                # Risk gauge
                st.progress(min(risk_score, 1.0))

                # Safety audit
                st.subheader("📝 Safety Audit Report")
                from rag.rag_llm import generate_template_audit
                audit = generate_template_audit(
                    risk_score, mismatches, 0, has_deletion, chromatin, pam_ok
                )
                st.markdown(f'<div class="audit-box">{audit}</div>', unsafe_allow_html=True)

    # ================================================================
    # MODEL PERFORMANCE PAGE
    # ================================================================
    elif page == "📊 Model Performance":
        st.header("Model Performance Comparison")

        # Table 1 from paper
        comparison_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Spearman ρ"],
            "DeepCRISPR (CNN)": [0.87, 0.84, 0.81, 0.82, 0.79],
            "CRISPR-Net (RNN)": [0.91, 0.89, 0.85, 0.87, 0.84],
            "Neuro-CRISPR-KAN": [0.94, 0.93, 0.89, 0.91, 0.88],
        }
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Bar chart
        import plotly.graph_objects as go

        metrics = comparison_data["Metric"][:4]
        fig = go.Figure()
        colors = {"DeepCRISPR (CNN)": "#FF6B6B", "CRISPR-Net (RNN)": "#4ECDC4",
                  "Neuro-CRISPR-KAN": "#1E88E5"}

        for model_name in ["DeepCRISPR (CNN)", "CRISPR-Net (RNN)", "Neuro-CRISPR-KAN"]:
            fig.add_trace(go.Bar(
                name=model_name,
                x=metrics,
                y=comparison_data[model_name][:4],
                marker_color=colors[model_name],
                text=[f"{v:.2f}" for v in comparison_data[model_name][:4]],
                textposition="auto",
            ))

        fig.update_layout(
            title="Performance Comparison on CFTR ΔF508 Dataset",
            yaxis_title="Score",
            barmode="group",
            yaxis_range=[0, 1.1],
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================================================================
    # ABLATION STUDY PAGE
    # ================================================================
    elif page == "🧪 Ablation Study":
        st.header("Ablation: Null Tensor vs Zero-Padding")

        st.info(
            "The Null Tensor encoding preserves positional indices of flanking "
            "nucleotides around the ΔF508 deletion, while zero-padding treats "
            "the deletion as featureless noise."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Null Tensor (Proposed)")
            st.code(
                "A → [1, 0, 0, 0, 0]\n"
                "T → [0, 1, 0, 0, 0]\n"
                "G → [0, 0, 1, 0, 0]\n"
                "C → [0, 0, 0, 1, 0]\n"
                "GAP → [0, 0, 0, 0, 1]  ← Explicit structural signal",
                language="text"
            )
        with col2:
            st.subheader("Zero-Padding (Baseline)")
            st.code(
                "A → [1, 0, 0, 0]\n"
                "T → [0, 1, 0, 0]\n"
                "G → [0, 0, 1, 0]\n"
                "C → [0, 0, 0, 1]\n"
                "GAP → [0, 0, 0, 0]  ← Lost as noise!",
                language="text"
            )

        st.subheader("Deletion-Specific Recall Comparison")
        st.metric(
            "Recall Improvement",
            "+~10%",
            delta="Null Tensor outperforms Zero-Padding",
            delta_color="normal"
        )

    # ================================================================
    # DATASET INFO PAGE
    # ================================================================
    elif page == "📋 Dataset Info":
        st.header("Synthetic Dataset Statistics")

        stats = {
            "Total Samples": "10,000",
            "Sequence Length": "23 bp",
            "Positive (Off-target)": "~30%",
            "Negative (Safe)": "~70%",
            "With ΔF508 Deletion": "~40%",
            "Encoding Dimensions": "5 (Null Tensor)",
            "Features": "sgRNA, DNA, mismatches, chromatin, PAM, deletion",
        }
        for k, v in stats.items():
            st.markdown(f"**{k}:** {v}")


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def _demo_predict(sgrna, dna, has_deletion, chromatin):
    """Demo prediction (replace with actual model inference)."""
    mismatches = _count_mismatches(sgrna, dna)
    pam_ok = dna[-2:] == "GG" if len(dna) >= 2 else False

    base_risk = max(0, 0.95 - 0.15 * mismatches)
    if not pam_ok:
        base_risk *= 0.1
    base_risk *= (0.5 + 0.5 * chromatin)
    if has_deletion:
        base_risk *= 0.9

    return np.clip(base_risk + np.random.normal(0, 0.03), 0, 1)


def _count_mismatches(seq1, seq2):
    """Count mismatches between two sequences."""
    min_len = min(len(seq1), len(seq2))
    return sum(1 for i in range(min_len) if seq1[i] != seq2[i])


if __name__ == "__main__":
    main()
