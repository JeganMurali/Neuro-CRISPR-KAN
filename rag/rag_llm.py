"""
RAG + LLM Safety Audit Module
==============================
Retrieval-Augmented Generation for interpretable safety summaries.

From the paper:
    "The RAG module queries external databases for relevant sequence-level
     and annotation context... The language model then takes over as a
     translator, integrating this technical metadata with the raw score
     of risk to produce a clear, formatted safety summary."

Pipeline:
    1. Build a knowledge base of genomic annotations (ChromaDB vector store)
    2. When a high-risk prediction is made, retrieve relevant context
    3. Feed context + risk score + sequence info to LLM
    4. Generate human-readable safety audit summary

This module works with small models on Colab T4:
    - Embeddings: sentence-transformers/all-MiniLM-L6-v2
    - LLM: google/flan-t5-base (or groq API for larger models)
"""

import os
import json
from typing import List, Dict, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config


# ================================================================
# KNOWLEDGE BASE: Genomic annotations for RAG retrieval
# ================================================================

GENOMIC_KNOWLEDGE_BASE = [
    {
        "id": "cftr_delta_f508",
        "text": "The CFTR ΔF508 mutation is a 3-nucleotide deletion (CTT) at position 1521-1523 of the CFTR gene on chromosome 7. This deletion removes phenylalanine at position 508 of the CFTR protein, causing misfolding and degradation. It accounts for approximately 70% of Cystic Fibrosis alleles worldwide.",
        "category": "mutation",
    },
    {
        "id": "pam_site_ngg",
        "text": "The PAM (Protospacer Adjacent Motif) sequence NGG is required for SpCas9 recognition and cleavage. Disruption of the PAM site at the off-target locus significantly reduces cleavage probability. PAM-proximal mismatches (within the seed region, positions 1-12) are more deleterious to binding than PAM-distal mismatches.",
        "category": "mechanism",
    },
    {
        "id": "seed_region",
        "text": "The seed region of the sgRNA (positions 1-12 from the PAM) is critical for target recognition. Mismatches in this region drastically reduce Cas9 binding affinity and cleavage efficiency. Even a single mismatch in positions 1-5 can reduce activity by over 90%.",
        "category": "mechanism",
    },
    {
        "id": "chromatin_accessibility",
        "text": "Chromatin accessibility significantly influences CRISPR-Cas9 activity. Open chromatin regions (high ATAC-seq signal) are more accessible to Cas9, increasing both on-target and off-target cleavage rates. Heterochromatin regions show reduced off-target activity due to nucleosome occlusion.",
        "category": "epigenetics",
    },
    {
        "id": "off_target_mechanisms",
        "text": "Off-target cleavage occurs when Cas9 binds and cuts genomic loci with partial complementarity to the sgRNA. Key factors include: number and position of mismatches, chromatin state, DNA bulges, and RNA-DNA hybridization thermodynamics. Off-target events can cause insertions, deletions, or chromosomal rearrangements.",
        "category": "safety",
    },
    {
        "id": "therapeutic_implications",
        "text": "For therapeutic CRISPR applications in Cystic Fibrosis, off-target mutations in tumor suppressor genes (e.g., TP53, BRCA1) or oncogenes pose the greatest safety concern. Comprehensive off-target profiling using methods like GUIDE-seq, DISCOVER-seq, or CIRCLE-seq is recommended before clinical application.",
        "category": "clinical",
    },
    {
        "id": "deletion_encoding_challenge",
        "text": "Structural deletions like ΔF508 create challenges for computational models because the missing nucleotides disrupt the positional alignment expected by convolutional neural networks. Zero-padding introduces artifacts where the model confuses the deletion signal with background noise, leading to increased false-negative rates for deletion-specific off-targets.",
        "category": "computational",
    },
    {
        "id": "kan_spline_advantage",
        "text": "Kolmogorov-Arnold Networks use learnable B-spline activation functions on edges instead of fixed activations on nodes. This provides local plasticity to model sharp nonlinear decision boundaries, which is critical for distinguishing rare off-target events from safe background noise. The adaptive splines reduce spectral bias present in standard MLPs.",
        "category": "architecture",
    },
]


class RAGModule:
    """
    Retrieval-Augmented Generation for safety audit summaries.

    Uses a simple in-memory vector store (can upgrade to ChromaDB).
    """

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = config.rag

        self.cfg = cfg
        self.knowledge_base = GENOMIC_KNOWLEDGE_BASE
        self.embeddings = None
        self.llm = None
        self.llm_tokenizer = None
        self._initialized = False

    def initialize(self):
        """
        Load embedding model and LLM. Call once before using.
        This is separate from __init__ to avoid loading models unnecessarily.
        """
        if self._initialized:
            return

        print("Initializing RAG module...")

        # Load sentence transformer for retrieval
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(self.cfg.embedding_model)

        # Pre-compute embeddings for knowledge base
        texts = [doc["text"] for doc in self.knowledge_base]
        self.kb_embeddings = self.embed_model.encode(texts, convert_to_tensor=True)

        # Load LLM for generation
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.cfg.llm_model)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.llm_model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = self.llm.to(device)
        self.llm.eval()

        self._initialized = True
        print("RAG module initialized.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant documents from knowledge base.

        Args:
            query: Search query (e.g., description of the prediction context)
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        import torch

        query_embedding = self.embed_model.encode(query, convert_to_tensor=True)

        # Cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.kb_embeddings,
            dim=1
        )

        # Get top-k
        top_indices = similarities.argsort(descending=True)[:top_k]
        results = [self.knowledge_base[i] for i in top_indices.cpu().numpy()]

        return results

    def generate_safety_audit(
        self,
        sgrna_seq: str,
        dna_seq: str,
        risk_score: float,
        num_mismatches: int,
        seed_mismatches: int = 0,
        has_deletion: bool = False,
        chromatin_score: float = 0.5,
        pam_intact: bool = True,
    ) -> str:
        """
        Generate a human-readable safety audit summary.

        Args:
            sgrna_seq: sgRNA sequence
            dna_seq: DNA target sequence
            risk_score: Predicted off-target risk probability [0, 1]
            num_mismatches: Number of mismatches detected
            seed_mismatches: Number of seed region mismatches
            has_deletion: Whether ΔF508 deletion is present
            chromatin_score: Chromatin accessibility [0, 1]
            pam_intact: Whether PAM site is intact

        Returns:
            Formatted safety audit summary string
        """
        self.initialize()
        import torch

        # Build context query
        risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"
        query = (
            f"CRISPR off-target prediction {risk_level} risk "
            f"{'with ΔF508 deletion' if has_deletion else ''} "
            f"{'PAM disrupted' if not pam_intact else 'PAM intact'} "
            f"{num_mismatches} mismatches {seed_mismatches} in seed region"
        )

        # Retrieve relevant context
        retrieved_docs = self.retrieve(query, top_k=self.cfg.top_k_retrieval)
        context = "\n".join([doc["text"] for doc in retrieved_docs])

        # Build prompt for LLM
        prompt = (
            f"Based on the following genomic context, generate a safety assessment:\n\n"
            f"Context: {context}\n\n"
            f"Prediction details:\n"
            f"- Risk score: {risk_score:.3f} ({risk_level})\n"
            f"- Mismatches: {num_mismatches} (seed: {seed_mismatches})\n"
            f"- ΔF508 deletion: {'present' if has_deletion else 'absent'}\n"
            f"- Chromatin accessibility: {chromatin_score:.2f}\n"
            f"- PAM site: {'intact' if pam_intact else 'disrupted'}\n\n"
            f"Generate a brief safety assessment explaining the risk prediction."
        )

        # Generate with LLM
        device = next(self.llm.parameters()).device
        inputs = self.llm_tokenizer(
            prompt, return_tensors="pt",
            max_length=512, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=200,
                temperature=self.cfg.temperature,
                do_sample=True,
                top_p=0.9,
            )

        generated_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Format the full audit report
        report = self._format_audit_report(
            sgrna_seq, dna_seq, risk_score, risk_level,
            num_mismatches, seed_mismatches, has_deletion,
            chromatin_score, pam_intact, generated_text
        )

        return report

    def _format_audit_report(
        self, sgrna, dna, risk_score, risk_level,
        num_mm, seed_mm, has_del, chromatin, pam_intact, llm_text
    ):
        """Format the complete safety audit report."""
        deletion_str = "ΔF508 Deletion Detected" if has_del else "No Deletion"
        pam_str = "Intact (NGG)" if pam_intact else "Disrupted"

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║  NEURO-CRISPR-KAN | PREDICTIVE SAFETY AUDIT REPORT         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  sgRNA:     {sgrna[:20]}...                                  ║
║  Target:    {dna[:20]}...                                    ║
║  Mutation:  {deletion_str:<40s}     ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  [1] PREDICTIVE ANALYTICS                                    ║
║  ─────────────────────────────────────────                   ║
║  • Risk Score:        {risk_score:.4f}                       ║
║  • Risk Level:        {risk_level:<10s}                      ║
║  • Mismatches:        {num_mm} total ({seed_mm} in seed)     ║
║  • PAM Status:        {pam_str:<15s}                         ║
║  • Chromatin Access:  {chromatin:.3f}                        ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  [2] AI-GENERATED INTERPRETATION                             ║
║  ─────────────────────────────────────────                   ║
║  {llm_text[:200]:<55s}║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  [3] SAFETY RECOMMENDATION                                   ║
║  ─────────────────────────────────────────                   ║"""

        if risk_level == "HIGH":
            report += """
║  ⚠️  HIGH RISK: This sgRNA is predicted to be UNSAFE for    ║
║     therapeutic editing. High probability of off-target       ║
║     mutations. Redesign recommended.                          ║"""
        elif risk_level == "MODERATE":
            report += """
║  ⚡ MODERATE RISK: Proceed with caution. Additional          ║
║     experimental validation (GUIDE-seq) recommended           ║
║     before clinical application.                              ║"""
        else:
            report += """
║  ✅ LOW RISK: This sgRNA shows low off-target probability.   ║
║     Suitable for further experimental validation.             ║"""

        report += """
║                                                              ║
╚══════════════════════════════════════════════════════════════╝"""

        return report

    def generate_batch_audits(self, predictions_df, top_n: int = 5):
        """
        Generate audit reports for top-N highest risk predictions.
        """
        # Sort by risk score descending
        high_risk = predictions_df.nlargest(top_n, "risk_score")
        reports = []

        for _, row in high_risk.iterrows():
            report = self.generate_safety_audit(
                sgrna_seq=row.get("sgrna_seq", "N/A"),
                dna_seq=row.get("dna_seq", "N/A"),
                risk_score=row.get("risk_score", 0),
                num_mismatches=row.get("num_mismatches", 0),
                seed_mismatches=row.get("seed_mismatches", 0),
                has_deletion=bool(row.get("has_deletion", False)),
                chromatin_score=row.get("chromatin_score", 0.5),
                pam_intact=bool(row.get("pam_intact", True)),
            )
            reports.append(report)

        return reports


# ================================================================
# Simple template-based fallback (no LLM needed)
# ================================================================

def generate_template_audit(risk_score, num_mismatches, seed_mismatches,
                            has_deletion, chromatin_score, pam_intact):
    """
    Template-based safety audit (no LLM required).
    Use this as a fallback if LLM loading fails on Colab.
    """
    risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"

    reasons = []
    if risk_score > 0.7:
        if num_mismatches <= 2:
            reasons.append("Low mismatch count suggests strong binding affinity at this off-target locus")
        if seed_mismatches == 0:
            reasons.append("No mismatches in the critical seed region increases cleavage probability")
        if pam_intact:
            reasons.append("Intact PAM site (NGG) enables Cas9 recognition")
        if chromatin_score > 0.6:
            reasons.append("Open chromatin state increases accessibility")
        if has_deletion:
            reasons.append("ΔF508 deletion creates a structural void that may alter binding geometry")
    elif risk_score > 0.4:
        reasons.append("Moderate number of mismatches with partial seed region involvement")
        if has_deletion:
            reasons.append("Deletion topology introduces positional uncertainty")
    else:
        if num_mismatches > 3:
            reasons.append("High mismatch density reduces binding stability")
        if seed_mismatches > 1:
            reasons.append("Multiple seed region mismatches strongly inhibit Cas9 activity")
        if not pam_intact:
            reasons.append("Disrupted PAM site prevents Cas9 recognition")

    reasoning = "; ".join(reasons) if reasons else "Standard risk profile"

    return f"""
RISK LEVEL: {risk_level} ({risk_score:.4f})
REASONING: {reasoning}
RECOMMENDATION: {'REDESIGN sgRNA' if risk_level == 'HIGH' else 'VALIDATE EXPERIMENTALLY' if risk_level == 'MODERATE' else 'PROCEED WITH STANDARD PROTOCOLS'}
"""


if __name__ == "__main__":
    # Test template-based audit (no model loading)
    audit = generate_template_audit(
        risk_score=0.85,
        num_mismatches=1,
        seed_mismatches=0,
        has_deletion=True,
        chromatin_score=0.7,
        pam_intact=True,
    )
    print(audit)

    # Test RAG module (requires model downloads)
    # rag = RAGModule()
    # rag.initialize()
    # report = rag.generate_safety_audit(...)
