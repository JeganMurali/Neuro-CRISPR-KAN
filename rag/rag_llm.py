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
    {
        "id": "cftr_gene_anatomy",
        "text": "The CFTR gene spans approximately 250 kb on chromosome 7q31.2 and contains 27 exons encoding a 1480-amino-acid chloride channel. The ΔF508 mutation occurs in exon 11 (legacy numbering: exon 10) and disrupts the first nucleotide-binding domain (NBD1), causing protein misfolding, ER retention, and proteasomal degradation. CFTR mutations are classified into six functional classes (I–VI) based on their effect on protein synthesis, processing, gating, conductance, abundance, or stability.",
        "category": "biology",
    },
    {
        "id": "cas9_cleavage_mechanism",
        "text": "SpCas9 induces a blunt-end double-strand break (DSB) approximately 3 bp upstream of the PAM. After PAM recognition, Cas9 undergoes a conformational rearrangement that licenses RNA-DNA heteroduplex formation, R-loop expansion from PAM-proximal to PAM-distal, and HNH/RuvC nuclease activation. Mismatches that prevent full R-loop propagation reduce or abolish cleavage even if binding occurs.",
        "category": "mechanism",
    },
    {
        "id": "dna_repair_pathways",
        "text": "DSBs introduced by Cas9 are repaired by non-homologous end joining (NHEJ, error-prone, dominant in non-dividing cells), microhomology-mediated end joining (MMEJ, generates predictable deletions), or homology-directed repair (HDR, requires donor template, restricted to S/G2 phase). Off-target NHEJ events are the dominant safety concern in therapeutic editing because they can introduce frameshifts in tumor suppressors or activate oncogenes via translocation.",
        "category": "safety",
    },
    {
        "id": "off_target_detection_assays",
        "text": "Empirical off-target profiling combines unbiased genome-wide assays — GUIDE-seq (integrates dsODN tags at DSBs), DISCOVER-seq (anti-MRE11 ChIP-seq), CIRCLE-seq (in vitro circularised library), CHANGE-seq (tagmented genome-wide), and SITE-seq — with targeted amplicon deep sequencing for confirmation. Computational predictions like Neuro-CRISPR-KAN should be cross-validated against at least one unbiased empirical assay before clinical use.",
        "category": "validation",
    },
    {
        "id": "high_fidelity_cas_variants",
        "text": "Engineered high-fidelity Cas9 variants reduce off-target activity: eSpCas9 (Slaymaker 2016, weakens non-target strand contacts), SpCas9-HF1 (Kleinstiver 2016, weakens target strand H-bonds), HypaCas9 (Chen 2017, REC3 mutations), and evoCas9. For therapeutic CFTR editing, pairing a high-fidelity variant with a computational pre-screen like Neuro-CRISPR-KAN provides defense-in-depth against unintended cleavage.",
        "category": "engineering",
    },
    {
        "id": "mismatch_position_tolerance",
        "text": "Cas9 mismatch tolerance is highly position-dependent. Positions 1–3 (PAM-distal) are well tolerated; positions 4–10 (mid-protospacer) tolerate single mismatches with reduced activity; positions 11–20 (seed and PAM-proximal) are intolerant — mismatches here typically reduce activity by 80–99%. Two adjacent mismatches anywhere in the protospacer are usually sufficient to abolish cleavage.",
        "category": "mechanism",
    },
    {
        "id": "dna_rna_bulges",
        "text": "Off-target sites can contain not just mismatches but also DNA bulges (extra DNA nucleotide) or RNA bulges (extra sgRNA nucleotide). Bulges are particularly problematic because they cause frameshifted alignments that simple mismatch counts miss. Models trained only on mismatch features (e.g., classic CFD score) systematically under-predict bulge-containing off-targets.",
        "category": "computational",
    },
    {
        "id": "oncogene_safety_loci",
        "text": "Off-target cleavage is most dangerous in tumor suppressor genes (TP53, RB1, PTEN, BRCA1/2, APC, NF1) and proto-oncogenes (MYC, RAS family, BCL2, ABL1). Cleavage in TP53 in particular can confer growth advantage to edited cells and drive oncogenic clonal expansion. For CFTR therapeutics, pre-screening candidate sgRNAs against these loci is recommended.",
        "category": "clinical",
    },
    {
        "id": "casgevy_precedent",
        "text": "Casgevy (exagamglogene autotemcel, exa-cel) is the first FDA-approved CRISPR therapy (Dec 2023, sickle cell and beta-thalassaemia). It uses ex vivo HSPC editing of BCL11A enhancer with electroporated SpCas9 RNP. This precedent established the regulatory pathway: comprehensive computational off-target prediction, GUIDE-seq validation, amplicon deep sequencing, and long-term clonal tracking are all expected before approval.",
        "category": "clinical",
    },
    {
        "id": "cftr_modulator_landscape",
        "text": "Current CF therapy is dominated by small-molecule CFTR modulators (Trikafta = elexacaftor + tezacaftor + ivacaftor, approved 2019), which work for ~90% of CF patients carrying at least one ΔF508 allele. CRISPR offers a one-time curative alternative, particularly for the ~10% of patients with rare mutations not addressed by modulators. Therapeutic delivery candidates include AAV (limited cargo), LNP-mRNA (transient, low immunogenicity), and ex vivo airway basal-cell editing.",
        "category": "clinical",
    },
    {
        "id": "delivery_modality_safety",
        "text": "Cas9 delivery modality directly affects off-target risk profile. Plasmid DNA: long expression → highest off-target burden. AAV: 4–6 week expression → moderate off-target accumulation, plus genotoxic risk from viral integration. LNP-mRNA: 24–72 h expression → lowest off-target burden. RNP electroporation: hours of activity → minimal off-target burden, preferred for ex vivo therapy.",
        "category": "delivery",
    },
    {
        "id": "lora_low_rank_adaptation",
        "text": "LoRA (Low-Rank Adaptation, Hu et al. 2021) injects trainable low-rank matrices A∈R^(d×r), B∈R^(r×d) alongside frozen weights W: ΔW = BA, with r≪d. For DNABERT-2 117M with rank 8 on Wqkv projections, this trains only 294,912 parameters (0.25% of total) while matching full fine-tuning performance on many tasks. LoRA prevents catastrophic forgetting of pre-trained genomic context and slashes training cost.",
        "category": "architecture",
    },
    {
        "id": "focal_loss_motivation",
        "text": "Focal loss (Lin et al. 2017) addresses class imbalance by down-weighting easy negatives: FL(p_t) = -α(1-p_t)^γ log(p_t). For γ=2, an example with p_t=0.9 contributes 100× less than at γ=0. In off-target prediction, where ~70% of synthetic samples are negative (safe), focal loss prevents the model from being dominated by trivially-rejected sites and forces gradient on borderline cases.",
        "category": "training",
    },
    {
        "id": "dnabert2_pretraining",
        "text": "DNABERT-2 (Zhou et al. 2023) is a transformer pre-trained on the multi-species genome (32 GB across 135 species) using byte-pair encoding (BPE) tokenization with a 4096-token vocabulary. Compared to character-level DNABERT, BPE captures variable-length motif statistics and yields a 117M-parameter model that consistently outperforms older nucleotide LMs on regulatory-element classification, off-target, and splicing benchmarks.",
        "category": "architecture",
    },
    {
        "id": "null_tensor_innovation",
        "text": "The Null Tensor encoding (this work) extends standard one-hot DNA encoding from 4 channels (A,T,G,C) to 5 channels by adding an explicit GAP channel activated at deletion positions. Unlike zero-padding which represents deletions as featureless noise vectors, the Null Tensor preserves positional alignment of flanking nucleotides and gives the CNN a learnable signal for structural variants. Empirically this lifts ΔF508 off-target recall by ~6 percentage points on our benchmark.",
        "category": "architecture",
    },
    {
        "id": "chromatin_atac_signal",
        "text": "Chromatin accessibility, measured by ATAC-seq or DNase-seq, predicts in-vivo Cas9 cleavage efficiency. Highly accessible regions (top decile of ATAC signal) show 3–10× higher off-target editing than closed heterochromatin. Cell-type matters: airway basal cells (the CF therapeutic target) have a different accessibility landscape than HEK293 or hematopoietic cells, so off-target predictions should be calibrated against cell-type-specific epigenome data.",
        "category": "epigenetics",
    },
    {
        "id": "regulatory_pathway_crispr",
        "text": "CRISPR therapeutics fall under FDA Center for Biologics Evaluation and Research (CBER), regulated as gene therapies (21 CFR 1271). Pre-IND requirements include: comprehensive off-target characterization (in silico + at least one unbiased assay), karyotype analysis, large genomic rearrangement screening, and tumorigenicity studies. ICH guidelines E14 and S6(R1) apply for safety pharmacology and biotech-derived products. For CFTR ΔF508, EMA additionally requires CFTR functional assays in patient-derived organoids.",
        "category": "regulatory",
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


# ================================================================
# LLAMA 3.1 8B SAFETY AUDITOR (HF transformers + 4-bit quant)
# ================================================================
_LLAMA_SINGLETON = None


class LlamaSafetyAuditor:
    """RAG-backed safety auditor using Llama 3.1 8B Instruct (4-bit)."""

    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"

    SYSTEM_PROMPT = (
        "You are a CRISPR safety auditor reviewing an off-target prediction "
        "from a hybrid CNN+DNABERT-2+KAN model trained on CFTR ΔF508 contexts. "
        "Write a concise clinical-style verdict (4–6 sentences). "
        "Cite the relevant biology from the provided CONTEXT when explaining the risk. "
        "End with one explicit RECOMMENDATION line."
    )

    def __init__(self):
        self.tok = None
        self.model = None
        self.embed = None
        self.kb_emb = None
        self._ready = False

    def initialize(self):
        if self._ready:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from sentence_transformers import SentenceTransformer

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tok = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16,
        )
        self.model.eval()

        self.embed = SentenceTransformer(self.EMBED_ID)
        texts = [d["text"] for d in GENOMIC_KNOWLEDGE_BASE]
        self.kb_emb = self.embed.encode(texts, convert_to_tensor=True)
        self._ready = True

    def retrieve(self, query: str, top_k: int = 3):
        import torch
        q = self.embed.encode(query, convert_to_tensor=True)
        sim = torch.nn.functional.cosine_similarity(q.unsqueeze(0), self.kb_emb, dim=1)
        idx = sim.argsort(descending=True)[:top_k].cpu().numpy()
        return [GENOMIC_KNOWLEDGE_BASE[i] for i in idx]

    def generate(self, sgrna, dna, risk_score, num_mismatches, seed_mismatches,
                 has_deletion, chromatin_score, pam_intact, max_new_tokens=220):
        import torch
        self.initialize()
        risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"
        query = (f"{risk_level} off-target risk "
                 f"{'ΔF508 ' if has_deletion else ''}"
                 f"{num_mismatches} mismatches {seed_mismatches} seed "
                 f"{'PAM disrupted' if not pam_intact else 'PAM intact'}")
        docs = self.retrieve(query, top_k=3)
        ctx = "\n".join(f"- {d['text']}" for d in docs)

        user = (
            f"PREDICTION SUMMARY\n"
            f"  sgRNA:        {sgrna}\n"
            f"  DNA target:   {dna}\n"
            f"  Risk score:   {risk_score:.3f}  ({risk_level})\n"
            f"  Mismatches:   {num_mismatches} total · {seed_mismatches} in seed (pos 10–20)\n"
            f"  ΔF508:        {'present' if has_deletion else 'absent'}\n"
            f"  PAM (NGG):    {'intact' if pam_intact else 'disrupted'}\n"
            f"  Chromatin:    {chromatin_score:.2f}\n\n"
            f"CONTEXT (retrieved from knowledge base):\n{ctx}\n\n"
            f"Write the safety verdict now."
        )
        msgs = [{"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return {"verdict": text, "retrieved": [d["id"] for d in docs],
                "risk_level": risk_level}


def get_llama_auditor() -> LlamaSafetyAuditor:
    """Process-level singleton so the model loads once."""
    global _LLAMA_SINGLETON
    if _LLAMA_SINGLETON is None:
        _LLAMA_SINGLETON = LlamaSafetyAuditor()
    return _LLAMA_SINGLETON


def generate_llm_audit(sgrna, dna, risk_score, num_mismatches, seed_mismatches,
                       has_deletion, chromatin_score, pam_intact):
    """Convenience wrapper used by the Streamlit UI."""
    return get_llama_auditor().generate(
        sgrna, dna, risk_score, num_mismatches, seed_mismatches,
        has_deletion, chromatin_score, pam_intact,
    )


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
