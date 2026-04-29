"""
Transformer Stream (DNABERT-2 + LoRA)
=====================================
Global semantic analysis using DNABERT-2 with LoRA fine-tuning.

DNABERT-2 uses BPE tokenization (no k-mer needed) and captures
long-range dependencies in DNA sequences via self-attention.

LoRA (Low-Rank Adaptation) allows efficient fine-tuning:
- Only trains small adapter matrices (rank 8)
- Base model weights stay frozen
- ~0.5% of parameters are trainable
- Fits on Colab T4 GPU

Architecture:
    sgRNA + DNA sequences (text)
    → DNABERT-2 tokenizer (BPE)
    → DNABERT-2 encoder (with LoRA on Q, V projections)
    → [CLS] token embedding
    → Linear projection → output_dim
"""

import torch
import torch.nn as nn

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config


class TransformerStream(nn.Module):
    """
    DNABERT-2 with LoRA adapter for global sequence understanding.

    NOTE: Requires `transformers` and `peft` packages.
    Install: pip install transformers peft triton
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.transformer

        self.cfg = cfg
        self.output_dim = cfg.output_dim

        # ------------------------------------------------------------------
        # Load DNABERT-2 model and tokenizer
        # ------------------------------------------------------------------
        from transformers import AutoTokenizer, AutoModel, AutoConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
        )

        # Load config first and patch missing attributes for compatibility
        model_config = AutoConfig.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
        )
        # DNABERT-2's bert_layers.py references pad_token_id but
        # the custom BertConfig doesn't define it (transformers v5+ issue)
        if not hasattr(model_config, 'pad_token_id') or model_config.pad_token_id is None:
            model_config.pad_token_id = self.tokenizer.pad_token_id or 0

        # Disable Flash Attention (triton kernel incompatible on Windows)
        model_config.use_flash_attn = False
        model_config._attn_implementation = "eager"

        try:
            self.base_model = AutoModel.from_pretrained(
                cfg.model_name,
                config=model_config,
                trust_remote_code=True,
            )
        except ValueError as e:
            # transformers 4.45+ trips on DNABERT-2's custom BertConfig vs HF BertConfig
            # mismatch in AutoModel.register. Fall back to loading the cached BertModel
            # class directly — bypasses the registration check.
            if "config_class" in str(e):
                from transformers import AutoModel as _AM
                from transformers.dynamic_module_utils import get_class_from_dynamic_module
                BertModelCls = get_class_from_dynamic_module(
                    "bert_layers.BertModel", cfg.model_name,
                )
                self.base_model = BertModelCls.from_pretrained(
                    cfg.model_name, config=model_config, trust_remote_code=True,
                )
            else:
                raise

        # DNABERT-2's bert_layers gates flash-attn purely on whether the
        # triton import succeeded — it ignores config.use_flash_attn. Newer
        # triton versions break the `tl.dot(..., trans_b=True)` call, so we
        # force the PyTorch fallback by nulling the module-level symbol.
        try:
            import sys as _sys
            for _name, _mod in list(_sys.modules.items()):
                if _name.endswith(".bert_layers") and "DNABERT" in _name:
                    _mod.flash_attn_qkvpacked_func = None
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Apply LoRA adapter
        # ------------------------------------------------------------------
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()

        # ------------------------------------------------------------------
        # Projection head: DNABERT-2 hidden_size → output_dim
        # ------------------------------------------------------------------
        hidden_size = self.base_model.config.hidden_size  # 768 for DNABERT-2
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, cfg.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def tokenize_sequences(self, sgrna_seqs, dna_seqs, device):
        """
        Tokenize sgRNA-DNA pairs for DNABERT-2.
        Concatenates sgRNA and DNA with a separator.

        Args:
            sgrna_seqs: List of sgRNA sequence strings
            dna_seqs: List of DNA sequence strings
            device: torch device

        Returns:
            Dict of tokenized inputs (input_ids, attention_mask)
        """
        # Concatenate sgRNA and DNA with space separator
        combined = [
            f"{sgrna} {dna}" for sgrna, dna in zip(sgrna_seqs, dna_seqs)
        ]

        tokens = self.tokenizer(
            combined,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )

        return {k: v.to(device) for k, v in tokens.items()}

    def forward(self, sgrna_seqs, dna_seqs, device=None):
        """
        Args:
            sgrna_seqs: List[str] of sgRNA sequences
            dna_seqs: List[str] of DNA target sequences
            device: torch device (defaults to model device)

        Returns:
            Tensor of shape (batch, output_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        # Tokenize
        inputs = self.tokenize_sequences(sgrna_seqs, dna_seqs, device)

        # Forward through DNABERT-2 + LoRA
        outputs = self.base_model(**inputs)

        # Extract [CLS] token embedding (first token)
        # DNABERT-2 returns a tuple (last_hidden_state, ...) not a ModelOutput
        hidden_state = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        cls_embedding = hidden_state[:, 0, :]  # (batch, hidden_size)

        # Project to output dimension
        features = self.projection(cls_embedding)  # (batch, output_dim)

        return features

    def get_attention_weights(self, sgrna_seqs, dna_seqs, device=None):
        """
        Extract attention weights for visualization/heatmaps.

        Returns:
            attention: Tuple of tensors, one per layer
            Each tensor: (batch, num_heads, seq_len, seq_len)
        """
        if device is None:
            device = next(self.parameters()).device

        inputs = self.tokenize_sequences(sgrna_seqs, dna_seqs, device)
        outputs = self.base_model(**inputs, output_attentions=True)

        return outputs.attentions  # Tuple of (batch, heads, seq, seq)


if __name__ == "__main__":
    print("Loading DNABERT-2 + LoRA...")
    model = TransformerStream()

    # Test with dummy sequences
    sgrna_list = ["ATCGATCGATCGATCGATCGNGG", "TTGCAAGCTTGCAAGCTTGCNGG"]
    dna_list = ["ATCGATCGATCGATCGATCGNGG", "TTGCAAGCTTGCAAGCTTGCNGG"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        features = model(sgrna_list, dna_list, device)
        print(f"Transformer output shape: {features.shape}")  # (2, 128)

        # Test attention extraction
        attentions = model.get_attention_weights(sgrna_list, dna_list, device)
        print(f"Number of attention layers: {len(attentions)}")
        print(f"Attention shape per layer: {attentions[0].shape}")
