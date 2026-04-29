"""
Smoke training v3: 5 epochs with TRUE constant learning rate.
Inline training loop to bypass the cosine scheduler entirely.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from configs.config import config
from utils.helpers import set_seed, get_device, count_parameters, save_checkpoint
from data.data_generation import generate_dataset, save_dataset
from data.encoding import create_dataloaders
from training.losses import CompoundLoss
from training.optimizer import create_optimizer


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0; total_focal = 0.0; total_reg = 0.0; n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        encoded = batch["encoded"].to(device)
        sgrna = batch["sgrna_seq"]; dna = batch["dna_seq"]
        labels = batch["label"].to(device)

        out = model(encoded, sgrna, dna)
        risk_logit = out["risk_logit"]
        spline_l1 = model.get_spline_l1_loss()
        loss, comps = criterion(risk_logit, labels, spline_l1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss); total_focal += float(comps["focal"]); total_reg += float(comps["spline_reg"]); n += 1
    return {"loss": total_loss/n, "focal": total_focal/n, "reg": total_reg/n}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0; correct = 0; total = 0; n = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        encoded = batch["encoded"].to(device)
        sgrna = batch["sgrna_seq"]; dna = batch["dna_seq"]
        labels = batch["label"].to(device)

        out = model(encoded, sgrna, dna)
        risk_logit = out["risk_logit"]
        spline_l1 = model.get_spline_l1_loss()
        loss, _ = criterion(risk_logit, labels, spline_l1)
        total_loss += float(loss); n += 1

        preds = (torch.sigmoid(risk_logit) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return {"loss": total_loss/n, "accuracy": correct/total}


def main():
    EPOCHS = 5
    set_seed(config.seed)
    device = get_device()

    csv_path = os.path.join(config.data.output_dir, "crispr_dataset.csv")
    print(f"[1/4] Loading dataset {csv_path}")
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else generate_dataset()
    if not os.path.exists(csv_path):
        save_dataset(df)

    print("[2/4] Building DataLoaders (null_tensor)...")
    loaders = create_dataloaders(
        df, encoder="null_tensor",
        batch_size=config.training.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
    )

    print("[3/4] Building NeuroCRISPRKAN model...")
    from models.neuro_crispr_kan import NeuroCRISPRKAN
    model = NeuroCRISPRKAN(config).to(device)
    p = count_parameters(model)
    print(f"  Total: {p['total']:,} | Trainable: {p['trainable']:,} ({p['trainable_pct']:.2f}%)")

    criterion = CompoundLoss(config.training)
    optimizer = create_optimizer(model, config.training)
    # No scheduler. Constant LR per param group as set by create_optimizer:
    # group 0 (LoRA): 1e-5, others (CNN, KAN, Fusion): 1e-4

    print(f"[4/4] Training {EPOCHS} epochs with CONSTANT LR (no scheduler)...")
    print(f"  Param-group LRs:")
    for i, g in enumerate(optimizer.param_groups):
        print(f"    group {i}: lr={g['lr']:.1e}, params={sum(p.numel() for p in g['params']):,}")

    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    t0 = time.time()
    for epoch in range(1, EPOCHS+1):
        ts = time.time()
        train_m = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_m = validate(model, loaders["val"], criterion, device)
        dt = time.time() - ts
        # Show ALL group LRs each epoch — proves no decay
        lrs = " ".join(f"g{i}={g['lr']:.1e}" for i, g in enumerate(optimizer.param_groups))
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss {train_m['loss']:.4f} (focal {train_m['focal']:.4f}, reg {train_m['reg']:.4f}) | "
              f"Val Loss {val_m['loss']:.4f} | Val Acc {val_m['accuracy']:.4f} | LRs[{lrs}] | {dt:.1f}s")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            save_checkpoint(model, optimizer, epoch, best_val_loss,
                            os.path.join(config.training.checkpoint_dir, "best_model.pt"))
            print(f"  >> New best (val_loss {best_val_loss:.4f})")

    print(f"\nElapsed: {(time.time()-t0)/60:.1f} min")

    print("\n[Eval] Test set...")
    from evaluation.evaluate import evaluate_model
    results = evaluate_model(model, loaders["test"], device)


if __name__ == "__main__":
    main()
