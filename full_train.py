"""
Full training: 50 epochs, cosine LR (paper spec), with per-epoch AUROC logging.
Saves best model on val_loss to ./checkpoints/best_model.pt
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

from configs.config import config
from utils.helpers import set_seed, get_device, count_parameters, save_checkpoint
from data.data_generation import generate_dataset, save_dataset
from data.encoding import create_dataloaders
from training.losses import CompoundLoss
from training.optimizer import create_optimizer


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tloss = tfocal = treg = 0.0; n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        encoded = batch["encoded"].to(device)
        sgrna = batch["sgrna_seq"]; dna = batch["dna_seq"]
        labels = batch["label"].to(device)

        out = model(encoded, sgrna, dna)
        spline_l1 = model.get_spline_l1_loss()
        loss, comps = criterion(out["risk_logit"], labels, spline_l1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tloss += float(loss); tfocal += float(comps["focal"]); treg += float(comps["spline_reg"]); n += 1
    return {"loss": tloss/n, "focal": tfocal/n, "reg": treg/n}


@torch.no_grad()
def eval_pass(model, loader, criterion, device):
    model.eval()
    tloss = 0.0; n = 0
    all_labels = []; all_probs = []; all_has_del = []
    for batch in tqdm(loader, desc="Eval", leave=False):
        encoded = batch["encoded"].to(device)
        sgrna = batch["sgrna_seq"]; dna = batch["dna_seq"]
        labels = batch["label"].to(device)
        out = model(encoded, sgrna, dna)
        spline_l1 = model.get_spline_l1_loss()
        loss, _ = criterion(out["risk_logit"], labels, spline_l1)
        tloss += float(loss); n += 1
        probs = torch.sigmoid(out["risk_logit"]).cpu().numpy().flatten()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy().flatten())
        all_has_del.append(batch["has_deletion"].cpu().numpy().flatten())

    y = np.concatenate(all_labels)
    p = np.concatenate(all_probs)
    hd = np.concatenate(all_has_del)
    preds = (p > 0.5).astype(int)
    metrics = {
        "loss": tloss/n,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "mcc": matthews_corrcoef(y, preds),
    }
    # Deletion-stratified recall
    if hd.sum() > 0:
        del_y, del_p = y[hd == 1], preds[hd == 1]
        nodel_y, nodel_p = y[hd == 0], preds[hd == 0]
        metrics["recall_deletion"] = recall_score(del_y, del_p, zero_division=0) if del_y.sum() > 0 else float("nan")
        metrics["recall_no_deletion"] = recall_score(nodel_y, nodel_p, zero_division=0) if nodel_y.sum() > 0 else float("nan")
    return metrics, y, p, hd


def main():
    EPOCHS = 50
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

    print("[3/4] Building NeuroCRISPRKAN...")
    from models.neuro_crispr_kan import NeuroCRISPRKAN
    model = NeuroCRISPRKAN(config).to(device)
    p = count_parameters(model)
    print(f"  Total: {p['total']:,} | Trainable: {p['trainable']:,} ({p['trainable_pct']:.2f}%)")

    criterion = CompoundLoss(config.training)
    optimizer = create_optimizer(model, config.training)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf"); best_auroc = 0.0; best_epoch = 0
    history = []
    t0 = time.time()
    print(f"\n[4/4] Training {EPOCHS} epochs (CosineAnnealingLR T_max={EPOCHS}, eta_min=1e-7)")
    print(f"  Param-group LRs (initial): " + ", ".join(f"g{i}={g['lr']:.1e}" for i, g in enumerate(optimizer.param_groups)))

    for epoch in range(1, EPOCHS+1):
        ts = time.time()
        train_m = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_m, _, _, _ = eval_pass(model, loaders["val"], criterion, device)
        scheduler.step()
        dt = time.time() - ts

        cur_lrs = [g["lr"] for g in optimizer.param_groups]
        print(
            f"E{epoch:02d}/{EPOCHS} | "
            f"trL {train_m['loss']:.4f} (foc {train_m['focal']:.4f} reg {train_m['reg']:.4f}) | "
            f"vL {val_m['loss']:.4f} | "
            f"vAcc {val_m['accuracy']:.3f} vAUC {val_m['auroc']:.3f} vF1 {val_m['f1']:.3f} vRec {val_m['recall']:.3f} | "
            f"lr {cur_lrs[1]:.1e} | {dt:.1f}s",
            flush=True,
        )
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
                        **{f"val_{k}": v for k, v in val_m.items()}, "lr_main": cur_lrs[1], "time": dt})

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]; best_auroc = val_m["auroc"]; best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, best_val_loss,
                            os.path.join(config.training.checkpoint_dir, "best_model.pt"))
            print(f"  >> best (vL {best_val_loss:.4f} | vAUC {best_auroc:.3f})", flush=True)

    elapsed = (time.time()-t0)/60
    print(f"\nElapsed: {elapsed:.1f} min")
    print(f"Best epoch: {best_epoch} | best val_loss {best_val_loss:.4f} | best val_auroc {best_auroc:.3f}")

    # Save history
    with open(os.path.join(config.training.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final test
    print("\n[Test] Running on test set with BEST checkpoint...")
    ckpt = torch.load(os.path.join(config.training.checkpoint_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m, y_test, p_test, hd_test = eval_pass(model, loaders["test"], criterion, device)

    print("\n" + "="*60)
    print("  TEST RESULTS (best checkpoint)")
    print("="*60)
    for k, v in test_m.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    print("="*60)

    # Save predictions for downstream analysis
    np.savez(os.path.join(config.training.checkpoint_dir, "test_predictions.npz"),
             y=y_test, p=p_test, has_deletion=hd_test)
    print(f"\nSaved: {config.training.checkpoint_dir}/best_model.pt, history.json, test_predictions.npz")


if __name__ == "__main__":
    main()
