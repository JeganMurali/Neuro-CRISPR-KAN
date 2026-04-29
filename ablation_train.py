"""
Ablation training: identical setup to full_train.py but with encoder='zero_pad'.
Saves to ./checkpoints/ablation_zeropad/
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
from data.encoding import create_dataloaders
from training.losses import CompoundLoss
from training.optimizer import create_optimizer

# Override CNN input channels for 4-channel zero-pad encoding
config.cnn.input_channels = 4
config.data.null_tensor_dim = 4  # used by some downstream code; safe override


from full_train import train_one_epoch, eval_pass


def main():
    EPOCHS = 50
    set_seed(config.seed)
    device = get_device()

    csv_path = os.path.join(config.data.output_dir, "crispr_dataset.csv")
    print(f"[1/4] Loading dataset {csv_path}")
    df = pd.read_csv(csv_path)

    print("[2/4] Building DataLoaders (ZERO-PAD encoder, 4 channels)...")
    loaders = create_dataloaders(
        df, encoder="zero_pad",
        batch_size=config.training.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
    )

    print("[3/4] Building NeuroCRISPRKAN (CNN input_channels=4)...")
    from models.neuro_crispr_kan import NeuroCRISPRKAN
    model = NeuroCRISPRKAN(config).to(device)
    p = count_parameters(model)
    print(f"  Total: {p['total']:,} | Trainable: {p['trainable']:,} ({p['trainable_pct']:.2f}%)")

    criterion = CompoundLoss(config.training)
    optimizer = create_optimizer(model, config.training)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    ckpt_dir = os.path.join(config.training.checkpoint_dir, "ablation_zeropad")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float("inf"); best_auroc = 0.0; best_epoch = 0
    history = []
    t0 = time.time()
    print(f"\n[4/4] Training {EPOCHS} epochs (Zero-Pad ablation)")

    for epoch in range(1, EPOCHS+1):
        ts = time.time()
        train_m = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_m, _, _, _ = eval_pass(model, loaders["val"], criterion, device)
        scheduler.step()
        dt = time.time() - ts

        cur_lrs = [g["lr"] for g in optimizer.param_groups]
        print(
            f"E{epoch:02d}/{EPOCHS} | "
            f"trL {train_m['loss']:.4f} | vL {val_m['loss']:.4f} | "
            f"vAcc {val_m['accuracy']:.3f} vAUC {val_m['auroc']:.3f} vF1 {val_m['f1']:.3f} vRec {val_m['recall']:.3f} | "
            f"lr {cur_lrs[1]:.1e} | {dt:.1f}s",
            flush=True,
        )
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
                        **{f"val_{k}": v for k, v in val_m.items()}, "lr_main": cur_lrs[1], "time": dt})

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]; best_auroc = val_m["auroc"]; best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, best_val_loss,
                            os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  >> best (vL {best_val_loss:.4f} | vAUC {best_auroc:.3f})", flush=True)

    elapsed = (time.time()-t0)/60
    print(f"\nElapsed: {elapsed:.1f} min")
    print(f"Best epoch: {best_epoch} | best val_loss {best_val_loss:.4f} | best val_auroc {best_auroc:.3f}")

    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n[Test] Test set with BEST checkpoint...")
    ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m, y_test, p_test, hd_test = eval_pass(model, loaders["test"], criterion, device)

    print("\n" + "="*60)
    print("  ZERO-PAD ABLATION TEST RESULTS")
    print("="*60)
    for k, v in test_m.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    print("="*60)

    np.savez(os.path.join(ckpt_dir, "test_predictions.npz"),
             y=y_test, p=p_test, has_deletion=hd_test)


if __name__ == "__main__":
    main()
