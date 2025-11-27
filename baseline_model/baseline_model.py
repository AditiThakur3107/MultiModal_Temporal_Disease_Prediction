#!/usr/bin/env python3
# Multimodal baseline: Image(ResNet18) + Text(BioClinicalBERT) + EHR TimeSeries(stats)
# - Uses a SINGLE --csv file (small dataset ~100 rows)
# - Internally splits into Train / Val (80% / 20%)
# - Up to 100 epochs with Early Stopping (patience=20 on Val Micro-F1)
# - Logs Train/Val loss + F1 + AUROC/AUPRC like train_model.py (GNN baseline)

import os
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ---- sklearn for AUROC / AUPRC ----
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _SKLEARN_OK = True
except ImportError:
    print("[WARN] sklearn not found; AUROC/AUPRC will be NaN.")
    _SKLEARN_OK = False

# to scan medical images use pydicom
try:
    import pydicom
except Exception:
    pydicom = None

# columns in sample training data
CHEXPERT = [
    "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity",
    "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis",
    "Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"
]

# --------------------------------------------------
# Utils
# --------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# find columns having label 0,1,-1,-2
def find_label_cols(df):
    cols = [c for c in CHEXPERT if c in df.columns]
    if cols:
        return cols
    out = []
    for c in df.columns:
        if c in ["subject_id","study_id","dicom_id","time_series","past_medical_history"]:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        vals = set(pd.unique(s.dropna()))
        if len(vals) > 0 and (vals.issubset({-1,0,1}) or vals.issubset({0,1})):
            out.append(c)
    return out

# label nan and negative values as 0 (u_zero)
def map_labels(row_vals, policy="u_zero"):
    v = np.array(row_vals, dtype=float)
    v[np.isnan(v)] = 0.0
    if policy == "u_zero":
        v[v < 0] = 0.0
    elif policy == "u_one":
        v[v < 0] = 1.0
    return v.astype(np.float32)

# load images and convert to RGB. If no image return None
IMG_EXTS = [".png",".jpg",".jpeg",".bmp",".tif",".tiff",".dcm"]

def try_load_image(dicom_id, base_dirs):
    for base in base_dirs:
        for ext in IMG_EXTS:
            p = Path(base) / f"{dicom_id}{ext}"
            if p.exists():
                if ext == ".dcm" and pydicom is not None:
                    ds = pydicom.dcmread(str(p))
                    arr = ds.pixel_array.astype(np.float32)
                    arr = (255 * (arr - arr.min()) / (arr.ptp() + 1e-6)).astype(np.uint8)
                    return Image.fromarray(arr).convert("RGB")
                else:
                    return Image.open(str(p)).convert("RGB")
    return None

# reads a time-series CSV file, extracts stats, returns numeric features
def read_timeseries_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return None
    means = num.mean(axis=0, skipna=True).values
    stds  = num.std(axis=0, ddof=0).values
    lasts = num.ffill().bfill().iloc[-1].values
    feat = np.concatenate([means, stds, lasts], axis=0).astype(np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

class TriModalDataset(Dataset):
    def __init__(self, df, text_col, img_dirs, ts_dir, label_cols, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.img_dirs = [d for d in img_dirs if d]
        self.ts_dir = Path(ts_dir) if ts_dir else None
        self.label_cols = label_cols
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        # text -> BERT max character 3000
        text = str(row.get(self.text_col, "") or "")
        enc = self.tokenizer(
            text[:3000],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}

        # image -> ResNet use black image if empty
        dicom_id = str(row.get("dicom_id","") or "")
        img = try_load_image(dicom_id, self.img_dirs) if dicom_id else None
        if img is None:
            img = Image.new("RGB", (224,224), 0)
        item["img"] = self.tf(img)

        # time series -> stats (128-d padded)
        ts_entry = row.get("time_series","")
        ts_path = None
        if isinstance(ts_entry, str) and ts_entry:
            ts_path = ts_entry if os.path.isabs(ts_entry) else (
                str(self.ts_dir / ts_entry) if self.ts_dir else ts_entry
            )

        vec = np.zeros((128,), dtype=np.float32)
        if ts_path and os.path.exists(ts_path):
            feat = read_timeseries_csv(ts_path)
            if isinstance(feat, np.ndarray) and feat.size > 0:
                n = min(128, feat.size)
                vec[:n] = feat[:n]
        item["ts_feat"] = torch.tensor(vec, dtype=torch.float32)

        # labels
        y = map_labels([row.get(c, 0) for c in self.label_cols], policy="u_zero")
        item["labels"] = torch.tensor(y, dtype=torch.float32)

        return item

# --------------------------------------------------
# Model
# --------------------------------------------------

class FusionModel(nn.Module):
    def __init__(self, num_labels, freeze_bert=True, freeze_resnet=True):
        super().__init__()
        # image encoder
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # -> 512-d
        if freeze_resnet:
            for p in self.resnet.parameters():
                p.requires_grad = False

        # text encoder
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")  # 768-d
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # time-series projection
        self.ts_proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # fusion head
        self.head = nn.Sequential(
            nn.Linear(512 + self.bert.config.hidden_size + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )

    def forward(self, batch):
        # image embedding (frozen)
        with torch.no_grad():
            img_emb = self.resnet(batch["img"])          # (B,512)

        # text embedding (CLS token)
        out = self.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None)
        )
        txt_emb = out.last_hidden_state[:, 0, :]         # (B,768)

        # time-series embedding
        ts_emb = self.ts_proj(batch["ts_feat"])          # (B,128)

        # concat embeddings
        z = torch.cat([img_emb, txt_emb, ts_emb], dim=1) # (B,1408)
        return self.head(z)

# --------------------------------------------------
# Metrics (like train_model.py)
# --------------------------------------------------

def micro_f1(y_true, y_pred):
    """
    y_true, y_pred: tensors of shape [N, L] with {0,1}.
    """
    y_true = y_true.int()
    y_pred = y_pred.int()

    tp = (y_true * y_pred).sum().item()
    fp = ((1 - y_true) * y_pred).sum().item()
    fn = (y_true * (1 - y_pred)).sum().item()

    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1


def macro_f1(y_true, y_pred):
    """
    Compute macro F1 across labels.
    y_true, y_pred: [N, L]
    """
    y_true = y_true.int()
    y_pred = y_pred.int()

    n_labels = y_true.shape[1]
    f1s = []
    for j in range(n_labels):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        tp = (yt * yp).sum().item()
        fp = ((1 - yt) * yp).sum().item()
        fn = (yt * (1 - yp)).sum().item()

        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        f1s.append(f1)

    if not f1s:
        return 0.0
    return float(sum(f1s) / len(f1s))


def auroc_auprc_micro_macro(y_true, y_prob):
    """
    Compute micro & macro AUROC and AUPRC.
    y_true, y_prob: tensors [N, L]
    Returns: auroc_mi, auroc_ma, auprc_mi, auprc_ma
    """
    if not _SKLEARN_OK:
        return float("nan"), float("nan"), float("nan"), float("nan")

    yt = y_true.numpy()
    yp = y_prob.numpy()

    # micro (flatten across labels)
    try:
        auroc_mi = roc_auc_score(yt.ravel(), yp.ravel())
    except ValueError:
        auroc_mi = float("nan")

    try:
        auprc_mi = average_precision_score(yt.ravel(), yp.ravel())
    except ValueError:
        auprc_mi = float("nan")

    # macro: per label, average only over labels with both pos & neg
    L = yt.shape[1]
    aurocs = []
    auprcs = []
    for j in range(L):
        yj = yt[:, j]
        pj = yp[:, j]
        # need at least one pos and one neg
        if yj.max() == yj.min():
            continue
        try:
            au_j = roc_auc_score(yj, pj)
            pr_j = average_precision_score(yj, pj)
            aurocs.append(au_j)
            auprcs.append(pr_j)
        except ValueError:
            continue

    auroc_ma = float(np.mean(aurocs)) if aurocs else float("nan")
    auprc_ma = float(np.mean(auprcs)) if auprcs else float("nan")
    return auroc_mi, auroc_ma, auprc_mi, auprc_ma

# --------------------------------------------------
# Train / Eval loops (like train_model.py)
# --------------------------------------------------

def train_one_epoch(model, dl, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_true = []
    all_pred = []

    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch)
        y_true = batch["labels"]

        if y_true.numel() == 0:
            continue

        loss = criterion(logits, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_true.shape[0]

        y_prob = torch.sigmoid(logits)
        y_hat = (y_prob >= 0.5).float()

        all_true.append(y_true.detach().cpu())
        all_pred.append(y_hat.detach().cpu())

    if not all_true:
        return 0.0, 0.0, 0.0

    all_true = torch.cat(all_true, dim=0)
    all_pred = torch.cat(all_pred, dim=0)

    avg_loss = total_loss / all_true.shape[0]
    mi_f1 = micro_f1(all_true, all_pred)
    ma_f1 = macro_f1(all_true, all_pred)

    return avg_loss, mi_f1, ma_f1


@torch.no_grad()
def eval_model(model, dl, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []

    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch)
        y_true = batch["labels"]

        if y_true.numel() == 0:
            continue

        loss = criterion(logits, y_true)
        total_loss += loss.item() * y_true.shape[0]

        y_prob = torch.sigmoid(logits)
        y_hat = (y_prob >= 0.5).float()

        all_true.append(y_true.detach().cpu())
        all_pred.append(y_hat.detach().cpu())
        all_prob.append(y_prob.detach().cpu())

    if not all_true:
        return 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan"), float("nan")

    all_true = torch.cat(all_true, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    all_prob = torch.cat(all_prob, dim=0)

    avg_loss = total_loss / all_true.shape[0]
    mi_f1 = micro_f1(all_true, all_pred)
    ma_f1 = macro_f1(all_true, all_pred)

    auroc_mi, auroc_ma, auprc_mi, auprc_ma = auroc_auprc_micro_macro(all_true, all_prob)

    return avg_loss, mi_f1, ma_f1, auroc_mi, auroc_ma, auprc_mi, auprc_ma

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="single CSV (all ~100 records)")
    ap.add_argument("--image_dirs", default="images;./", help="semicolon-separated image folders")
    ap.add_argument("--timeseries_dir", default="timeseries", help="base dir for EHR csvs")
    ap.add_argument("--text_col", default="past_medical_history")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # load full dataset (single CSV, ~100 rows)
    df = pd.read_csv(args.csv)
    for col in ["subject_id","dicom_id","time_series",args.text_col]:
        assert col in df.columns, f"Missing column: {col}"

    label_cols = find_label_cols(df)
    assert len(label_cols) > 0, "No label columns detected (0/1 or CheXpert names)."
    print(f"[INFO] Using labels: {label_cols}")

    # simple 80/20 train/val split (row-based, good enough for tiny dataset)
    n = len(df)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    cut = max(1, int(0.8 * n))  # at least 1 for small data
    tr_idx = idx[:cut]
    va_idx = idx[cut:] if cut < n else idx[-max(1, n // 5):]  # ensure some val

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)

    print(f"[INFO] Train size: {len(df_tr)} rows")
    print(f"[INFO] Val   size: {len(df_va)} rows")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    img_dirs = [p for p in args.image_dirs.split(";") if p]

    ds_tr = TriModalDataset(
        df_tr, args.text_col, img_dirs, args.timeseries_dir,
        label_cols, tokenizer, max_len=args.max_len
    )
    ds_va = TriModalDataset(
        df_va, args.text_col, img_dirs, args.timeseries_dir,
        label_cols, tokenizer, max_len=args.max_len
    )

    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = FusionModel(
        num_labels=len(label_cols),
        freeze_bert=True,
        freeze_resnet=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    steps = len(dl_tr) * max(1, args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * steps)),
        num_training_steps=steps
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mi_f1, train_ma_f1 = train_one_epoch(
            model, dl_tr, optimizer, criterion, device
        )
        # step scheduler once per epoch (already stepped per batch, but to stay close
        # to baseline you could move scheduler.step() here only if you want)
        scheduler.step()

        (
            val_loss,
            val_mi_f1,
            val_ma_f1,
            auroc_mi,
            auroc_ma,
            auprc_mi,
            auprc_ma,
        ) = eval_model(model, dl_va, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f}, miF1: {train_mi_f1:.4f}, maF1: {train_ma_f1:.4f} | "
            f"Val loss: {val_loss:.4f}, miF1: {val_mi_f1:.4f}, maF1: {val_ma_f1:.4f} | "
            f"AUROC(mi): {auroc_mi:.4f}, AUROC(ma): {auroc_ma:.4f}, "
            f"AUPRC(mi): {auprc_mi:.4f}, AUPRC(ma): {auprc_ma:.4f}"
        )

        # Early stopping based on validation micro-F1 (like train_model.py)
        if val_mi_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_mi_f1
            best_state = model.state_dict()
            patience_counter = 0
            print(f"[INFO] New best val micro-F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement for {patience_counter} epoch(s)")
            if patience_counter >= args.patience:
                print("[INFO] Early stopping triggered")
                break

    # Save best model + metadata
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), "multimodal_fusion_best.pt")
    with open("fusion_label_cols.json", "w") as f:
        json.dump(
            {
                "label_cols": label_cols,
                "num_rows_total": int(len(df)),
                "num_rows_train": int(len(df_tr)),
                "num_rows_val": int(len(df_va)),
            },
            f,
            indent=2,
        )
    with open("metrics.json", "w") as f:
        json.dump({"val_best_micro": float(best_val_f1)}, f, indent=2)
    print("[INFO] Saved multimodal_fusion_best.pt, fusion_label_cols.json, metrics.json")


if __name__ == "__main__":
    main()
