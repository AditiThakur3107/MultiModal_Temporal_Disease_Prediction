#!/usr/bin/env python3
# Multimodal baseline: Image(ResNet18) + Text(BioClinicalBERT) + EHR TimeSeries(stats)
# - Uses --csv (train), --val_csv, --test_csv
# - 100 epochs by default with Early Stopping (patience=20 on Val Micro-F1)
# - Saves best weights + labels + test metrics

import os, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

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

# -------------------- utils --------------------
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

# -------------------- dataset --------------------
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

# -------------------- model --------------------
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

# -------------------- eval --------------------
def evaluate(model, dl, device, crit=None):
    """
    If crit is provided -> returns (micro, macro, auc_mi, auc_ma, val_loss)
    Otherwise         -> returns (micro, macro, auc_mi, auc_ma)
    """
    model.eval()
    all_p, all_y = None, None
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dl:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch)

            if crit is not None:
                loss = crit(logits, batch["labels"])
                total_loss += loss.item()
                n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            ytrue = batch["labels"].cpu().numpy()

            all_p = probs if all_p is None else np.vstack([all_p, probs])
            all_y = ytrue if all_y is None else np.vstack([all_y, ytrue])

    preds = (all_p >= 0.5).astype(int)
    micro = f1_score(all_y, preds, average="micro", zero_division=0)
    macro = f1_score(all_y, preds, average="macro", zero_division=0)
    try:
        auc_mi = roc_auc_score(all_y, all_p, average="micro")
        auc_ma = roc_auc_score(all_y, all_p, average="macro")
    except Exception:
        auc_mi = auc_ma = float("nan")

    if crit is not None:
        val_loss = total_loss / max(1, n_batches)
        return micro, macro, auc_mi, auc_ma, val_loss
    else:
        return micro, macro, auc_mi, auc_ma

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="train CSV")
    ap.add_argument("--val_csv", default=None, help="validation CSV (optional)")
    ap.add_argument("--test_csv", default=None, help="test CSV (optional)")
    ap.add_argument("--image_dirs", default="images;./", help="semicolon-separated image folders")
    ap.add_argument("--timeseries_dir", default="timeseries", help="base dir for EHR csvs")
    ap.add_argument("--text_col", default="past_medical_history")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    # load train
    df_tr = pd.read_csv(args.csv)
    for col in ["subject_id","dicom_id","time_series",args.text_col]:
        assert col in df_tr.columns, f"Missing column: {col}"

    label_cols = find_label_cols(df_tr)
    assert len(label_cols) > 0, "No label columns detected (0/1 or CheXpert names)."
    print(f"Using labels: {label_cols}")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    img_dirs = [p for p in args.image_dirs.split(";") if p]

    # load val
    if args.val_csv:
        df_va = pd.read_csv(args.val_csv)
    else:
        ids = df_tr["subject_id"].astype(str)
        uniq = ids.dropna().unique()
        rng = np.random.default_rng(42)
        rng.shuffle(uniq)
        cut = int(0.8 * len(uniq))
        tr_ids = set(uniq[:cut])
        df_va = df_tr[~ids.isin(tr_ids)]
        df_tr = df_tr[ids.isin(tr_ids)]

    # load test (optional)
    df_te = pd.read_csv(args.test_csv) if args.test_csv else None

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
    model = FusionModel(
        num_labels=len(label_cols),
        freeze_bert=True,
        freeze_resnet=True
    ).to(device)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    steps = len(dl_tr) * max(1, args.epochs)
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=max(1, int(0.1 * steps)),
        num_training_steps=steps
    )
    crit = nn.BCEWithLogitsLoss()

    best_micro = -1.0
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0

        for batch in dl_tr:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch)
            loss = crit(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tr_loss += loss.item()

        avg_tr_loss = tr_loss / max(1, len(dl_tr))

        # validation metrics + loss
        mi, ma, auc_mi, auc_ma, val_loss = evaluate(model, dl_va, device, crit)

        print(f"Epoch {ep:03d}")
        print(f"  Train Loss : {avg_tr_loss:.4f}")
        print(f"  Val   Loss : {val_loss:.4f}")
        print(f"  Micro F1   : {mi:.4f}")
        print(f"  Macro F1   : {ma:.4f}")
        print(f"  AUC (micro): {auc_mi:.4f}")
        print(f"  AUC (macro): {auc_ma:.4f}")

        # early stopping on validation micro-F1
        if mi > best_micro:
            best_micro = mi
            no_improve = 0
            torch.save(model.state_dict(), "multimodal_fusion_best.pt")
            with open("fusion_label_cols.json", "w") as f:
                json.dump({"label_cols": label_cols}, f, indent=2)
            print("  -> Saved multimodal_fusion_best.pt + fusion_label_cols.json")
        else:
            no_improve += 1
            print(f"  -> No improvement for {no_improve} epoch(s). Patience={args.patience}")

        if no_improve >= args.patience:
            print(f"Early stopping triggered (no improvement for {args.patience} epochs).")
            break

    # optional test evaluation
    if df_te is not None:
        ds_te = TriModalDataset(
            df_te, args.text_col, img_dirs, args.timeseries_dir,
            label_cols, tokenizer, max_len=args.max_len
        )
        dl_te = DataLoader(
            ds_te, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        t_mi, t_ma, t_auc_mi, t_auc_ma = evaluate(model, dl_te, device)
        print(f"[TEST] Micro F1   : {t_mi:.4f}")
        print(f"[TEST] Macro F1   : {t_ma:.4f}")
        print(f"[TEST] AUC (mi)   : {t_auc_mi:.4f}")
        print(f"[TEST] AUC (ma)   : {t_auc_ma:.4f}")
        with open("metrics.json", "w") as f:
            json.dump(
                {
                    "val_best_micro": float(best_micro),
                    "test_micro": float(t_mi),
                    "test_macro": float(t_ma),
                    "test_auc_micro": float(t_auc_mi),
                    "test_auc_macro": float(t_auc_ma),
                },
                f,
                indent=2,
            )
        print("Saved metrics.json")

if __name__ == "__main__":
    main()
