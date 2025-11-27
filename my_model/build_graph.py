#!/usr/bin/env python3
"""
Build PyG HeteroData graphs (visit + entity nodes) from the multimodal dataset.

For each patient (subject_id) we create one HeteroData graph:

Nodes
-----
- visit:
    x : [n_visits, D_visit]
        concat[ image_embedding || timeseries_embedding || pmh_text_embedding ]
    y : [n_visits, n_labels]  (multi-label 0/1)
    img_path : list of image file paths (str)
    ts_path  : list of timeseries csv paths (str)
    text_raw : list of PMH strings
    label_cols : list of label column names

- entity:
    x : [n_entities, TEXT_DIM]  (CLS embedding of entity term from clinical BERT)
    terms : list[str] entity surface forms

Edges
-----
- ("visit", "to", "entity") and ("entity", "to", "visit"):
    unweighted edges, one per (visit, entity) occurrence.

- ("visit", "to", "visit") and ("visit", "to", "visit_rev"):
    temporal edges between successive visits (sorted by stay_id only).

Usage
-----
python3 build_graph_full.py \
  --input ../datasets/sample_training_data.csv \
  --outdir ./input_graphs_full \
  --img_root ../images \
  --ts_root ../timeseries
"""

import argparse
import json
import os
import re
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from medical_ner import run_medical_ner  # keep this as a separate file

from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image

# ------------------------------------------------------
# Globals / configuration
# ------------------------------------------------------

NER_BACKEND = "medical_ner.run_medical_ner"
TEXT_MODEL_NAME = "samrawal/bert-base-uncased_clinical-ner"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image encoder globals
_IMG_MODEL = None
_IMG_TRANSFORM = None
_IMG_DIM = None
_IMG_SIZE = 224

# text encoder globals
_TEXT_TOKENIZER = None
_TEXT_MODEL = None
_TEXT_DIM = None

# timeseries globals
_TS_NUMERIC_COLS = None
_TS_CATEGORICAL_COLS = None
_TS_DIM = None  # = 4 * #numeric + TEXT_DIM (for categorical summary)

# stopwords for entity cleaning
STOPWORDS = set(
    """
    a an the and or of to for with on in at by from no not none denies deny
    denied without is are was were be been being has have had hx past history current currently
    since as per than then pt patient mild severe moderate acute chronic day days week weeks
    month months year years today yesterday status post
    """.split()
)

GLUED_SPLIT_TERMS = [
    "hypertension",
    "hypothyroidism",
    "hypercholesterolemia",
    "hyponatremia",
    "siadh",
    "diverticulosis",
    "stenosis",
    "cataracts",
    "rhinitis",
    "osteoporosis",
    "glaucoma",
    "eczema",
    "ulcers",
    "scoliosis",
    "hemorrhoids",
    "prolapse",
    "bleeding",
]

CLINICAL_NOISE = [
    "hx",    # history
    "sp",    # status post
    "s/p",
    "tahbso",
]

# ------------------------------------------------------
# Utility: text cleaning for PMH
# ------------------------------------------------------


def clean_pmh_text(raw_text: str) -> str:
    """
    PRE-NER preprocessing of PMH:
    - lowercasing
    - remove some clinical shorthand/noise tokens
    - split glued disease names using GLUED_SPLIT_TERMS
    - normalize whitespace
    """
    if not isinstance(raw_text, str):
        return ""

    text = raw_text.lower()

    # remove obvious clinical noise tokens
    for noise in CLINICAL_NOISE:
        text = re.sub(rf"\b{re.escape(noise)}\b", " ", text)

    # split glued disease names
    for term in GLUED_SPLIT_TERMS:
        text = re.sub(re.escape(term), f" {term} ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def fuzzy_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def dedup_terms_fuzzy(terms, threshold: float = 0.85):
    """
    Deduplicate a list of terms using fuzzy string similarity.
    Keeps order of first occurrences; drops later ones that are too similar.
    """
    kept = []
    for t in terms:
        drop = False
        for k in kept:
            if fuzzy_sim(t, k) >= threshold:
                drop = True
                break
        if not drop:
            kept.append(t)
    return kept


def extract_entity_terms_from_pmh(text: str):
    """
    Run HF-clinical NER (via medical_ner.run_medical_ner) on cleaned PMH
    and return a deduplicated list of entity phrases (fuzzy dedup, option 3).
    """
    cleaned = clean_pmh_text(text)
    if not cleaned:
        return []

    ner_entities = run_medical_ner(cleaned)
    raw_terms = []
    for ent in ner_entities:
        phrase = (ent.get("entity") or "").strip()
        if not phrase:
            continue
        norm = re.sub(r"\s+", " ", phrase.lower())
        if len(norm) < 3:
            continue
        if norm in STOPWORDS:
            continue
        raw_terms.append(norm)

    # fuzzy dedup INSIDE this visit
    uniq = dedup_terms_fuzzy(raw_terms, threshold=0.85)
    return uniq


# ------------------------------------------------------
# Encoders
# ------------------------------------------------------


def get_text_encoder():
    global _TEXT_TOKENIZER, _TEXT_MODEL, _TEXT_DIM
    if _TEXT_MODEL is None:
        print(f"[INFO] Loading text encoder: {TEXT_MODEL_NAME}")
        _TEXT_TOKENIZER = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        _TEXT_MODEL = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        _TEXT_MODEL.to(device)
        _TEXT_MODEL.eval()
        _TEXT_DIM = _TEXT_MODEL.config.hidden_size
        print(f"[INFO] Text encoder hidden size: {_TEXT_DIM}")
    return _TEXT_TOKENIZER, _TEXT_MODEL, _TEXT_DIM


def encode_text(text: str, max_len: int = 128) -> torch.Tensor:
    tok, model, dim = get_text_encoder()
    if not isinstance(text, str) or not text.strip():
        return torch.zeros(dim, dtype=torch.float32, device=device)

    inputs = tok(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # [1, dim]
    return cls.squeeze(0)


def get_image_encoder():
    global _IMG_MODEL, _IMG_TRANSFORM, _IMG_DIM
    if _IMG_MODEL is None:
        print("[INFO] Loading image encoder: EfficientNet-B3 (ImageNet)")
        _IMG_TRANSFORM = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        model = efficientnet_b3(weights=weights)
        # Replace classifier with identity to get feature vector
        if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Sequential):
            model.classifier[-1] = torch.nn.Identity()
        else:
            model.classifier = torch.nn.Identity()

        model.to(device)
        model.eval()
        _IMG_MODEL = model

        # infer feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, _IMG_SIZE, _IMG_SIZE, device=device)
            feat = _IMG_MODEL(dummy)
            _IMG_DIM = int(feat.shape[1])
        print(f"[INFO] Image encoder feature dim: {_IMG_DIM}")
    return _IMG_MODEL, _IMG_TRANSFORM, _IMG_DIM


def encode_image(path: str) -> torch.Tensor:
    model, tfm, dim = get_image_encoder()
    if not path or not os.path.exists(path):
        return torch.zeros(dim, dtype=torch.float32, device=device)

    try:
        img = Image.open(path).convert("L")  # chest X-ray is grayscale
        img = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img)  # [1, dim]
        return feat.squeeze(0)
    except Exception as e:
        print(f"[WARN] Could not load image '{path}': {e}")
        return torch.zeros(dim, dtype=torch.float32, device=device)


def encode_timeseries(path: str) -> torch.Tensor:
    """
    Use ALL columns of the *_timeseries.csv file.

    - Numeric columns: for each, compute [mean, std, min, max].
    - Non-numeric columns: treat as categorical/text;
      build a combined string of "col:value" tokens and encode with clinical BERT.

    Final ts embedding = concat[numeric_stats || categorical_text_embedding].
    """
    global _TS_NUMERIC_COLS, _TS_CATEGORICAL_COLS, _TS_DIM

    if not path or not os.path.exists(path):
        if _TS_DIM is None:
            return torch.zeros(0, dtype=torch.float32, device=device)
        return torch.zeros(_TS_DIM, dtype=torch.float32, device=device)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read timeseries '{path}': {e}")
        if _TS_DIM is None:
            return torch.zeros(0, dtype=torch.float32, device=device)
        return torch.zeros(_TS_DIM, dtype=torch.float32, device=device)

    if _TS_NUMERIC_COLS is None:
        # detect numeric and non-numeric cols once (global schema)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.sort()
        cat_cols = [c for c in df.columns if c not in numeric_cols]
        cat_cols.sort()

        _TS_NUMERIC_COLS = numeric_cols
        _TS_CATEGORICAL_COLS = cat_cols

        # one extra TEXT_DIM for categorical summary
        _, _, text_dim = get_text_encoder()
        _TS_DIM = len(numeric_cols) * 4 + text_dim

        print(f"[INFO] Timeseries numeric columns ({len(numeric_cols)}): {_TS_NUMERIC_COLS}")
        print(f"[INFO] Timeseries categorical columns ({len(cat_cols)}): {_TS_CATEGORICAL_COLS}")
        print(
            f"[INFO] Timeseries feature dim: 4 × {len(numeric_cols)} + {text_dim} = {_TS_DIM}"
        )

    # ---- numeric part ----
    df_num = df.reindex(columns=_TS_NUMERIC_COLS)
    numeric_feats = []
    for col in _TS_NUMERIC_COLS:
        vals = pd.to_numeric(df_num[col], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            numeric_feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            mean = float(vals.mean())
            std = float(vals.std()) if vals.size > 1 else 0.0
            min_v = float(vals.min())
            max_v = float(vals.max())
            numeric_feats.extend([mean, std, min_v, max_v])

    numeric_feats = torch.tensor(numeric_feats, dtype=torch.float32, device=device)

    # ---- categorical/text part ----
    tokens = []
    for col in _TS_CATEGORICAL_COLS or []:
        if col not in df.columns:
            continue
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v.strip() for v in vals if v.strip()]
        if not vals:
            continue
        # example token: "event:NA", "event:MECH"
        for v in sorted(vals):
            tokens.append(f"{col}:{v}")

    if tokens:
        cat_text = " ".join(tokens)
        cat_vec = encode_text(cat_text, max_len=128)
    else:
        # no categorical info present
        _, _, text_dim = get_text_encoder()
        cat_vec = torch.zeros(text_dim, dtype=torch.float32, device=device)

    ts_vec = torch.cat([numeric_feats, cat_vec], dim=-1)
    return ts_vec


# ------------------------------------------------------
# Other helpers
# ------------------------------------------------------


def detect_label_columns(df: pd.DataFrame, required_cols):
    """Heuristically find 0/1 label columns, excluding required_cols."""
    labels = []
    for c in df.columns:
        if c in required_cols:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        vals = vals.dropna()
        if vals.empty:
            continue
        try:
            uniq = set(vals.astype(int).unique())
        except Exception:
            continue
        if uniq.issubset({0, 1}):
            labels.append(c)
    return labels


def find_file_recursive(root: Path, target_name: str):
    """
    Search for a file named `target_name` under `root` (recursively).
    Returns full path string if found, else empty string.
    """
    if not root or not target_name:
        return ""
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return ""

    p = Path(target_name)
    if p.is_absolute() and p.exists():
        return str(p)

    basename = p.name
    try:
        for f in root.rglob("*"):
            try:
                if f.is_file() and f.name == basename:
                    return str(f)
            except Exception:
                continue
    except Exception:
        return ""
    return ""


# ------------------------------------------------------
# Main graph building
# ------------------------------------------------------


def build_graphs(input_path: Path, outdir: Path, img_root: Path = None, ts_root: Path = None):
    # read tabular input
    if input_path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(input_path, sheet_name=0)
    else:
        df = pd.read_csv(input_path)

    required = [
        "subject_id",
        "stay_id",
        "time_series",
        "period_length",
        "dicom_id",
        "past_medical_history",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    label_cols = detect_label_columns(df, required)
    print(
        f"[INFO] Using explicit EHR label columns ({len(label_cols)}-task): {len(label_cols)} found."
    )
    print("Using labels ({}): {}".format(len(label_cols), label_cols))

    outdir.mkdir(parents=True, exist_ok=True)

    rows_meta = []
    total_resolved_images = 0
    total_resolved_ts = 0

    print(f"[INFO] NER backend in use: {NER_BACKEND}")

    # group by patient
    for subject, df_p in df.groupby("subject_id"):
        try:
            print(f"[INFO] Building graph for subject: {subject}")

            # sort within patient by numeric stay_id only
            df_p = df_p.copy()
            df_p["_sid"] = pd.to_numeric(df_p["stay_id"], errors="coerce")
            df_p = df_p.sort_values(["_sid", "stay_id"])

            # merge rows with same stay_id into one visit
            merged_rows = []
            for stay_id, g_visit in df_p.groupby("stay_id", sort=False):
                row = {"subject_id": subject, "stay_id": stay_id}

                # take first non-empty time_series / dicom_id
                ts_vals = g_visit["time_series"].dropna().astype(str)
                row["time_series"] = ts_vals.iloc[0] if len(ts_vals) > 0 else ""

                img_vals = g_visit["dicom_id"].dropna().astype(str)
                row["dicom_id"] = img_vals.iloc[0] if len(img_vals) > 0 else ""

                # concatenate PMH texts
                pmh_vals = g_visit["past_medical_history"].dropna().astype(str)
                if len(pmh_vals) > 0:
                    row["past_medical_history"] = " ".join(v for v in pmh_vals if v.strip())
                else:
                    row["past_medical_history"] = ""

                # OR labels across duplicates (max over rows)
                for c in label_cols:
                    vals = pd.to_numeric(g_visit[c], errors="coerce")
                    vals = vals.dropna()
                    row[c] = int(vals.max()) if not vals.empty else 0

                merged_rows.append(row)

            sub = pd.DataFrame(merged_rows)
            # sort visits by numeric stay_id
            sub["_sid"] = pd.to_numeric(sub["stay_id"], errors="coerce")
            sub = sub.sort_values("_sid").reset_index(drop=True)
            n_vis = len(sub)

            data = HeteroData()

            # labels per visit: [n_vis, n_labels]
            if n_vis > 0 and label_cols:
                visit_y = np.array(
                    [
                        [int(row[c]) if c in row and not pd.isna(row[c]) else 0 for c in label_cols]
                        for _, row in sub.iterrows()
                    ],
                    dtype=np.float32,
                )
            else:
                visit_y = np.zeros((n_vis, 0), dtype=np.float32)

            img_paths = []
            ts_paths = []
            raw_texts = []
            visit_feats = []

            entity_terms_all = []

            for i, row in sub.iterrows():
                # -------- resolve paths --------
                img_path = ""
                ts_path = ""

                if img_root is not None:
                    img_path = find_file_recursive(
                        img_root, str(row.get("dicom_id", "")).strip()
                    )
                    if img_path:
                        total_resolved_images += 1

                if ts_root is not None:
                    ts_path = find_file_recursive(
                        ts_root, str(row.get("time_series", "")).strip()
                    )
                    if ts_path:
                        total_resolved_ts += 1

                img_paths.append(img_path)
                ts_paths.append(ts_path)

                raw_text = str(row.get("past_medical_history", "") or "").strip()
                raw_texts.append(raw_text)

                # -------- per-modality embeddings --------
                img_vec = encode_image(img_path)
                ts_vec = encode_timeseries(ts_path)
                text_vec = encode_text(raw_text)

                # concatenated visit embedding
                visit_vec = torch.cat([img_vec, ts_vec, text_vec], dim=-1)
                visit_feats.append(visit_vec)

                # -------- NER entities for this visit --------
                terms = extract_entity_terms_from_pmh(raw_text)
                for t in terms:
                    entity_terms_all.append({"visit_idx": i, "term": t})

            # stack visit features -> [n_vis, D_visit]
            if visit_feats:
                visit_x = torch.stack(visit_feats, dim=0)
            else:
                visit_x = torch.zeros((0, 0), dtype=torch.float32)

            data["visit"].x = visit_x.to(torch.float32)
            data["visit"].y = torch.tensor(visit_y, dtype=torch.float32)
            data["visit"].img_path = img_paths
            data["visit"].ts_path = ts_paths
            data["visit"].text_raw = raw_texts
            data["visit"].label_cols = label_cols

            # -------- entity nodes + edges --------
            if entity_terms_all:
                ent_terms = [e["term"] for e in entity_terms_all]

                # embed each entity term with same clinical BERT encoder
                ent_embeddings = []
                for term in ent_terms:
                    ent_embeddings.append(encode_text(term, max_len=16))
                ent_x = torch.stack(ent_embeddings, dim=0)

                data["entity"].x = ent_x.to(torch.float32)
                data["entity"].terms = ent_terms

                src = [e["visit_idx"] for e in entity_terms_all]
                dst = list(range(len(entity_terms_all)))
                data[("visit", "to", "entity")].edge_index = torch.tensor(
                    [src, dst], dtype=torch.long
                )
                data[("entity", "to", "visit")].edge_index = torch.tensor(
                    [dst, src], dtype=torch.long
                )
            else:
                # fallback: zero-dim entities
                _, _, text_dim = get_text_encoder()
                data["entity"].x = torch.zeros((0, text_dim), dtype=torch.float32)
                data[("visit", "to", "entity")].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long
                )
                data[("entity", "to", "visit")].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long
                )

            # -------- temporal visit edges (sorted by stay_id) --------
            if n_vis > 1:
                src = list(range(n_vis - 1))
                dst = list(range(1, n_vis))
                data[("visit", "to", "visit")].edge_index = torch.tensor(
                    [src, dst], dtype=torch.long
                )
                data[("visit", "to", "visit_rev")].edge_index = torch.tensor(
                    [dst, src], dtype=torch.long
                )
            else:
                data[("visit", "to", "visit")].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long
                )
                data[("visit", "to", "visit_rev")].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long
                )

            # -------- meta + save --------
            data.meta = {
                "patient_id": str(subject),
                "label_cols": label_cols,
                "source": str(input_path),
                "ner_backend": NER_BACKEND,
            }

            out_path = outdir / f"patient_{subject}.pt"
            torch.save(data, out_path)
            print(f"[OK] wrote {out_path}")

            rows_meta.append(
                {
                    "patient_id": subject,
                    "num_visits": n_vis,
                    "num_entities": len(entity_terms_all),
                    "path": str(out_path),
                }
            )

        except Exception as e:
            msg = f"ERROR building graph for subject {subject}: {repr(e)}"
            print(msg)
            with open(outdir / "errors.log", "a") as ef:
                ef.write(msg + "\n")
            continue

    # -------- index + stats --------
    idx_df = pd.DataFrame(rows_meta)
    idx_df.to_csv(outdir / "index.csv", index=False)

    # if at least one patient succeeded, `data` is the last graph built
    visit_emb_dim = int(data["visit"].x.shape[1]) if len(idx_df) > 0 else 0
    _, _, text_dim_final = get_text_encoder()

    stats = {
        "num_rows": int(len(df)),
        "num_patients": int(len(idx_df)),
        "resolved_images": int(total_resolved_images),
        "resolved_timeseries": int(total_resolved_ts),
        "label_cols": label_cols,
        "ner_backend": NER_BACKEND,
        "visit_emb_dim": visit_emb_dim,
        "entity_emb_dim": int(text_dim_final),
    }
    with open(outdir / "stats.json", "w") as sf:
        json.dump(stats, sf, indent=2)

    print(f"Wrote {len(idx_df)} patient graphs to {outdir}")
    print("Stats:", stats)


# ------------------------------------------------------
# CLI
# ------------------------------------------------------


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV or Excel input file")
    ap.add_argument("--outdir", required=True, help="output directory for .pt graphs")
    ap.add_argument("--img_root", default=None, help="root folder for images")
    ap.add_argument("--ts_root", default=None, help="root folder for timeseries CSVs")
    args = ap.parse_args()

    build_graphs(
        Path(args.input),
        Path(args.outdir),
        Path(args.img_root) if args.img_root else None,
        Path(args.ts_root) if args.ts_root else None,
    )
