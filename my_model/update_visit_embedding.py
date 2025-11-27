#!/usr/bin/env python3
"""
Update visit embeddings using connected entity importance + patient demographics.

- Reads patient graphs (patient_*.pt) from --input_dir
- Reads patient metadata (patient.csv) from --patient_csv (must have subject_id, anchor_age, gender)
- For each visit, computes weights over entities connected to that visit:
    score = dot(normalize(visit_vec), normalize(entity_vec_proj))
    score_adj = score * (1 + age_scale * age_z) * (1 + gender_multiplier)
    weights = softmax(score_adj)
  entity_summary = sum(weights * entity_vec_proj)
  updated_visit = visit + alpha * entity_summary   (updates the full visit embedding)
- Saves updated graphs to --outdir
- Adds per-visit importance info into data.meta['entity_importance'] as a list of lists:
    entity_importance[visit_idx] = list of (entity_idx, weight, entity_term) sorted desc

Notes:
- Entity vectors are deterministically padded/truncated to match visit embedding size
  (no random weights).
- This script does NOT use labels for importance.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

# -----------------------
# Helper utilities
# -----------------------

def load_patient_demographics(patient_csv_path: str) -> Dict[str, Dict]:
    """
    Load patient.csv and return mapping from subject_id (string) to dict with keys:
      - 'age' (float or None)
      - 'gender' (string lowercased) e.g. 'male', 'female', 'other', or ''
    """
    df = pd.read_csv(patient_csv_path)
    # Ensure columns exist
    req = {"subject_id", "anchor_age", "gender"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"patient.csv must contain columns: {req}, got: {df.columns.tolist()}")
    mapping = {}
    for _, r in df.iterrows():
        sid = str(r["subject_id"])
        try:
            age = float(r["anchor_age"]) if not pd.isna(r["anchor_age"]) else None
        except Exception:
            age = None
        gender = str(r["gender"]).strip().lower() if not pd.isna(r["gender"]) else ""
        mapping[sid] = {"age": age, "gender": gender}
    return mapping


def collect_age_stats(demo_map: Dict[str, Dict]) -> Tuple[float, float]:
    ages = [v["age"] for v in demo_map.values() if v["age"] is not None]
    if not ages:
        return 0.0, 1.0
    a = float(np.mean(ages))
    s = float(np.std(ages)) if len(ages) > 1 else 1.0
    if s == 0:
        s = 1.0
    return a, s


def gender_multiplier(gender_str: str) -> float:
    """Map gender string to a small multiplier. Tunable."""
    if not gender_str:
        return 0.0
    g = gender_str.lower()
    if g in ("female", "f"):
        return 0.03
    if g in ("male", "m"):
        return 0.02
    # other / unknown
    return 0.01


def pad_or_truncate(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Deterministic expansion/truncation:
      - if vec dim < target_dim: pad with zeros at the end
      - if vec dim > target_dim: truncate
    """
    cur = vec.shape[0]
    if cur == target_dim:
        return vec
    if cur < target_dim:
        pad = torch.zeros(target_dim - cur, dtype=vec.dtype, device=vec.device)
        return torch.cat([vec, pad], dim=0)
    else:
        return vec[:target_dim]


def normalize_vec(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = v.norm(p=2)
    if norm < eps:
        return v * 0.0
    return v / (norm + eps)


def softmax_scores(scores: torch.Tensor) -> torch.Tensor:
    # scores: [n_entities]
    if scores.numel() == 0:
        return scores
    return F.softmax(scores, dim=0)


# -----------------------
# Main update logic
# -----------------------

def process_graph_file(
    path_in: Path,
    path_out: Path,
    demo_map: Dict[str, Dict],
    age_mean: float,
    age_std: float,
    alpha: float = 0.5,
    device: torch.device = torch.device("cpu"),
):
    data: HeteroData = torch.load(str(path_in), map_location=device, weights_only=False)

    # metadata
    patient_id = str(data.meta.get("patient_id", "") or "")
    demographics = demo_map.get(patient_id, {"age": None, "gender": ""})
    age = demographics.get("age", None)
    gender = demographics.get("gender", "")

    # compute age z score (safe)
    if age is None:
        age_z = 0.0
    else:
        age_z = (float(age) - age_mean) / (age_std + 1e-6)

    gmult = gender_multiplier(gender)

    # node arrays
    if "visit" not in data.node_types:
        raise RuntimeError(f"No visit nodes in graph {path_in}")

    visit_x: torch.Tensor = data["visit"].x  # [n_vis, visit_dim]
    entity_x: torch.Tensor = data["entity"].x  # [n_ent, ent_dim]
    n_vis = 0 if visit_x is None else int(visit_x.shape[0])
    n_ent = 0 if entity_x is None else int(entity_x.shape[0])

    visit_dim = visit_x.shape[1] if n_vis > 0 else 0
    ent_dim = entity_x.shape[1] if n_ent > 0 else 0

    # Build adjacency mapping from ("visit","to","entity") edges to list of entity indices per visit
    ent_edge = data.get(("visit", "to", "entity"))
    visit2ents: Dict[int, List[int]] = {}
    if ent_edge is not None and hasattr(ent_edge, "edge_index"):
        edge_index = ent_edge.edge_index.cpu().numpy()
        if edge_index.shape[0] == 2 and edge_index.shape[1] > 0:
            srcs = edge_index[0].tolist()
            dsts = edge_index[1].tolist()
            for s, d in zip(srcs, dsts):
                visit2ents.setdefault(int(s), []).append(int(d))

    # Prepare holder for importance metadata
    per_visit_importance: List[List[Tuple[int, float, str]]] = []

    # Update each visit embedding
    updated_visit_x = visit_x.clone().to(device)

    # Pre-normalize visit vectors
    for vi in range(n_vis):
        v = visit_x[vi].to(device).float()
        v_norm = normalize_vec(v)

        ent_idxs = visit2ents.get(vi, [])
        if not ent_idxs:
            per_visit_importance.append([])
            continue

        scores = []
        ent_projs = []
        for eidx in ent_idxs:
            evec = entity_x[eidx].to(device).float()
            # deterministic projection: pad/truncate entity vec to visit dim
            eproj = pad_or_truncate(evec, visit_dim).to(device).float()
            ent_projs.append(eproj)
            # normalized dot-product (cosine-like)
            eproj_n = normalize_vec(eproj)
            score = float((v_norm * eproj_n).sum().item())  # scalar
            scores.append(score)

        scores_t = torch.tensor(scores, dtype=torch.float32, device=device)

        # demographic adjustment: multiplicative factor
        # small tuning constants (safe defaults)
        age_scale = 0.05  # how strongly age affects weight
        # age_z may be large; clamp to [-3, 3]
        age_z_clamped = max(min(age_z, 3.0), -3.0)
        demo_factor = (1.0 + age_scale * age_z_clamped) * (1.0 + gmult)

        scores_adj = scores_t * float(demo_factor)

        # softmax weights
        weights = softmax_scores(scores_adj)

        # weighted sum of entity projections
        ent_stack = torch.stack(ent_projs, dim=0)  # [k, visit_dim]
        # weights [k] -> [k,1]
        ent_summary = (weights.unsqueeze(1) * ent_stack).sum(dim=0)  # [visit_dim]

        # update visit embedding (additive)
        updated_visit_vec = v + alpha * ent_summary
        updated_visit_x[vi] = updated_visit_vec

        # store importance metadata sorted
        ent_terms = getattr(data["entity"], "terms", [str(i) for i in range(n_ent)])
        importance_list = []
        for idx_local, eidx in enumerate(ent_idxs):
            importance_list.append((int(eidx), float(weights[idx_local].item()), ent_terms[eidx] if eidx < len(ent_terms) else ""))
        # sort by weight desc
        importance_list.sort(key=lambda x: x[1], reverse=True)
        per_visit_importance.append(importance_list)

    # Replace visit.x with updated embeddings
    data["visit"].x = updated_visit_x

    # Save metadata - append or create
    meta = dict(data.meta) if hasattr(data, "meta") else {}
    meta["entity_importance_demog"] = per_visit_importance
    meta["demographics_used"] = {"age_mean": float(age_mean), "age_std": float(age_std)}
    data.meta = meta

    # Save to out path
    torch.save(data, str(path_out))


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory with patient_*.pt graphs (input)")
    ap.add_argument("--patient_csv", required=True, help="Path to patient.csv (must contain subject_id, anchor_age, gender)")
    ap.add_argument("--outdir", required=True, help="Where to write updated graphs")
    ap.add_argument("--alpha", type=float, default=0.5, help="Scale factor to add entity summary to visit embedding")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load patient demographics
    demo_map = load_patient_demographics(args.patient_csv)
    age_mean, age_std = collect_age_stats(demo_map)

    # find graph files
    files = sorted([p for p in input_dir.glob("patient_*.pt") if p.is_file()])

    if len(files) == 0:
        print(f"[WARN] No patient_*.pt files found in {input_dir}")
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[INFO] Running on device: {device}; found {len(files)} graphs; writing to {outdir}")
    print(f"[INFO] Age mean/std: {age_mean:.2f} / {age_std:.2f}")

    for f in files:
        try:
            outp = outdir / f.name
            print(f"[INFO] Processing {f.name}")
            process_graph_file(f, outp, demo_map, age_mean, age_std, alpha=args.alpha, device=device)
            print(f"[OK] Wrote {outp}")
        except Exception as e:
            print(f"[ERROR] Failed {f.name}: {repr(e)}")

if __name__ == "__main__":
    main()
