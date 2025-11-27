#!/usr/bin/env python3
import argparse
import math
import torch

from transformers import AutoTokenizer, AutoModel

# ---------------- TEXT ENCODER (for labels) ----------------

TEXT_MODEL_NAME = "samrawal/bert-base-uncased_clinical-ner"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TOKENIZER = None
_TEXT_MODEL = None
_TEXT_DIM = None


def get_text_encoder():
    global _TOKENIZER, _TEXT_MODEL, _TEXT_DIM
    if _TEXT_MODEL is None:
        print(f"[INFO] Loading label text encoder: {TEXT_MODEL_NAME}")
        _TOKENIZER = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        _TEXT_MODEL = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        _TEXT_MODEL.to(_DEVICE)
        _TEXT_MODEL.eval()
        _TEXT_DIM = _TEXT_MODEL.config.hidden_size
        print(f"[INFO] Label text encoder hidden size: {_TEXT_DIM}")
    return _TOKENIZER, _TEXT_MODEL, _TEXT_DIM


def encode_text(text: str, max_len: int = 128) -> torch.Tensor:
    tok, model, dim = get_text_encoder()
    if not isinstance(text, str) or not text.strip():
        return torch.zeros(dim, dtype=torch.float32, device=_DEVICE)

    inputs = tok(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # [1, dim]
    return cls.squeeze(0)


# ---------------- COSINE + SOFTMAX UTILS ----------------


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1D tensors."""
    a_norm = a.norm(p=2).item()
    b_norm = b.norm(p=2).item()
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / (a_norm * b_norm))


def softmax(xs):
    """Softmax over a list of floats (for importance weights)."""
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s == 0:
        return [0.0 for _ in xs]
    return [e / s for e in exps]


# ---------------- MAIN SCRIPT ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        required=True,
        help="Path to a patient .pt file (e.g., patient_10019172.pt)",
    )
    args = ap.parse_args()
    file_path = args.path

    # ---- load graph ----
    data = torch.load(file_path, map_location="cpu", weights_only=False)

    print("\n=== BASIC INFO ===")
    print("File:", file_path)
    print("Type of object:", type(data))
    print("Node types:", data.node_types)
    print("Edge types:", data.edge_types)

    # ---- meta ----
    if hasattr(data, "meta"):
        print("\n=== META ===")
        for k, v in data.meta.items():
            print(f"{k}: {v}")

    if "visit" not in data.node_types or "entity" not in data.node_types:
        print("\n[WARN] Graph does not contain both 'visit' and 'entity' node types.")
        return

    visit = data["visit"]
    entity = data["entity"]

    visit_x = visit.x  # [num_visits, dv]
    entity_x = entity.x  # [num_entities, de]
    num_visits = visit_x.size(0)
    num_entities = entity_x.size(0)

    label_cols = getattr(visit, "label_cols", [])
    visit_y = getattr(visit, "y", None)
    text_raw = getattr(visit, "text_raw", [])
    img_paths = getattr(visit, "img_path", [])
    ts_paths = getattr(visit, "ts_path", [])
    ent_terms = getattr(entity, "terms", [])

    print("\n=== SHAPES ===")
    print("visit.x shape:", visit_x.shape)
    print("entity.x shape:", entity_x.shape)
    print("num_visits:", num_visits)
    print("num_entities:", num_entities)

    # ---- build visit -> entity adjacency ----
    visit_to_entity = {i: [] for i in range(num_visits)}
    if ("visit", "to", "entity") in data.edge_types:
        edge_index = data[("visit", "to", "entity")].edge_index
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            if 0 <= s < num_visits and 0 <= d < num_entities:
                visit_to_entity[s].append(d)

    # entity embedding dim (should match text part of visit emb)
    ent_dim = entity_x.size(1) if entity_x.dim() == 2 and num_entities > 0 else 0

    # make sure text encoder is initialised (we'll need TEXT_DIM)
    _, _, label_text_dim = get_text_encoder()

    print("\n================ VISIT-WISE DETAIL ================\n")

    for vid in range(num_visits):
        print(f"---------------- VISIT {vid} ----------------")

        # ---- original visit "details" ----
        # raw PMH
        print("\n[PMH RAW TEXT]")
        if text_raw and vid < len(text_raw):
            print(text_raw[vid])
        else:
            print("(none stored)")

        # image / timeseries paths
        if img_paths and vid < len(img_paths):
            print("\n[IMAGE PATH]")
            print(img_paths[vid])
        if ts_paths and vid < len(ts_paths):
            print("\n[TIMESERIES PATH]")
            print(ts_paths[vid])

        # labels for this visit
        label_text = None
        active_label_idxs = []
        if visit_y is not None and visit_y.numel() > 0 and label_cols:
            print("\n[VISIT LABELS == 1]")
            row = visit_y[vid]
            active_label_idxs = [i for i, v in enumerate(row.tolist()) if v == 1.0]
            if not active_label_idxs:
                print("  (no labels == 1)")
            else:
                active_names = []
                for idx in active_label_idxs:
                    if 0 <= idx < len(label_cols):
                        name = label_cols[idx]
                        active_names.append(name)
                        print(f"  - {name}")
                if active_names:
                    # build one string from active label names
                    label_text = " ; ".join(active_names)
        else:
            print("\n[VISIT LABELS]")
            print("  (no label info stored)")

        # ---- previous graph entities (just names) ----
        ents_for_visit = visit_to_entity.get(vid, [])
        print("\n[CONNECTED ENTITY TERMS (UNRANKED)]")
        if not ents_for_visit:
            print("  (no entities connected)")
        else:
            for eid in ents_for_visit:
                name = ent_terms[eid] if eid < len(ent_terms) else "UNKNOWN"
                print(f"  Entity {eid}: {name}")

        # ---- entity importance via similarity ----
        print("\n[ENTITY IMPORTANCE (RANKED)]")
        if not ents_for_visit or ent_dim == 0 or visit_x.size(1) == 0:
            print("  (cannot compute importance; missing embeddings or entities)")
            print()
            continue

        # visit text slice (same as before)
        v_emb_full = visit_x[vid]
        if v_emb_full.size(0) >= ent_dim:
            v_emb = v_emb_full[:ent_dim]
        else:
            v_emb = v_emb_full  # fallback

        # label embedding (if there are active labels)
        if label_text is not None and label_text_dim == ent_dim:
            label_vec = encode_text(label_text, max_len=128).cpu()
        elif label_text is not None:
            # dims mismatch; still compute label_vec but only use if sizes match
            label_vec = encode_text(label_text, max_len=128).cpu()
        else:
            label_vec = None

        v_emb_cpu = v_emb.cpu()

        scores = []
        for eid in ents_for_visit:
            e_emb = entity_x[eid].cpu()

            # similarity to visit text
            s_visit = cosine_sim(v_emb_cpu, e_emb)

            if label_vec is not None and label_vec.size(0) == e_emb.size(0):
                s_label = cosine_sim(label_vec, e_emb)
                # combine visit-text + label-text
                score = 0.5 * s_visit + 0.5 * s_label
            else:
                # fallback: only visit-text similarity
                score = s_visit

            scores.append((eid, score))

        if not scores:
            print("  (no importance scores)")
            print()
            continue

        # softmax for importance weights
        raw_scores = [s for _, s in scores]
        weights = softmax(raw_scores)

        # sort by score desc
        scored = sorted(
            zip([eid for eid, _ in scores], raw_scores, weights),
            key=lambda t: t[1],
            reverse=True,
        )

        for eid, sc, wt in scored:
            name = ent_terms[eid] if eid < len(ent_terms) else "UNKNOWN"
            print(f"  Entity {eid:3d}:  score={sc: .4f}  weight={wt: .4f}  term='{name}'")

        print()  # blank line between visits


if __name__ == "__main__":
    main()
