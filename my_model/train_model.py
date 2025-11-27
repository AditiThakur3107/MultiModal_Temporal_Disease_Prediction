#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import random

from torch_geometric.nn import HeteroConv, GATConv

# ---- sklearn for AUROC / AUPRC ----
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _SKLEARN_OK = True
except ImportError:
    print("[WARN] sklearn not found; AUROC/AUPRC will be NaN.")
    _SKLEARN_OK = False


# -------------------- Model -------------------- #

class HeteroVisitGNN(nn.Module):
    def __init__(self, visit_in_dim, entity_in_dim, num_labels,
                 hidden_dim=64, dropout=0.3):
        super().__init__()

        self.visit_in_dim = visit_in_dim
        self.entity_in_dim = entity_in_dim
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        # Project raw embeddings to a shared hidden space
        self.lin_visit_in = nn.Linear(visit_in_dim, hidden_dim)
        self.lin_entity_in = nn.Linear(entity_in_dim, hidden_dim)

        # HeteroConv: message passing across relations
        self.conv1 = HeteroConv(
            {
                ("visit", "to", "visit"): GATConv(
                    hidden_dim, hidden_dim,
                    heads=1, concat=False, add_self_loops=False
                ),
                ("visit", "to", "visit_rev"): GATConv(
                    hidden_dim, hidden_dim,
                    heads=1, concat=False, add_self_loops=False
                ),
                ("visit", "to", "entity"): GATConv(
                    hidden_dim, hidden_dim,
                    heads=1, concat=False, add_self_loops=False
                ),
                ("entity", "to", "visit"): GATConv(
                    hidden_dim, hidden_dim,
                    heads=1, concat=False, add_self_loops=False
                ),
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                ("visit", "to", "visit"): GATConv(
                    hidden_dim, hidden_dim,
                    heads=1, concat=False, add_self_loops=False
                ),
                ("visit", "to", "visit_rev"): GATConv(
                    hidden_dim, hidden_dim,
                    heads=1, concat=False, add_self_loops=False
                ),
                ("visit", "to", "entity"): GATConv(
                    hidden_dim, hidden_dim,
                    heads=1, concat=False, add_self_loops=False
                ),
                ("entity", "to", "visit"): GATConv(
                    hidden_dim, hidden_dim,
                    heads=1, concat=False, add_self_loops=False
                ),
            },
            aggr="sum",
        )

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        # Final classifier on visit nodes
        self.lin_out = nn.Linear(hidden_dim, num_labels)

    def forward(self, x_dict, edge_index_dict):
        # Input projection
        x_visit = self.lin_visit_in(x_dict["visit"])
        x_entity = self.lin_entity_in(x_dict["entity"])

        x_dict = {"visit": x_visit, "entity": x_entity}

        # Conv layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.act(self.dropout(v)) for k, v in x_dict.items()}

        # Conv layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: self.act(self.dropout(v)) for k, v in x_dict.items()}

        # Only visit nodes produce predictions
        logits = self.lin_out(x_dict["visit"])  # [num_visits, num_labels]
        return logits


# -------------------- Utils -------------------- #

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight_from_paths(train_paths, device):
    """
    Compute per-label pos_weight using the raw graph files (no DataLoader).
    pos_weight[j] = (N_neg / N_pos) for label j.
    """
    ys = []

    for path in train_paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        y = data["visit"].y  # [n_visits_for_this_patient, num_labels]
        if y.numel() == 0:
            continue
        ys.append(y)

    if not ys:
        raise RuntimeError("No labels found in training graphs for pos_weight computation.")

    all_y = torch.cat(ys, dim=0)  # [N_visits_total, num_labels]

    num_pos = all_y.sum(dim=0)           # [num_labels]
    num_total = all_y.shape[0]
    num_neg = num_total - num_pos

    pos_weight = num_neg / (num_pos + 1e-6)
    return pos_weight.to(device)


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


def train_one_epoch(model, train_paths, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_true = []
    all_pred = []

    for path in train_paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        data = data.to(device)

        logits = model(data.x_dict, data.edge_index_dict)
        y_true = data["visit"].y.to(device)

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
def eval_model(model, val_paths, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []

    for path in val_paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        data = data.to(device)

        logits = model(data.x_dict, data.edge_index_dict)
        y_true = data["visit"].y.to(device)

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


# -------------------- Main -------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graphs_dir",
        type=str,
        default="./input_graphs_neural_ode",
        help="Folder containing patient_*.pt graphs",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    graphs_dir = Path(args.graphs_dir)
    assert graphs_dir.exists(), f"{graphs_dir} does not exist"

    # Collect all patient graph files
    graph_paths = sorted(
        str(p) for p in graphs_dir.glob("patient_*.pt") if p.is_file()
    )
    print(f"[INFO] Found {len(graph_paths)} patient graphs")

    if len(graph_paths) < 5:
        print("[WARN] Very few graphs found; training may be unstable.")

    # Shuffle and split into train/val by patient
    random.shuffle(graph_paths)
    n_total = len(graph_paths)
    n_train = int(0.8 * n_total)
    train_paths = graph_paths[:n_train]
    val_paths = graph_paths[n_train:]

    print(f"[INFO] Train patients: {len(train_paths)}, Val patients: {len(val_paths)}")

    # Peek one graph to get dimensions
    sample_data = torch.load(train_paths[0], map_location="cpu", weights_only=False)
    visit_in_dim = sample_data["visit"].x.shape[1]
    entity_in_dim = sample_data["entity"].x.shape[1]
    num_labels = sample_data["visit"].y.shape[1]

    print(f"[INFO] visit_in_dim  = {visit_in_dim}")
    print(f"[INFO] entity_in_dim = {entity_in_dim}")
    print(f"[INFO] num_labels    = {num_labels}")

    model = HeteroVisitGNN(
        visit_in_dim=visit_in_dim,
        entity_in_dim=entity_in_dim,
        num_labels=num_labels,
        hidden_dim=args.hidden_dim,
        dropout=0.3,
    ).to(device)

    # Compute pos_weight directly from graph files
    pos_weight = compute_pos_weight_from_paths(train_paths, device)
    print("[INFO] pos_weight:", pos_weight)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mi_f1, train_ma_f1 = train_one_epoch(
            model, train_paths, optimizer, criterion, device
        )
        (
            val_loss,
            val_mi_f1,
            val_ma_f1,
            auroc_mi,
            auroc_ma,
            auprc_mi,   # still computed but not printed
            auprc_ma,
        ) = eval_model(model, val_paths, criterion, device)

        # pretty formatting like your baseline
        if isinstance(auroc_mi, float) and np.isnan(auroc_mi):
            s_auroc_mi = "nan"
        else:
            s_auroc_mi = f"{auroc_mi:.4f}"

        if isinstance(auroc_ma, float) and np.isnan(auroc_ma):
            s_auroc_ma = "nan"
        else:
            s_auroc_ma = f"{auroc_ma:.4f}"

        print(f"Epoch {epoch:03d}")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val   Loss : {val_loss:.4f}")
        print(f"  Micro F1   : {val_mi_f1:.4f}")
        print(f"  Macro F1   : {val_ma_f1:.4f}")
        print(f"  AUC (micro): {s_auroc_mi}")
        print(f"  AUC (macro): {s_auroc_ma}")

        # Early stopping based on validation micro-F1
        if val_mi_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_mi_f1
            best_state = model.state_dict()
            patience_counter = 0
            print(
                f"-> Improvement! Best Micro F1: {best_val_f1:.4f}. "
                f"Patience reset to 0."
            )
        else:
            patience_counter += 1
            print(
                f"-> No improvement for {patience_counter} epoch(s). "
                f"Patience={args.patience}"
            )
            if patience_counter >= args.patience:
                print(
                    f"Early stopping triggered (no improvement for {args.patience} epochs)."
                )
                break

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)
    save_dir = Path(__file__).parent
    out_path = save_dir / "gnn_model_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "visit_in_dim": visit_in_dim,
            "entity_in_dim": entity_in_dim,
            "num_labels": num_labels,
            "hidden_dim": args.hidden_dim,
        },
        out_path,
    )
    print(f"[INFO] Saved best model to: {out_path}")


if __name__ == "__main__":
    main()
