import torch
from pathlib import Path
from train_model import (
    HeteroVisitGNN,
    eval_model,
)
import argparse
import numpy as np


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_dir", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    graphs_dir = Path(args.graphs_dir)
    model_path = Path(args.model_path)

    print(f"[INFO] Loading model from: {model_path}")

    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    visit_in_dim = ckpt["visit_in_dim"]
    entity_in_dim = ckpt["entity_in_dim"]
    num_labels = ckpt["num_labels"]
    hidden_dim = ckpt["hidden_dim"]

    # Build model
    model = HeteroVisitGNN(
        visit_in_dim, entity_in_dim, num_labels, hidden_dim
    )

    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Collect graph files
    graph_paths = sorted(
        list(str(p) for p in graphs_dir.glob("patient_*.pt"))
    )
    print(f"[INFO] Found {len(graph_paths)} patient graphs")

    # Evaluate on ALL graphs (whole dataset)
    print("[INFO] Running evaluation...")

    (
        val_loss,
        val_mi_f1,
        val_ma_f1,
        auroc_mi,
        auroc_ma,
        auprc_mi,
        auprc_ma,
    ) = eval_model(model, graph_paths, torch.nn.BCEWithLogitsLoss(), device)

    print("\n===== FINAL METRICS =====")
    print(f"Loss        : {val_loss:.4f}")
    print(f"Micro F1    : {val_mi_f1:.4f}")
    print(f"Macro F1    : {val_ma_f1:.4f}")
    print(f"AUC (micro) : {auroc_mi}")
    print(f"AUC (macro) : {auroc_ma}")
    print(f"AUPRC (micro): {auprc_mi}")
    print(f"AUPRC (macro): {auprc_ma}")
    print("=========================")


if __name__ == "__main__":
    main()
