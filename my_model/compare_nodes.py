#!/usr/bin/env python3
import torch
from torch_geometric.data import HeteroData

# ==== CONFIG ====
PATIENT_ID = "10014354"  # <-- change this to any patient you want
ORIG_DIR = "input_graphs_attn"        # graphs before neural ODE
ODE_DIR  = "input_graphs_neural_ode"  # graphs after neural ODE
# ================





def load_graph(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise TypeError(f"{path} is not a HeteroData object.")
    return data


def main():
    orig_path = f"{ORIG_DIR}/patient_{PATIENT_ID}.pt"
    ode_path  = f"{ODE_DIR}/patient_{PATIENT_ID}.pt"

    print(f"[INFO] Loading original graph: {orig_path}")
    data_orig = load_graph(orig_path)

    print(f"[INFO] Loading ODE-updated graph: {ode_path}")
    data_ode = load_graph(ode_path)

    x_orig = data_orig["visit"].x.float()
    x_ode  = data_ode["visit"].x.float()

    print("\n=== BASIC SHAPES ===")
    print("Original visit.x shape:", x_orig.shape)
    print("ODE      visit.x shape:", x_ode.shape)

    if x_orig.shape != x_ode.shape:
        print("\n[WARN] Shapes differ, comparison may be invalid.")
        return

    n_visits, dim = x_orig.shape
    print(f"\nNumber of visits: {n_visits}, embedding dim: {dim}")

    # Per-visit norms and differences
    print("\n=== PER-VISIT NORM & CHANGE ===")
    for i in range(n_visits):
        v0 = x_orig[i]
        v1 = x_ode[i]

        norm0 = v0.norm().item()
        norm1 = v1.norm().item()
        diff_norm = (v1 - v0).norm().item()

        print(f"Visit {i}:")
        print(f"  ||v_before|| = {norm0:.4f}")
        print(f"  ||v_after || = {norm1:.4f}")
        print(f"  ||Δv||      = {diff_norm:.4f}")

    # Optional: overall change summary
    total_change = (x_ode - x_orig).norm().item()
    avg_change = total_change / max(1, n_visits)
    print("\n=== OVERALL CHANGE ===")
    print(f"Total ||X_after - X_before||_F = {total_change:.4f}")
    print(f"Avg per-visit change           = {avg_change:.4f}")


if __name__ == "__main__":
    main()
