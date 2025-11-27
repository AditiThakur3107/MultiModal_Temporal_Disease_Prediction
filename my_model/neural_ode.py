#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import HeteroData

# ---- Neural ODE solver ----
try:
    from torchdiffeq import odeint
except ImportError as e:
    raise ImportError(
        "torchdiffeq is required. Install with: pip install torchdiffeq"
    ) from e


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  Neural ODE + RNN for visit embeddings (ODE-RNN style)
# ============================================================

class ODEFunc(nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.Tanh(),
            nn.Linear(d_hid, d_hid),
        )

    def forward(self, t, h):
        # h: [batch, d_hid] or [d_hid]
        return self.net(h)


class VisitNeuralODE(nn.Module):
    """
    Given a sequence of visit embeddings v_i and time gaps Δt_i,
    computes updated embeddings with an ODE-RNN and attention.

    Inputs:
      visit_x : [T, D_visit]
      delta_t : [T]  (delta_t[0] can be 0, others >=0)

    Outputs:
      visit_x_new : [T, D_visit] (updated embeddings)
      alpha       : [T]          (attention weights over visits)
    """

    def __init__(self, d_visit, d_hid=256):
        super().__init__()
        self.d_visit = d_visit
        self.d_hid = d_hid

        self.proj_in = nn.Linear(d_visit, d_hid)
        self.odefunc = ODEFunc(d_hid)
        self.gru_cell = nn.GRUCell(d_visit, d_hid)
        self.proj_out = nn.Linear(d_hid, d_visit)

        self.att_mlp = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.Tanh(),
            nn.Linear(d_hid, 1),
        )

    def forward(self, visit_x: torch.Tensor, delta_t: torch.Tensor):
        """
        visit_x: [T, D]
        delta_t: [T]  (first element can be zero)
        """
        visit_x = visit_x.to(device)
        delta_t = delta_t.to(device)

        T, D = visit_x.shape
        if T == 0:
            raise ValueError("Empty visit sequence")

        # absolute times from delta_t
        times = torch.cumsum(delta_t, dim=0)  # [T]
        # initial hidden state from first visit
        h = self.proj_in(visit_x[0])  # [d_hid]
        h_list = [h]

        t_prev = times[0]

        # iterate over visits 1..T-1
        for i in range(1, T):
            t_i = times[i]

            # integrate ODE from t_prev to t_i
            t_span = torch.stack([t_prev, t_i])
            h_ode = odeint(
                self.odefunc,
                h,
                t_span,
                rtol=1e-3,
                atol=1e-3,
                method="dopri5",
            )[-1]  # take final state

            # ODE-RNN update with current visit embedding
            h = self.gru_cell(visit_x[i].unsqueeze(0), h_ode.unsqueeze(0)).squeeze(0)
            h_list.append(h)
            t_prev = t_i

        H = torch.stack(h_list, dim=0)  # [T, d_hid]

        # compute attention over visits
        scores = self.att_mlp(H).squeeze(-1)  # [T]
        alpha = torch.softmax(scores, dim=0)  # [T]

        # map back to visit space
        visit_x_new = self.proj_out(H)  # [T, D]
        return visit_x_new, alpha


# ============================================================
#  Helpers: load Δt per patient from CSV
# ============================================================

def build_delta_t_for_patient(df_all: pd.DataFrame, subject_id) -> list:
    """
    Reproduce the visit ordering we used in build_graph.py:
      - filter by subject_id
      - sort by numeric stay_id
      - group by stay_id (each stay is one visit)

    Then, for each visit index i, produce a time gap Δt_i.
    Here we approximate:
       Δt_0 = 0
       Δt_i = period_length of previous visit (or 1.0 if missing)
    """
    df_p = df_all[df_all["subject_id"] == subject_id].copy()
    if df_p.empty:
        return []

    df_p["_sid"] = pd.to_numeric(df_p["stay_id"], errors="coerce")
    df_p = df_p.sort_values(["_sid", "stay_id"])

    # One row per visit (unique stay_id), in the same order as build_graph
    merged = []
    for stay_id, g in df_p.groupby("stay_id", sort=False):
        row = {"stay_id": stay_id}
        # pick first non-null period_length for this stay
        if "period_length" in g.columns:
            vals = pd.to_numeric(g["period_length"], errors="coerce").dropna()
            row["period_length"] = float(vals.iloc[0]) if not vals.empty else np.nan
        else:
            row["period_length"] = np.nan
        merged.append(row)

    df_v = pd.DataFrame(merged)
    df_v["_sid"] = pd.to_numeric(df_v["stay_id"], errors="coerce")
    df_v = df_v.sort_values("_sid").reset_index(drop=True)

    T = len(df_v)
    if T == 0:
        return []

    # Build delta_t: first visit has 0, subsequent use previous visit's period_length
    deltas = [0.0]
    for i in range(1, T):
        pl = df_v.loc[i - 1, "period_length"]
        if np.isnan(pl):
            pl = 1.0
        # you can also normalize pl here (e.g. log(1+pl)) if needed
        deltas.append(float(pl))

    return deltas


# ============================================================
#  Main logic
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--graphs_dir",
        required=True,
        help="Directory with original patient_*.pt graphs (from build_graph.py)",
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="Original sample_training_data.csv (used in build_graph.py)",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Where to save updated graphs with new visit embeddings",
    )
    ap.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for Neural ODE-RNN",
    )
    args = ap.parse_args()

    graphs_dir = Path(args.graphs_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load CSV once
    df_all = pd.read_csv(args.csv)

    # find all patient_*.pt graphs
    graph_files = sorted(graphs_dir.glob("patient_*.pt"))
    if not graph_files:
        print(f"No graphs found in {graphs_dir}")
        return

    print(f"[INFO] Found {len(graph_files)} graphs in {graphs_dir}")

    # load one graph to infer visit embedding dimension
    tmp_data = torch.load(graph_files[0], map_location="cpu", weights_only=False)
    d_visit = int(tmp_data["visit"].x.shape[1])
    print(f"[INFO] Visit embedding dim: {d_visit}")

    model = VisitNeuralODE(d_visit=d_visit, d_hid=args.hidden_dim).to(device)
    model.eval()  # we're not training here

    stats = {
        "num_graphs": 0,
        "visit_emb_dim": d_visit,
        "hidden_dim": args.hidden_dim,
    }

    for gpath in graph_files:
        try:
            data: HeteroData = torch.load(gpath, map_location="cpu", weights_only=False)
            subject_id = data.meta.get("patient_id", None)
            if subject_id is None:
                # infer from filename: patient_XXXXX.pt
                subject_id = gpath.stem.replace("patient_", "")

            visit_x = data["visit"].x  # [T, D]
            T = visit_x.shape[0]
            if T == 0:
                print(f"[WARN] No visit nodes in {gpath}, skipping.")
                continue

            # build delta_t from CSV
            deltas = build_delta_t_for_patient(df_all, subject_id)
            if len(deltas) != T:
                print(
                    f"[WARN] delta_t length {len(deltas)} != num visits {T} "
                    f"for patient {subject_id}, using uniform Δt=1.0."
                )
            if not deltas or len(deltas) != T:
                deltas = [0.0] + [1.0] * (T - 1)

            delta_t_tensor = torch.tensor(deltas, dtype=torch.float32)

            # run Neural ODE-RNN
            with torch.no_grad():
                visit_x_new, alpha = model(visit_x, delta_t_tensor)

            # update visit.x and store importance
            data["visit"].x = visit_x_new.cpu()
            data["visit"].importance = alpha.cpu().tolist()

            # update meta to record this phase
            meta = dict(data.meta) if hasattr(data, "meta") else {}
            meta["temporal_model"] = "NeuralODE-RNN"
            meta["graphs_source_dir"] = str(graphs_dir)
            data.meta = meta

            # save to outdir
            out_path = outdir / gpath.name
            torch.save(data, out_path)
            stats["num_graphs"] += 1
            print(f"[OK] Updated and saved: {out_path}")

        except Exception as e:
            print(f"[ERROR] Failed on {gpath}: {repr(e)}")
            continue

    # save simple stats
    stats_path = outdir / "temporal_update_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Wrote stats to {stats_path}")
    print(f"[INFO] Updated {stats['num_graphs']} graphs.")


if __name__ == "__main__":
    main()
