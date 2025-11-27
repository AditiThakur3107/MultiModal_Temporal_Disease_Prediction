import torch
from torch_geometric.data import HeteroData

path = "/home/shivani_2221cs18/aditi_2411ai31/first_phase/FINAL/INPUT_GRAPH_1/patient_19991135.pt"   # change to your graph

data = torch.load(path, map_location="cpu", weights_only=False)

print("\n=== VISIT NODE EMBEDDINGS ===")
print("Shape:", data["visit"].x.shape)
print(data["visit"].x)

print("\n=== ENTITY NODE EMBEDDINGS ===")
print("Shape:", data["entity"].x.shape)
print(data["entity"].x)

print("\n=== VISIT LABELS ===")
print(data["visit"].y)
