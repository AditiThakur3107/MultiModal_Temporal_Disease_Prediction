import torch

# ------------- CONFIG -------------
# put your .pt filename here (or pass absolute path)
file_path = "/home/shivani_2221cs18/aditi_2411ai31/first_phase/FINAL/graph_with_updated_visit_embedding/patient_19991135.pt"
# ----------------------------------


def main():
    # Load the .pt file with full unpickling
    data = torch.load(file_path, map_location="cpu", weights_only=False)

    print("\n=== BASIC INFO ===")
    print("Type of object:", type(data))
    print("Node types:", data.node_types)
    print("Edge types:", data.edge_types)

    # ---------- METADATA ----------
    if hasattr(data, "meta"):
        print("\n=== META ===")
        for k, v in data.meta.items():
            print(f"{k}: {v}")

    # ---------- VISIT NODE ----------
    if "visit" in data.node_types:
        visit = data["visit"]
        print("\n=== VISIT NODE ===")

        print("visit.x shape:", visit.x.shape)
        print("visit.y shape:", visit.y.shape)

        # Show full label names
        if hasattr(visit, "label_cols"):
            print("\nLabel columns:")
            for i, col in enumerate(visit.label_cols):
                print(f"  [{i}] {col}")

        print("\nvisit.y tensor:")
        print(visit.y)

        # Paths
        if hasattr(visit, "img_path"):
            print("\nImage paths per visit:")
            for i, p in enumerate(visit.img_path):
                print(f"  Visit {i}: {p}")

        if hasattr(visit, "ts_path"):
            print("\nTime-series paths per visit:")
            for i, p in enumerate(visit.ts_path):
                print(f"  Visit {i}: {p}")

        if hasattr(visit, "text_raw"):
            print("\nRaw PMH text per visit:")
            for i, txt in enumerate(visit.text_raw):
                print(f"  Visit {i}: {txt}")

    # ---------- ENTITY NODE ----------
    if "entity" in data.node_types:
        entity = data["entity"]
        print("\n=== ENTITY NODE ===")

        print("entity.x shape:", entity.x.shape)

        # Print terms
        if hasattr(entity, "terms"):
            print("\nEntity terms:")
            for i, term in enumerate(entity.terms):
                print(f"  Entity {i}: {term}")

        # Debug: all attributes
        print("\nRaw entity attributes (__dict__):")
        for k, v in entity.__dict__.items():
            print(f"{k}: {type(v)}")

    # ---------- EDGES ----------
    print("\n=== EDGE INDEXES ===")
    for etype in data.edge_types:
        e = data[etype]
        if hasattr(e, "edge_index"):
            print(f"{etype}: shape={e.edge_index.shape}")
            print(e.edge_index)

    # ---------- VISIT → ENTITY WITH NAMES ----------
    if ("visit", "to", "entity") in data.edge_types and hasattr(data["entity"], "terms"):
        print("\n=== VISIT → ENTITY (MAPPED WITH TERMS) ===")
        edge_index = data[("visit", "to", "entity")].edge_index
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        terms = data["entity"].terms

        for s, d in zip(src, dst):
            name = terms[d] if d < len(terms) else "UNKNOWN"
            print(f"  Visit {s}  -->  Entity {d}:  {name}")


if __name__ == "__main__":
    main()
