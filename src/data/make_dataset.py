import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os


def process_data(raw_dir="data/raw", processed_dir="data/processed"):
    print("Loading raw CSVs...")
    # 1. Load Classes
    df_classes = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_classes.csv"))

    # 2. Load Features (This CSV has no headers)
    # Col 0: txId, Col 1: time_step, Col 2-166: features
    df_features = pd.read_csv(
        os.path.join(raw_dir, "elliptic_txs_features.csv"), header=None
    )

    # 3. Load Edgelist
    df_edges = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_edgelist.csv"))

    print("Mapping Node IDs to contiguous integers...")
    # Map original string/int txIds to 0, 1, ..., N-1
    node_ids = df_features[0].values
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(node_ids)}

    print("Processing Features and Time Steps...")
    time_steps = torch.tensor(df_features[1].values, dtype=torch.long)
    features = torch.tensor(df_features.iloc[:, 2:].values, dtype=torch.float)

    print("Processing Classes...")
    # Ensure classes dataframe is ordered exactly like the features dataframe
    df_classes = df_classes.set_index("txId").reindex(node_ids).reset_index()

    # Map Kaggle classes: '1' -> 0 (illicit), '2' -> 1 (licit), 'unknown' -> 2
    class_mapping = {"1": 0, "2": 1, "unknown": 2}
    df_classes["class"] = df_classes["class"].map(class_mapping)
    labels = torch.tensor(df_classes["class"].values, dtype=torch.long)

    print("Processing Edges...")
    # Map source and target node IDs in the edgelist to our new contiguous IDs
    df_edges["txId1"] = df_edges["txId1"].map(id_mapping)
    df_edges["txId2"] = df_edges["txId2"].map(id_mapping)

    # PyTorch Geometric expects edge_index shape [2, num_edges]
    edge_index = torch.tensor(
        [df_edges["txId1"].values, df_edges["txId2"].values], dtype=torch.long
    )

    print("Creating PyTorch Geometric Data object...")
    data = Data(x=features, edge_index=edge_index, y=labels, time_step=time_steps)

    # Create Temporal Masks for Inductive Learning
    # Train on past (time_step < 35), Test on future (time_step >= 35)
    data.train_mask = data.time_step < 35
    data.test_mask = data.time_step >= 35

    # Save the processed data
    os.makedirs(processed_dir, exist_ok=True)
    torch.save(data, os.path.join(processed_dir, "elliptic_graph.pt"))

    print(f"Data processing complete! Saved to {processed_dir}/elliptic_graph.pt")


if __name__ == "__main__":
    process_data()
