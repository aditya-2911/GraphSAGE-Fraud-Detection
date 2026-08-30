import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd

from src.models.graphsage import GraphSAGENet


def run_discovery(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Semi-Supervised Discovery on device: {device}")

    # 1. Load Data
    data_path = os.path.join("data", "processed", "elliptic_graph.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data not found. Run make_dataset.py first.")
    data = torch.load(data_path, weights_only=False).to(device)

    # 2. Initialize Model and Load Best Weights
    model = GraphSAGENet(data.num_features, args.hidden_dim, 2, aggr=args.aggr).to(
        device
    )
    weight_path = f"models/saved/best_{args.model}.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 3. Isolate the "Unknown" nodes (Class 2)
    unknown_mask = data.y == 2
    unknown_nodes = unknown_mask.nonzero(as_tuple=False).view(-1)
    print(f"Total 'Unknown' transactions in graph: {unknown_nodes.size(0)}")

    # 4. Run Inference on the whole graph
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)  # Convert logits to 0-1 probabilities

        # Extract the probability of being Fraud (Class 0) for the unknown nodes
        fraud_probs = probs[unknown_mask, 0]

    # 5. Filter for high confidence predictions (e.g., > 95% sure it is fraud)
    suspicious_idx = (fraud_probs > args.threshold).nonzero(as_tuple=False).view(-1)

    print(f"\n--- Discovery Results ---")
    print(
        f"Model flagged {suspicious_idx.size(0)} hidden fraudulent transactions with > {args.threshold * 100}% confidence."
    )

    # 6. Save the actionable intelligence to a CSV file
    if suspicious_idx.size(0) > 0:
        results = pd.DataFrame(
            {
                "node_index": unknown_nodes[suspicious_idx].cpu().numpy(),
                "fraud_probability": fraud_probs[suspicious_idx].cpu().numpy(),
            }
        )
        # Sort so the highest probability frauds are at the top
        results = results.sort_values(by="fraud_probability", ascending=False)

        os.makedirs("data/predictions", exist_ok=True)
        save_path = "data/predictions/hidden_fraud_discovered.csv"
        results.to_csv(save_path, index=False)
        print(f"Saved detailed investigation list to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="graphsage")
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--hidden_dim", type=int, default=64)
    # The threshold for flagging a transaction as fraud (Default: 95%)
    parser.add_argument("--threshold", type=float, default=0.95)

    args = parser.parse_args()
    run_discovery(args)
