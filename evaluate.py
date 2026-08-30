import os
import argparse
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.models.baselines import MLPNet
from src.models.graphsage import GraphSAGENet


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    data = torch.load("data/processed/elliptic_graph.pt", weights_only=False).to(device)

    # Filter out unknown nodes
    labeled_mask = data.y != 2
    data.test_mask = data.test_mask & labeled_mask

    # 2. Initialize Model
    in_channels = data.num_features
    out_channels = 2

    if args.model == "mlp":
        model = MLPNet(in_channels, args.hidden_dim, out_channels, dropout=0.5).to(
            device
        )
    elif args.model == "graphsage":
        model = GraphSAGENet(
            in_channels, args.hidden_dim, out_channels, dropout=0.5, aggr=args.aggr
        ).to(device)

    # 3. Load Saved Weights
    weight_path = f"models/saved/best_{args.model}.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 4. Run Inference
    with torch.no_grad():
        out = (
            model(data.x, data.edge_index)
            if args.model == "graphsage"
            else model(data.x)
        )
        pred = out.argmax(dim=1)

        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()

    # 5. Print Professional Metrics
    print(f"--- Evaluation Results for {args.model.upper()} ---")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    # target_names: 0 is Illicit (Fraud), 1 is Licit (Normal)
    print(
        classification_report(
            y_true, y_pred, target_names=["Illicit (Fraud)", "Licit (Normal)"]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="graphsage", choices=["mlp", "graphsage"]
    )
    parser.add_argument(
        "--aggr", type=str, default="mean", choices=["mean", "max", "lstm"]
    )
    parser.add_argument("--hidden_dim", type=int, default=64)
    args = parser.parse_args()
    evaluate(args)
