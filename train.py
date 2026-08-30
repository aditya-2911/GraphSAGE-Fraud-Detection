import os
import argparse
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score, recall_score

# Import the models we just wrote
from src.models.baselines import MLPNet
from src.models.graphsage import GraphSAGENet


def main(args):
    # 1. Initialize Weights & Biases for experiment tracking
    wandb.init(
        project="graphsage-fraud-detection",
        config=args,
        name=f"{args.model}_{args.aggr}",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Processed Data
    data_path = os.path.join("data", "processed", "elliptic_graph.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data not found at {data_path}. Run make_dataset.py first."
        )

    data = torch.load(data_path, weights_only=False).to(device)
    
    labeled_mask = data.y != 2
    data.train_mask = data.train_mask & labeled_mask
    data.test_mask = data.test_mask & labeled_mask

    # 3. Initialize the chosen Model
    in_channels = data.num_features
    out_channels = 2  # Illicit (0) and Licit (1)

    if args.model == "mlp":
        model = MLPNet(
            in_channels, args.hidden_dim, out_channels, dropout=args.dropout
        ).to(device)
    elif args.model == "graphsage":
        model = GraphSAGENet(
            in_channels,
            args.hidden_dim,
            out_channels,
            dropout=args.dropout,
            aggr=args.aggr,
        ).to(device)
    else:
        raise ValueError("Invalid model type")

    # 4. Setup Optimizer and Imbalanced Loss
    # Class 0 (illicit fraud) is rare, so we assign it a higher weight in the loss function
    class_weights = torch.tensor([9.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 5. Training Loop
    best_val_f1 = 0.0
    os.makedirs("models/saved", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x, data.edge_index)

        # Calculate loss ONLY on the past training nodes
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # 6. Inductive Evaluation on Unseen Future Nodes
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = pred[data.test_mask].cpu().numpy()

            val_f1 = f1_score(y_true, y_pred, average="macro")
            val_recall = recall_score(
                y_true, y_pred, pos_label=0
            )  # Recall for illicit class

        # Log to WandB
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": loss.item(),
                "val_macro_f1": val_f1,
                "val_illicit_recall": val_recall,
            }
        )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} | Val Recall (Fraud): {val_recall:.4f}"
            )

        # Save best model weights
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"models/saved/best_{args.model}.pth")

    print(f"\nTraining complete. Best Validation F1: {best_val_f1:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="graphsage", choices=["mlp", "graphsage"]
    )
    parser.add_argument(
        "--aggr", type=str, default="mean", choices=["mean", "max", "lstm"]
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)

    args = parser.parse_args()
    main(args)
