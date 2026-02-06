"""
Train tile classifier: Phikon-v2 frozen encoder + MLP head.
Split by slide_id (GroupShuffleSplit), BCEWithLogitsLoss with pos_weight.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Defaults
PHIKON_MODEL_NAME = "owkin/phikon-v2"
HIDDEN_DIM = 1024  # Phikon-v2 CLS embedding size
HEAD_HIDDEN = 256
DROPOUT = 0.2
VAL_FRAC = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.5


class TileDataset(Dataset):
    def __init__(self, paths, labels, processor, transform=None):
        self.paths = paths
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e
        if self.transform:
            img = self.transform(img)
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = torch.tensor(self.labels[i], dtype=torch.float32)
        return pixel_values, label


class PhikonMLPHead(nn.Module):
    def __init__(self, hidden_dim: int, head_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="Train Phikon-v2 + MLP tile classifier")
    parser.add_argument("--labels_csv", type=str, default="outputs/labels.csv", help="Path to labels CSV")
    parser.add_argument("--out_dir", type=str, default="outputs/models", help="Directory for checkpoints")
    parser.add_argument("--val_frac", type=float, default=VAL_FRAC, help="Validation fraction (by slide)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for head")
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    if df.empty:
        raise ValueError("Labels CSV is empty")
    for col in ["tile_path", "slide_id", "label"]:
        if col not in df.columns:
            raise ValueError(f"Labels CSV must have column: {col}")

    # Split by slide_id
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.random_state)
    train_idx, val_idx = next(gss.split(df, df["label"], groups=df["slide_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    print("Loading Phikon-v2 processor and encoder...")
    processor = AutoImageProcessor.from_pretrained(PHIKON_MODEL_NAME)
    encoder = AutoModel.from_pretrained(PHIKON_MODEL_NAME)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device)
    encoder.eval()

    train_ds = TileDataset(
        train_df["tile_path"].tolist(),
        train_df["label"].tolist(),
        processor,
    )
    val_ds = TileDataset(
        val_df["tile_path"].tolist(),
        val_df["label"].tolist(),
        processor,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    head = PhikonMLPHead(HIDDEN_DIM, head_hidden=HEAD_HIDDEN, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0.0
    for epoch in range(args.epochs):
        head.train()
        for pixel_values, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            with torch.no_grad():
                enc_out = encoder(pixel_values=pixel_values)
                feat = enc_out.last_hidden_state[:, 0, :]
            logits = head(feat)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()

        head.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                enc_out = encoder(pixel_values=pixel_values)
                feat = enc_out.last_hidden_state[:, 0, :]
                logits = head(feat)
                val_logits.append(logits.cpu().numpy())
                val_labels.append(labels.numpy())
        val_logits = np.concatenate(val_logits)
        val_labels = np.concatenate(val_labels)
        probs = 1.0 / (1.0 + np.exp(-val_logits))
        auc = roc_auc_score(val_labels, probs) if len(np.unique(val_labels)) > 1 else 0.0
        preds = (probs >= THRESHOLD).astype(int)
        acc = accuracy_score(val_labels, preds)
        cm = confusion_matrix(val_labels, preds)

        print(f"Epoch {epoch+1}  val_auc={auc:.4f}  val_acc={acc:.4f}  cm={cm.tolist()}")

        if auc > best_auc:
            best_auc = auc
            ckpt = {
                "head_state_dict": head.state_dict(),
                "model_name": PHIKON_MODEL_NAME,
                "hidden_dim": HIDDEN_DIM,
                "head_hidden": HEAD_HIDDEN,
                "dropout": DROPOUT,
                "threshold": THRESHOLD,
            }
            torch.save(ckpt, out_dir / "phikon_head_best.pt")
            print(f"  -> Saved best checkpoint (auc={auc:.4f})")

    print(f"Done. Best val AUC: {best_auc:.4f}")


def collect_hard_negatives_stub(
    head, encoder, val_loader, device, top_k: int = 100
) -> list[tuple[str, float]]:
    """
    Optional: collect top-k false positives on negative slides (tiles predicted positive but label=0).
    Returns list of (tile_path, prob). Stub: implement by running validation and filtering.
    """
    # TODO: run inference on val set, keep tiles where label=0 and pred_prob > threshold, sort by prob, return top_k
    return []


if __name__ == "__main__":
    main()
