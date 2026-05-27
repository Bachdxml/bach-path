import torch
from torch.utils.data import DataLoader, TensorDataset

def infer_with_neighborhood(model, tiles, device, tile_size=512, stride=None, k=1, batch_size=16):
    if stride is None:
        stride = tile_size

    def to_grid(x, y):
        return y // stride, x // stride

    coords = list(tiles.keys())
    tensors = torch.cat([tiles[c] for c in coords], dim=0)  # [N, 3, H, W]

    # ── Pass 1: batched density prediction ───────────────────────
    density_preds = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch = tensors[i:i+batch_size].to(device)
            _, density_logits, _, _ = model(batch)
            labels = density_logits.argmax(dim=1).cpu().tolist()
            for j, label in enumerate(labels):
                x, y = coords[i + j]
                density_preds[to_grid(x, y)] = label

    # ── Neighborhood aggregation ──────────────────────────────────
    def consensus_label(row, col):
        neighbors = [
            density_preds[(row + dr, col + dc)]
            for dr in range(-k, k + 1)
            for dc in range(-k, k + 1)
            if (row + dr, col + dc) in density_preds
        ]
        return max(sorted(set(neighbors)), key=neighbors.count)

    # ── Pass 2: batched segmentation ─────────────────────────────
    consensus_labels = torch.tensor(
        [consensus_label(*to_grid(x, y)) for x, y in coords],
        dtype=torch.long,
    )  # [N] — built once, then sliced per batch

    seg_masks = {}
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch = tensors[i:i+batch_size].to(device)
            batch_labels = consensus_labels[i:i+batch_size].to(device)
            seg_logits, _, _, _ = model(batch, density_label=batch_labels)
            probs = torch.sigmoid(seg_logits).cpu()
            for j in range(len(batch_labels)):
                x, y = coords[i + j]
                seg_masks[(x, y)] = probs[j:j+1]  # [1, 1, H, W]

    return seg_masks
