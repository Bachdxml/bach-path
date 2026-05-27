import torch

def infer_with_neighborhood(model, tiles, device, tile_size=512, k=1):
    """
    tiles : dict of {(x, y): image_tensor [1,3,512,512]}
    k     : neighborhood radius (k=1 → 3×3 = up to 8 neighbors)
    """

    def to_grid(x, y):
        return y // tile_size, x // tile_size  # (row, col)

    # ── Pass 1: density prediction only ──────────────────────────
    density_preds = {}  # (row, col) → int label
    model.eval()
    with torch.no_grad():
        for (x, y), tile in tiles.items():
            _, density_logits, _, _ = model(tile.to(device))
            density_preds[to_grid(x, y)] = density_logits.argmax(dim=1).item()

    # ── Neighborhood aggregation ──────────────────────────────────
    def consensus_label(row, col):
        neighbors = [
            density_preds[(row + dr, col + dc)]
            for dr in range(-k, k + 1)
            for dc in range(-k, k + 1)
            if (row + dr, col + dc) in density_preds
        ]
        return max(set(neighbors), key=neighbors.count)  # majority vote

    # ── Pass 2: segmentation with neighborhood label ──────────────
    seg_masks = {}
    with torch.no_grad():
        for (x, y), tile in tiles.items():
            row, col = to_grid(x, y)
            label = torch.tensor(
                [consensus_label(row, col)], dtype=torch.long, device=device
            )
            seg_logits, _, _, _ = model(tile.to(device), density_label=label)
            seg_masks[(x, y)] = torch.sigmoid(seg_logits).cpu()

    return seg_masks
