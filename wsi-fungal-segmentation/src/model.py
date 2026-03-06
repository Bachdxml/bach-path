import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = nn.Sequential()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1, 1),   nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class DensityEmbedding(nn.Module):
    """
    Learns a per-density-class bias that is added to the bottleneck
    feature map. This gives the attention gates an explicit signal about
    the density regime before decoding begins.

    num_classes : number of density classes (low/medium/high/negative = 4)
    bottleneck_ch : channels at bottleneck (1024 in your architecture)
    """
    def __init__(self, num_classes: int = 4, bottleneck_ch: int = 1024):
        super().__init__()
        # Small MLP: class index → embedding vector of size bottleneck_ch
        self.embed = nn.Sequential(
            nn.Embedding(num_classes, 128),   # sparse lookup
            nn.Flatten(),
            nn.Linear(128, bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_ch, bottleneck_ch),
        )

    def forward(self, density_label: torch.Tensor, spatial_shape: tuple):
        """
        density_label : LongTensor [B]
        spatial_shape : (H, W) of the bottleneck feature map
        Returns       : [B, bottleneck_ch, H, W] bias to add to bottleneck
        """
        emb = self.embed[0](density_label)          # [B, 128]
        emb = self.embed[1](emb)                    # [B, 128]  (flatten is no-op here)
        emb = self.embed[2](emb)                    # [B, 1024]
        emb = self.embed[3](emb)
        emb = self.embed[4](emb)                    # [B, 1024]
        # Broadcast across spatial dims → [B, 1024, H, W]
        return emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *spatial_shape)


class ResidualAttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, num_density_classes=4):
        super().__init__()

        # Encoder
        self.enc1 = ResidualBlock(in_ch, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)

        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)

        # Density conditioning injected at bottleneck
        self.density_embed = DensityEmbedding(
            num_classes=num_density_classes, bottleneck_ch=1024)

        # Attention gates
        self.att4 = AttentionGate(F_g=512,  F_l=512,  F_int=256)
        self.att3 = AttentionGate(F_g=256,  F_l=256,  F_int=128)
        self.att2 = AttentionGate(F_g=128,  F_l=128,  F_int=64)
        self.att1 = AttentionGate(F_g=64,   F_l=64,   F_int=32)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4    = ResidualBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3    = ResidualBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2    = ResidualBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1    = ResidualBlock(128, 64)

        self.out  = nn.Conv2d(64, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, density_label: torch.Tensor):
        """
        x             : [B, 3, H, W]  image tiles
        density_label : [B]            long tensor, one class per tile
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # ── Density conditioning ──────────────────────────────────────────
        # Add the density bias to every spatial position in the bottleneck.
        # The attention gates in the decoder will now receive feature maps
        # that are already modulated by density context.
        b = b + self.density_embed(density_label, b.shape[2:])
        # ─────────────────────────────────────────────────────────────────

        # Decoder
        d4 = self.upconv4(b)
        d4 = self.dec4(torch.cat([d4, self.att4(g=d4, x=e4)], dim=1))

        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(g=d3, x=e3)], dim=1))

        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(g=d2, x=e2)], dim=1))

        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(g=d1, x=e1)], dim=1))

        return self.out(d1)


# ── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model  = ResidualAttentionUNet(in_ch=3, out_ch=1, num_density_classes=4)
    x      = torch.randn(2, 3, 512, 512)
    labels = torch.tensor([0, 2], dtype=torch.long)   # low, high
    out    = model(x, labels)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")   # should be [2, 1, 512, 512]
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
