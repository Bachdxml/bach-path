import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and a skip connection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.skip = nn.Sequential()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)

        return out


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.

    F_g:   channels in gating signal (from decoder)
    F_l:   channels in skip connection (from encoder)
    F_int: intermediate channels
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """g: gating signal from decoder. x: skip connection from encoder."""
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class ResidualAttentionUNet(nn.Module):
    """
    U-Net with residual blocks and attention gates on skip connections.

    in_ch:  number of input channels (3 for RGB)
    out_ch: number of output channels (1 for binary segmentation)
    """

    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        # Encoder
        self.enc1 = ResidualBlock(in_ch, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)

        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)

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

        # Output
        self.out  = nn.Conv2d(64, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = self.dec4(torch.cat([d4, self.att4(g=d4, x=e4)], dim=1))

        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(g=d3, x=e3)], dim=1))

        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(g=d2, x=e2)], dim=1))

        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(g=d1, x=e1)], dim=1))

        return self.out(d1)


if __name__ == "__main__":
    model = ResidualAttentionUNet(in_ch=3, out_ch=1)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
