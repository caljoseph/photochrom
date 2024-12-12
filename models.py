import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ResNet18 pretrained for semantic feature extraction
        resnet = models.resnet18(pretrained=True)
        # Remove final layers, keep feature extraction
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Freeze weights
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Handle grayscale input
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Semantic encoder
        self.semantic_encoder = SemanticEncoder()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),  # 512 -> 256
            nn.LeakyReLU(0.2)
        )
        self.enc2 = self.encoder_block(64, 128)  # 256 -> 128
        self.enc3 = self.encoder_block(128, 256) # 128 -> 64
        self.enc4 = self.encoder_block(256, 512) # 64 -> 32

        # Attention blocks
        self.attention1 = AttentionBlock(512)
        self.attention2 = AttentionBlock(256)

        # Decoder (adding an extra decoder block)
        self.dec1 = self.decoder_block(512, 256)  # 32 -> 64
        self.dec2 = self.decoder_block(512, 128)  # 64 -> 128
        self.dec3 = self.decoder_block(256, 64)   # 128 -> 256
        self.dec4 = self.decoder_block(128, 64)   # 256 -> 512 (new block)

        # Adjust color refinement to handle 64 input channels now
        self.color_adj = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Extract semantic features
        semantic_features = self.semantic_encoder(x)

        # Encoder
        e1 = self.enc1(x)   # 64@256x256
        e2 = self.enc2(e1)  # 128@128x128
        e3 = self.enc3(e2)  # 256@64x64
        e4 = self.enc4(e3)  # 512@32x32

        # Attention on bottleneck
        e4 = self.attention1(e4)

        # Decoder
        d1 = self.dec1(e4)  # 256@64x64
        d1 = self.attention2(d1)
        d1 = torch.cat([d1, e3], 1)  # 512@64x64

        d2 = self.dec2(d1)  # 128@128x128
        d2 = torch.cat([d2, e2], 1)  # 256@128x128

        d3 = self.dec3(d2)  # 64@256x256
        d3 = torch.cat([d3, e1], 1)  # 128@256x256

        # New decoder step to get back to 512x512
        d4 = self.dec4(d3)  # 64@512x512

        # Final color refinement
        return self.color_adj(d4)


class ColorHistogramLoss(nn.Module):
    def __init__(self, bins=64):
        super().__init__()
        self.bins = bins

    def forward(self, pred, target):
        # Convert to Lab color space for better color comparison
        pred = pred * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
        target = target * 0.5 + 0.5

        # Calculate histogram for each channel
        loss = 0
        for i in range(3):
            pred_hist = torch.histc(pred[:, i, :, :], bins=self.bins, min=0, max=1)
            target_hist = torch.histc(target[:, i, :, :], bins=self.bins, min=0, max=1)

            # Normalize histograms
            pred_hist = pred_hist / pred_hist.sum()
            target_hist = target_hist / target_hist.sum()

            # Calculate Earth Mover's Distance
            loss += F.l1_loss(pred_hist.cumsum(0), target_hist.cumsum(0))

        return loss / 3.0


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4],  # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16],  # relu3_3
        ])
        for param in self.parameters():
            param.requires_grad = False

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

    def forward(self, x, target):
        loss = 0
        style_loss = 0
        x = x.repeat(1, 1, 1, 1) if x.size(1) == 1 else x
        target = target.repeat(1, 1, 1, 1) if target.size(1) == 1 else target

        for block in self.blocks:
            x = block(x)
            target = block(target)
            loss += F.mse_loss(x, target)
            # Add style loss using gram matrices
            style_loss += F.mse_loss(self.gram_matrix(x), self.gram_matrix(target))

        return loss + 0.5 * style_loss