import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionBlock(nn.Module):
    """Efficient self-attention block optimized for GPU training"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # Initialize weights
        for m in [self.query, self.key, self.value]:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, C, H, W = x.size()

        # Compute query, key, value transformations
        query = self.query(x).view(batch_size, -1, H * W)  # B x C' x N
        key = self.key(x).view(batch_size, -1, H * W)  # B x C' x N
        value = self.value(x).view(batch_size, -1, H * W)  # B x C x N

        # Compute attention scores
        attention = torch.bmm(
            query.permute(0, 2, 1),  # B x N x C'
            key  # B x C' x N
        )  # B x N x N

        # Normalize attention scores
        attention = F.softmax(attention, dim=-1)

        # Apply attention to value
        out = torch.bmm(
            value,  # B x C x N
            attention.permute(0, 2, 1)  # B x N x N
        )  # B x C x N

        # Reshape back to spatial dimensions
        out = out.view(batch_size, C, H, W)

        # Apply learnable scaling factor and residual connection
        return self.gamma * out + x

class UpsampleBlock(nn.Module):
    """Upsampling block with optional residual connection"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            scale_factor: int = 2,
            use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual

        # Main convolution path
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, padding=1),
            nn.BatchNorm2d(out_channels * (scale_factor ** 2)),
            nn.ReLU(inplace=True)
        )

        # Pixel shuffle upsampling
        self.shuffle = nn.PixelShuffle(scale_factor)

        # Optional residual connection
        if use_residual and in_channels == out_channels:
            self.residual = nn.Upsample(
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False
            )
        else:
            self.residual = None

        # Gaussian blur to reduce checkerboard artifacts
        kernel_size = scale_factor * 2 - 1
        self.blur = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode='reflect',
            bias=False,
            groups=out_channels
        )

        # Initialize blur kernel with Gaussian
        with torch.no_grad():
            kernel = torch.zeros(kernel_size, kernel_size)
            center = kernel_size // 2
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

            for x in range(kernel_size):
                for y in range(kernel_size):
                    diff = torch.tensor([x - center, y - center])
                    kernel[x, y] = torch.exp(-(diff @ diff) / (2 * sigma ** 2))

            kernel = kernel / kernel.sum()

            # Set kernel for all output channels
            kernel = kernel.expand(out_channels, 1, kernel_size, kernel_size)
            self.blur.weight.data = kernel
            self.blur.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.shuffle(out)

        if self.residual is not None:
            out = out + self.residual(x)

        return self.blur(out)


class PhotochromGenerator(nn.Module):
    """Generator model for photochrom-style colorization"""

    def __init__(
            self,
            pretrained: bool = True,
            backbone: str = 'resnet101',
            debug: bool = False
    ):
        super().__init__()
        self.debug = debug

        # Get specified backbone model
        if backbone == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract encoder layers
        self.encoder_stages = nn.ModuleList([
            nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1
            ),
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        ])

        # Get channel dimensions for each stage
        enc_channels = [256, 512, 1024, 2048]
        dec_channels = [256, 128, 64, 32]

        # Create decoder stages
        self.decoder_stages = nn.ModuleList([
            UpsampleBlock(enc_channels[3], dec_channels[0]),
            UpsampleBlock(dec_channels[0] + enc_channels[2], dec_channels[1]),
            UpsampleBlock(dec_channels[1] + enc_channels[1], dec_channels[2]),
            UpsampleBlock(dec_channels[2] + enc_channels[0], dec_channels[3])
        ])

        # Add attention blocks after each decoder stage
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(dec_channels[0]),
            AttentionBlock(dec_channels[1]),
            AttentionBlock(dec_channels[2]),
            AttentionBlock(dec_channels[3])
        ])

        # Final output layers
        self.final = nn.Sequential(
            nn.Conv2d(dec_channels[3], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),  # UV channels only
            nn.Tanh()
        )

        # Initialize final layers
        for m in self.final:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if debug:
            logger.setLevel(logging.DEBUG)

    def _debug_shape(self, name: str, tensor: torch.Tensor):
        """Helper to log tensor shapes during forward pass"""
        if self.debug:
            logger.debug(f"{name} shape: {tensor.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of generator
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            UV color channels of shape (B, 2, H, W)
        """
        self._debug_shape("Input", x)

        # Store encoder outputs for skip connections
        encoder_features = []

        # Encoder forward pass
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            self._debug_shape(f"Encoder stage {i}", x)
            encoder_features.append(x)

        # Decoder forward pass with skip connections
        for i, (dec_stage, attn) in enumerate(zip(self.decoder_stages, self.attention_blocks)):
            if i > 0:  # Skip connection after first decoder block
                x = torch.cat([x, encoder_features[-(i + 1)]], dim=1)
                self._debug_shape(f"Skip connection {i}", x)

            x = dec_stage(x)
            self._debug_shape(f"Decoder stage {i}", x)

            x = attn(x)
            self._debug_shape(f"Attention {i}", x)

        # Final output
        x = self.final(x)
        self._debug_shape("Output", x)

        return x


def init_weights(model: nn.Module):
    """Initialize network weights using kaiming normal"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for training"""

    def __init__(self, style_weight: float = 0.0):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),  # relu1_2
            nn.Sequential(*list(vgg.children())[4:9]),  # relu2_2
            nn.Sequential(*list(vgg.children())[9:16]),  # relu3_3
            nn.Sequential(*list(vgg.children())[16:23])  # relu4_3
        ])

        for param in self.parameters():
            param.requires_grad = False

        self.style_weight = style_weight
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input = self.normalize(input)
        target = self.normalize(target)

        content_loss = 0.0
        style_loss = 0.0

        for slice in self.slices:
            input = slice(input)
            target = slice(target)

            content_loss += F.l1_loss(input, target)

            if self.style_weight > 0:
                style_loss += F.l1_loss(
                    self.gram_matrix(input),
                    self.gram_matrix(target)
                )

        if self.style_weight > 0:
            return content_loss + self.style_weight * style_loss, style_loss

        return content_loss, None


class ColorHistogramLoss(nn.Module):
    """Color histogram matching loss"""

    def __init__(self, bins: int = 64):
        super().__init__()
        self.bins = bins

    def get_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute histogram for batch of images"""
        x = x.view(x.size(0), x.size(1), -1)
        values = torch.linspace(-1, 1, self.bins, device=x.device)
        hist = []

        for i in range(x.size(1)):  # For each channel
            channel = x[:, i, :]
            channel_hist = []

            for j in range(self.bins - 1):
                # Count values in bin range
                mask = (channel >= values[j]) & (channel < values[j + 1])
                count = mask.float().sum(dim=1) / channel.size(1)
                channel_hist.append(count)

            # Last bin includes upper bound
            mask = (channel >= values[-1])
            count = mask.float().sum(dim=1) / channel.size(1)
            channel_hist.append(count)

            hist.append(torch.stack(channel_hist, dim=1))

        return torch.cat(hist, dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_hist = self.get_histogram(input)
        target_hist = self.get_histogram(target)
        return F.l1_loss(input_hist, target_hist)