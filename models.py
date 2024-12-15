import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Optional
import logging

from debug_utils import MemoryTracker

logger = logging.getLogger(__name__)


class EfficientAttentionBlock(nn.Module):
    """Memory-efficient attention with local windows and reduced channel dimensions"""

    def __init__(
            self,
            in_channels: int,
            reduction_factor: int = 8,
            window_size: int = 8,
            stride: int = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        # Use input-dependent stride to handle different feature map sizes
        self.stride = window_size if stride is None else stride

        # Reduced channel dimensions for Q,K
        self.query = nn.Conv2d(in_channels, in_channels // reduction_factor, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction_factor, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learned scale parameter
        self.gamma = nn.Parameter(torch.zeros(1))

        # Initialize weights
        for m in [self.query, self.key, self.value]:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _extract_windows(self, x: torch.Tensor) -> torch.Tensor:
        """Extract local windows from input tensor"""
        B, C, H, W = x.shape

        # Pad input if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Extract windows using unfold
        windows = x.unfold(2, self.window_size, self.stride) \
            .unfold(3, self.window_size, self.stride)

        # Reshape to (B, num_windows, C, window_size, window_size)
        windows = windows.permute(0, 2, 3, 1, 4, 5) \
            .reshape(-1, C, self.window_size, self.window_size)
        return windows

    def _merge_windows(self, windows: torch.Tensor, orig_size: tuple) -> torch.Tensor:
        """Merge windows back to original feature map size"""
        B, C, H, W = orig_size

        # Calculate number of windows in each dimension
        H_win = (H + self.stride - 1) // self.stride
        W_win = (W + self.stride - 1) // self.stride

        # Reshape windows back to feature map
        windows = windows.view(B, H_win, W_win, C, self.window_size, self.window_size)
        windows = windows.permute(0, 3, 1, 4, 2, 5)

        # Handle padding if present
        out = windows.reshape(B, C, H_win * self.window_size, W_win * self.window_size)
        if out.size(2) > H or out.size(3) > W:
            out = out[:, :, :H, :W]

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Extract windows
        windows = self._extract_windows(x)

        # Compute Q, K, V
        q = self.query(windows)
        k = self.key(windows)
        v = self.value(windows)

        # Reshape for attention computation
        Q = q.reshape(q.size(0), -1, self.window_size * self.window_size)
        K = k.reshape(k.size(0), -1, self.window_size * self.window_size)
        V = v.reshape(v.size(0), -1, self.window_size * self.window_size)

        # Compute attention scores
        attn = torch.bmm(Q.transpose(1, 2), K)
        attn = F.softmax(attn / (self.window_size * self.window_size) ** 0.5, dim=-1)

        # Apply attention to values
        out = torch.bmm(V, attn.transpose(1, 2))

        # Reshape back to windows
        out = out.reshape(windows.size(0), C, self.window_size, self.window_size)

        # Merge windows back to feature map
        out = self._merge_windows(out, (B, C, H, W))

        # Apply learned scale and residual
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
            backbone: str = 'resnet50',
            debug: bool = False,
            tracker: Optional[MemoryTracker] = None
    ):
        super().__init__()
        self.debug = debug
        self.tracker = tracker

        def debug_shape(name: str, tensor: torch.Tensor):
            """Internal helper to log tensor shapes"""
            if self.debug:
                logger.debug(f"{name} shape: {tensor.shape}")

        self.debug_shape = debug_shape  # Save as instance method

        if self.tracker:
            self.tracker.log_memory("generator_init_start")

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

        if self.tracker:
            self.tracker.log_memory("after_encoder_init")

        # Get channel dimensions for each stage
        enc_channels = [256, 512, 1024, 2048]
        dec_channels = [256, 128, 64, 32]

        # Create decoder stages
        self.decoder_stages = nn.ModuleList([
            UpsampleBlock(enc_channels[3], dec_channels[0], scale_factor=2),
            UpsampleBlock(dec_channels[0] + enc_channels[2], dec_channels[1], scale_factor=2),
            UpsampleBlock(dec_channels[1] + enc_channels[1], dec_channels[2], scale_factor=2),
            UpsampleBlock(dec_channels[2] + enc_channels[0], dec_channels[3], scale_factor=2)
        ])

        # Add final upsampling
        self.final_upsample = UpsampleBlock(dec_channels[3], dec_channels[3], scale_factor=2)

        # Add attention blocks - pass tracker to each
        self.attention_blocks = nn.ModuleList([
            # Early layers: larger windows for coarse features
            EfficientAttentionBlock(dec_channels[0], window_size=16),
            # Middle layers: medium windows
            EfficientAttentionBlock(dec_channels[1], window_size=8),
            # Later layers: smaller windows for fine details
            EfficientAttentionBlock(dec_channels[2], window_size=8),
            EfficientAttentionBlock(dec_channels[3], window_size=4)
        ])

        if self.tracker:
            self.tracker.log_memory("after_decoder_init")

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

        if self.tracker:
            self.tracker.log_memory("generator_init_complete")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of generator with memory tracking"""
        if self.tracker:
            self.tracker.log_memory("generator_forward_start")
            self.tracker.log_tensor("generator_input", x)

        self.debug_shape("Input", x)  # Use instance method

        # Store encoder outputs for skip connections
        encoder_features = []

        # Encoder forward pass
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if self.tracker:
                self.tracker.log_tensor(f"encoder_stage_{i}_output", x)
            self.debug_shape(f"Encoder stage {i}", x)  # Use instance method
            encoder_features.append(x)

        if self.tracker:
            self.tracker.log_memory("encoder_complete")

        # Decoder forward pass with skip connections
        for i, (dec_stage, attn) in enumerate(zip(self.decoder_stages, self.attention_blocks)):
            if i > 0:  # Skip connection after first decoder block
                x = torch.cat([x, encoder_features[-(i + 1)]], dim=1)
                if self.tracker:
                    self.tracker.log_tensor(f"skip_connection_{i}", x)
                self.debug_shape(f"Skip connection {i}", x)

            x = dec_stage(x)
            if self.tracker:
                self.tracker.log_tensor(f"decoder_stage_{i}_output", x)
            self.debug_shape(f"Decoder stage {i}", x)

            x = attn(x)
            if self.tracker:
                self.tracker.log_tensor(f"attention_{i}_output", x)
            self.debug_shape(f"Attention {i}", x)

        # Final upsampling and output
        x = self.final_upsample(x)
        x = self.final(x)

        if self.tracker:
            self.tracker.log_tensor("generator_output", x)
            self.tracker.log_memory("generator_forward_complete")

        self.debug_shape("Output", x)
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
    def __init__(self, style_weight: float = 0.0, tracker: Optional[MemoryTracker] = None):
        super().__init__()
        self.tracker = tracker
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

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Track input values
        if self.tracker:
            self.tracker.log_tensor("perceptual_input", input)
            self.tracker.log_tensor("perceptual_target", target)

        input = self.normalize(input)
        target = self.normalize(target)

        content_loss = 0.0
        style_loss = 0.0

        for idx, slice in enumerate(self.slices):
            input_feat = slice(input)
            target_feat = slice(target)

            if self.tracker:
                self.tracker.log_tensor(f"perceptual_slice_{idx}_input", input_feat)
                self.tracker.log_tensor(f"perceptual_slice_{idx}_target", target_feat)

            current_loss = F.l1_loss(input_feat, target_feat)
            content_loss += current_loss

            if self.tracker:
                self.tracker.log_memory(f"perceptual_slice_{idx}_loss: {current_loss.item():.4f}")

            if self.style_weight > 0:
                gram_input = self.gram_matrix(input_feat)
                gram_target = self.gram_matrix(target_feat)
                current_style_loss = F.l1_loss(gram_input, gram_target)
                style_loss += current_style_loss

                if self.tracker:
                    self.tracker.log_memory(f"perceptual_slice_{idx}_style_loss: {current_style_loss.item():.4f}")

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


def rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to YUV"""
    # Matrix for RGB to YUV conversion
    transform = torch.tensor([
        [0.299, 0.587, 0.114],  # Y
        [-0.147, -0.289, 0.436],  # U
        [0.615, -0.515, -0.100]  # V
    ], device=rgb.device)

    # Reshape for matrix multiplication
    rgb_reshaped = rgb.permute(0, 2, 3, 1)  # [B, H, W, C]
    yuv_reshaped = torch.matmul(rgb_reshaped, transform.t())
    yuv = yuv_reshaped.permute(0, 3, 1, 2)  # [B, C, H, W]

    return yuv


def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    """Convert YUV to RGB"""
    # Matrix for YUV to RGB conversion
    transform = torch.tensor([
        [1.0, 0.0, 1.14],
        [1.0, -0.395, -0.581],
        [1.0, 2.032, 0.0]
    ], device=yuv.device)

    # Reshape for matrix multiplication
    yuv_reshaped = yuv.permute(0, 2, 3, 1)  # [B, H, W, C]
    rgb_reshaped = torch.matmul(yuv_reshaped, transform.t())
    rgb = rgb_reshaped.permute(0, 3, 1, 2)  # [B, C, H, W]

    return rgb


class ColorAwareLoss(nn.Module):
    """Loss function that handles color space conversion"""

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, generated_yuv: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        # Extract Y channel from input image (assuming input is grayscale repeated 3 times)
        y_channel = target_rgb[:, 0:1]  # Take first channel since R=G=B for grayscale

        # Combine Y with generated UV
        generated_full_yuv = torch.cat([y_channel, generated_yuv], dim=1)

        # Convert both to RGB for comparison
        generated_rgb = yuv_to_rgb(generated_full_yuv)

        return self.l1_loss(generated_rgb, target_rgb)