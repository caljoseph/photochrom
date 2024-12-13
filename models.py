import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


def debug_tensor(tensor, name, detailed=False):
    """Enhanced debug function with distribution analysis and gradient handling"""
    with torch.no_grad():  # Prevent gradient computation during debugging
        # Detach tensor for stats computation
        t = tensor.detach()

        stats = {
            "shape": t.shape,
            "range": [t.min().item(), t.max().item()],
            "mean": t.mean().item(),
            "std": t.std().item()
        }

        if detailed:
            # Analyze distribution
            percentiles = torch.tensor([0, 25, 50, 75, 100], device=t.device)
            stats["percentiles"] = torch.quantile(t.flatten(), percentiles / 100).cpu().numpy()

            # Channel-wise stats for feature maps
            if len(t.shape) == 4:
                step = max(1, t.shape[1] // 4)  # Ensure step is at least 1
                stats["channel_means"] = t.mean(dim=(0, 2, 3))[::step].cpu().numpy()

        print(f"\n=== {name} ===")
        print(f"Shape: {stats['shape']}")
        print(f"Range: [{stats['range'][0]:.3f}, {stats['range'][1]:.3f}]")
        print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")

        if detailed:
            print(f"Percentiles (0,25,50,75,100): {stats['percentiles']}")
            if len(t.shape) == 4:
                print(f"Sample channel means: {stats['channel_means']}")

        return stats


class ColorAwareAttention(nn.Module):
    """
    Custom attention mechanism designed specifically for photochrom colorization.
    Uses separate structural and color pathways with hierarchical region understanding.
    """

    def __init__(self, channels, debug=False):
        super().__init__()
        self.debug = debug
        self.channels = channels

        # Structure pathway
        self.structure_proj = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GroupNorm(8, channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1, groups=channels // 32),
            nn.GroupNorm(8, channels // 4),
            nn.GELU()
        )

        # Color pathway
        self.color_proj = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GroupNorm(8, channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1, groups=channels // 32),
            nn.GroupNorm(8, channels // 4),
            nn.GELU()
        )

        # Multi-scale region understanding
        self.region_scales = [1, 2, 4]
        self.region_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels // 4, channels // 4, kernel_size=3,
                          padding=1, stride=scale),
                nn.GroupNorm(8, channels // 4),
                nn.GELU()
            ) for scale in self.region_scales
        ])

        # Value projection
        self.value = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(16, channels),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels // 16),
            nn.GroupNorm(16, channels),
            nn.GELU()
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(len(self.region_scales) * channels, channels, 1),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )

        self.scale = nn.Parameter(torch.ones(len(self.region_scales)))
        self.gamma = nn.Parameter(torch.zeros(1))

    def _compute_region_attention(self, q, k, v, scale_idx):
        B, C, H, W = q.shape

        # Reshape all tensors to have compatible dimensions
        q_flat = q.view(B, C, -1)  # B, C, HW
        k_flat = k.view(B, C, -1)  # B, C, HW
        v_flat = v.view(B, v.size(1), -1)  # B, C_v, HW

        # Compute attention scores
        attn = torch.bmm(q_flat.permute(0, 2, 1), k_flat)  # B, HW, HW
        attn = attn * self.scale[scale_idx] / np.sqrt(C)
        attn = F.softmax(attn, dim=-1)

        if self.debug:
            debug_tensor(attn, f"Attention Scale {scale_idx}")

        # Apply attention
        out = torch.bmm(v_flat, attn.permute(0, 2, 1))  # B, C_v, HW
        return out.view(B, v.size(1), H, W)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.debug:
            debug_tensor(x, "Attention Input")

        # Extract structure and color features
        struct_feats = self.structure_proj(x)
        color_feats = self.color_proj(x)
        value = self.value(x)

        if self.debug:
            debug_tensor(struct_feats, "Structure Features")
            debug_tensor(color_feats, "Color Features")

        # Multi-scale attention
        outputs = []
        for i, (scale, conv) in enumerate(zip(self.region_scales, self.region_convs)):
            # Get regional features at current scale
            q_struct = conv(struct_feats)
            k_struct = conv(struct_feats)
            q_color = conv(color_feats)

            # Scale value features to match current resolution
            current_h, current_w = H // scale, W // scale
            v_scaled = F.interpolate(value, size=(current_h, current_w),
                                     mode='bilinear', align_corners=False)

            # Compute attention with dimension matching
            struct_out = self._compute_region_attention(q_struct, k_struct, v_scaled, i)
            color_out = self._compute_region_attention(q_color, k_struct, v_scaled, i)

            # Return to original resolution
            if scale > 1:
                struct_out = F.interpolate(struct_out, size=(H, W),
                                           mode='bilinear', align_corners=False)

            outputs.append(struct_out)  # Only use structural attention output for simplicity

        # Combine multi-scale outputs
        combined = torch.cat(outputs, dim=1)
        out = self.out_proj(combined)

        return x + self.gamma * out

class SemanticEncoder(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )

        self.projection = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.GroupNorm(16, 512),
            nn.GELU(),
            nn.Conv2d(512, 512, 3, padding=1, groups=16),
            nn.GroupNorm(16, 512),
            nn.GELU()
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        if self.debug:
            debug_tensor(x, "Semantic Encoder Input")

        features = self.encoder(x)
        features = self.projection(features)

        if self.debug:
            debug_tensor(features, "Semantic Features", detailed=True)

        return features


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, debug=False):
        super().__init__()
        self.debug = debug

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

        self.residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=16),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=16),
            nn.GroupNorm(8, out_channels)
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        if self.debug:
            debug_tensor(x, "Upsample Input")
            debug_tensor(skip, "Skip Connection")

        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)

        residual = self.residual(x)
        attention = self.channel_gate(residual)
        out = x + residual * attention

        if self.debug:
            debug_tensor(out, "Upsample Output")

        return out


class Generator(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        init_channels = 48  # Base channel count
        final_encoder_channels = init_channels * 8  # This will be 384

        self.semantic_encoder = SemanticEncoder(debug=debug)

        # Project semantic features to match encoder dimensions
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(512, final_encoder_channels, 1),
            nn.GroupNorm(16, final_encoder_channels),
            nn.GELU()
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, init_channels, 7, padding=3),
            nn.GroupNorm(8, init_channels),
            nn.GELU()
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(init_channels, init_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, init_channels * 2),
            nn.GELU()
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(init_channels * 2, init_channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, init_channels * 4),
            nn.GELU()
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(init_channels * 4, final_encoder_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, final_encoder_channels),
            nn.GELU()
        )

        # Attention blocks
        self.attention1 = ColorAwareAttention(init_channels * 8, debug=debug)
        self.attention2 = ColorAwareAttention(init_channels * 4, debug=debug)

        # Decoder
        self.dec1 = UpsampleBlock(final_encoder_channels, init_channels * 4, init_channels * 4, debug=debug)
        self.dec2 = UpsampleBlock(init_channels * 4, init_channels * 2, init_channels * 2, debug=debug)
        self.dec3 = UpsampleBlock(init_channels * 2, init_channels, init_channels, debug=debug)

        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(init_channels, init_channels // 2, 3, padding=1),
            nn.GroupNorm(8, init_channels // 2),
            nn.GELU(),
            nn.Conv2d(init_channels // 2, 3, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        if self.debug:
            debug_tensor(x, "Generator Input", detailed=True)

        # Extract and project semantic features
        semantic_features = self.semantic_encoder(x)
        semantic_features = self.semantic_projection(semantic_features)

        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        if self.debug:
            debug_tensor(e4, "Main Encoder Features", detailed=True)
            debug_tensor(semantic_features, "Projected Semantic Features", detailed=True)

        # Feature fusion
        e4 = e4 + semantic_features

        if self.debug:
            debug_tensor(e4, "Fused Features", detailed=True)

        # Attention and decoder path
        e4 = self.attention1(e4)
        d1 = self.dec1(e4, e3)
        d1 = self.attention2(d1)
        d2 = self.dec2(d1, e2)
        d3 = self.dec3(d2, e1)

        out = self.final(d3)

        if self.debug:
            debug_tensor(out, "Final Output", detailed=True)

        return out

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:23]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x * 0.5 + 0.5 - self.mean) / self.std

    def forward(self, generated, target):
        generated = self.normalize(generated)
        target = self.normalize(target)
        return F.mse_loss(self.vgg(generated), self.vgg(target))


class ColorHistogramLoss(nn.Module):
    def __init__(self, bins=64):
        super().__init__()
        self.bins = bins
        self.eps = 1e-8

    def get_histogram(self, x):
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)

        hist_list = []
        for i in range(1, 3):  # a and b channels
            hist = torch.histc(x[:, i].flatten(), bins=self.bins, min=0, max=1)
            hist = hist + self.eps
            hist = hist / hist.sum()
            hist_list.append(hist)

        return torch.cat(hist_list)

    def forward(self, generated, target):
        gen_hist = self.get_histogram(generated)
        target_hist = self.get_histogram(target)
        return F.kl_div((gen_hist + self.eps).log(), target_hist, reduction='batchmean')

# def get_recommended_hyperparameters():
#     """Get recommended training hyperparameters"""
#     return {
#         'learning_rate': 2e-4,
#         'batch_size': 1,  # For MPS, increase for CUDA
#         'adam_betas': (0.5, 0.999),
#         'lr_scheduler_patience': 3,
#         'lr_scheduler_factor': 0.5,
#         'loss_weights': {
#             'l1': 1.0,
#             'perceptual': 0.1,
#             'color_histogram': 0.05
#         },
#         'gradient_clip_val': 1.0
#     }