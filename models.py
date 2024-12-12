import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import kornia.losses as kl
import kornia.color as kc


class SemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_net = deeplabv3_resnet50(pretrained=True)
        self.features = self.semantic_net.backbone
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x)['out']


class ColorPalette(nn.Module):
    def __init__(self, num_colors=32, feature_dim=512):
        super().__init__()
        self.num_colors = num_colors
        self.projection = nn.Linear(feature_dim, num_colors, bias=False)

        initial_colors = torch.tensor([
            [0.529, 0.808, 0.922],  # Sky blue
            [0.196, 0.804, 0.196],  # Green
            [0.545, 0.271, 0.075],  # Brown
            [0.941, 0.902, 0.549],  # Sand
            [0.608, 0.349, 0.714],  # Purple
            [0.941, 0.196, 0.196],  # Red
            [0.118, 0.565, 1.000],  # Deep blue
            [0.827, 0.827, 0.827],  # Gray
        ])
        expanded_colors = torch.randn(num_colors - len(initial_colors), 3)
        expanded_colors = torch.cat([initial_colors, expanded_colors])
        self.color_embeddings = nn.Parameter(expanded_colors)

    def forward(self, semantic_features):
        b, c, h, w = semantic_features.size()
        features_flat = semantic_features.view(b, c, h * w).permute(0, 2, 1)
        attention = self.projection(features_flat)  # (B, H*W, num_colors)
        attention = F.softmax(attention, dim=-1)
        color_proposals = torch.matmul(attention, self.color_embeddings)  # (B, H*W, 3)
        return color_proposals.permute(0, 2, 1).view(b, 3, h, w)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.semantic_encoder = SemanticEncoder()
        self.color_palette = ColorPalette(num_colors=32, feature_dim=512)

        # Texture encoder
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self.encoder_block(64, 128),
            self.encoder_block(128, 256),
            self.encoder_block(256, 512)
        )

        # Semantic feature refinement
        self.semantic_refine = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # Color refinement decoder
        self.color_decoder = nn.ModuleList([
            self.decoder_block(1024, 256),
            self.decoder_block(512, 128),
            self.decoder_block(256, 64),
            self.decoder_block(128, 32)
        ])

        # Final color adjustment
        self.final_adjust = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )

        # Region-aware attention
        self.region_attention = nn.ModuleList([
            AttentionBlock(256),
            AttentionBlock(128),
            AttentionBlock(64),
            AttentionBlock(32)
        ])

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        semantic_features = self.semantic_encoder(x)
        semantic_features = self.semantic_refine(semantic_features)
        color_proposals = self.color_palette(semantic_features)
        texture_features = self.texture_encoder(x)

        # Match spatial sizes
        semantic_features = F.interpolate(
            semantic_features,
            size=(texture_features.size(2), texture_features.size(3)),
            mode='bilinear',
            align_corners=False
        )
        features = torch.cat([semantic_features, texture_features], dim=1)

        skip_connections = []
        for i, decoder in enumerate(self.color_decoder):
            features = decoder(features)
            features = self.region_attention[i](features)
            if i < len(self.color_decoder) - 1:
                skip_connections.append(features)
                features = torch.cat([features, skip_connections[-1]], dim=1)

        output = self.final_adjust(features)

        # Ensure color_proposals matches output size before combining
        color_proposals = F.interpolate(
            color_proposals,
            size=(output.size(2), output.size(3)),
            mode='bilinear',
            align_corners=False
        )

        final_output = output * 0.7 + color_proposals * 0.3
        return final_output


class PhotochromLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual = PerceptualLoss()
        self.color_hist = ColorHistogramLoss()
        self.semantic_consistency = SemanticConsistencyLoss()
        self.color_diversity = ColorDiversityLoss()
        self.ssim = kl.SSIM(window_size=11, reduction='mean')

    def forward(self, generated, target, semantic_features):
        perceptual_loss = self.perceptual(generated, target)
        if torch.isnan(perceptual_loss).any():
            print("DEBUG: NaN detected in Perceptual Loss output.")

        color_hist_loss = self.color_hist(generated, target)
        if torch.isnan(color_hist_loss).any():
            print("DEBUG: NaN detected in Color Histogram Loss output.")

        semantic_loss = self.semantic_consistency(generated, semantic_features)
        if torch.isnan(semantic_loss).any():
            print("DEBUG: NaN detected in Semantic Consistency Loss output.")

        diversity_loss = self.color_diversity(generated)
        if torch.isnan(diversity_loss).any():
            print("DEBUG: NaN detected in Color Diversity Loss output.")

        ssim_loss = 1 - self.ssim(generated, target)
        if torch.isnan(ssim_loss).any():
            print("DEBUG: NaN detected in SSIM Loss output.")

        total_loss = (
            1.0 * perceptual_loss +
            0.5 * color_hist_loss +
            0.3 * semantic_loss +
            0.2 * diversity_loss +
            0.5 * ssim_loss
        )

        if torch.isnan(total_loss).any():
            print("DEBUG: NaN detected in total loss.")
            print("DEBUG: perceptual_loss:", perceptual_loss)
            print("DEBUG: color_hist_loss:", color_hist_loss)
            print("DEBUG: semantic_loss:", semantic_loss)
            print("DEBUG: diversity_loss:", diversity_loss)
            print("DEBUG: ssim_loss:", ssim_loss)

        return {
            'total': total_loss,
            'perceptual': perceptual_loss,
            'color_hist': color_hist_loss,
            'semantic': semantic_loss,
            'diversity': diversity_loss,
            'ssim': ssim_loss
        }


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4],
            vgg[4:9],
            vgg[9:16],
            vgg[16:23]
        ])
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

    def forward(self, x, target):
        # Convert from [-1,1] to [0,1]
        x = (x + 1) / 2
        target = (target + 1) / 2

        # ImageNet normalization
        x = (x - self.mean) / self.std
        target = (target - self.mean) / self.std

        # Force float32 to avoid half precision issues
        x = x.float()
        target = target.float()

        if torch.isnan(x).any() or torch.isnan(target).any():
            print("DEBUG: NaN detected in inputs before VGG forward.")
            print("DEBUG: x range:", x.min().item(), x.max().item())
            print("DEBUG: target range:", target.min().item(), target.max().item())

        loss = 0
        style_loss = 0

        for i, block in enumerate(self.blocks):
            x = block(x)
            with torch.no_grad():
                t = block(target)
            if torch.isnan(x).any() or torch.isnan(t).any():
                print(f"DEBUG: NaN detected within VGG block {i}.")
                print("DEBUG: x range:", x.min().item(), x.max().item())
                print("DEBUG: target range:", t.min().item(), t.max().item())
            loss_val = F.mse_loss(x, t)
            style_val = F.mse_loss(self.gram_matrix(x), self.gram_matrix(t))
            if torch.isnan(loss_val) or torch.isnan(style_val):
                print("DEBUG: NaN in Perceptual Loss computation.")
                print("DEBUG: x Gram:", self.gram_matrix(x))
                print("DEBUG: target Gram:", self.gram_matrix(t))
            loss += loss_val
            style_loss += style_val

        return loss + 0.3 * style_loss


class ColorHistogramLoss(nn.Module):
    def __init__(self, bins=64):
        super().__init__()
        self.bins = bins

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        pred_lab = kc.rgb_to_lab(pred * 0.5 + 0.5)
        target_lab = kc.rgb_to_lab(target * 0.5 + 0.5)

        loss = 0
        eps = 1e-8
        for i in range(3):
            pred_hist = torch.histc(pred_lab[:, i, :, :], bins=self.bins, min=0.0, max=1.0)
            target_hist = torch.histc(target_lab[:, i, :, :], bins=self.bins, min=0.0, max=1.0)

            pred_sum = pred_hist.sum()
            target_sum = target_hist.sum()

            if torch.isnan(pred_hist).any() or torch.isnan(target_hist).any() or torch.isinf(pred_hist).any() or torch.isinf(target_hist).any():
                print(f"DEBUG: NaN/Inf in histogram for channel {i}.")
                print("DEBUG: pred_hist:", pred_hist)
                print("DEBUG: target_hist:", target_hist)
                print("DEBUG: pred_sum:", pred_sum.item(), "target_sum:", target_sum.item())

            # Avoid division by zero
            if pred_sum == 0:
                pred_sum = eps
            if target_sum == 0:
                target_sum = eps

            pred_hist = pred_hist / pred_sum
            target_hist = target_hist / target_sum

            if torch.isnan(pred_hist).any() or torch.isnan(target_hist).any():
                print(f"DEBUG: NaN after histogram normalization for channel {i}.")
                print("DEBUG: pred_hist (normalized):", pred_hist)
                print("DEBUG: target_hist (normalized):", target_hist)

            loss += F.l1_loss(pred_hist.cumsum(0), target_hist.cumsum(0))

        return loss / 3.0


class SemanticConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, generated, semantic_features):
        generated = generated.float()
        semantic_features = semantic_features.float()
        b, c, h, w = generated.size()
        semantic_flat = F.normalize(semantic_features.view(b, -1, h * w), dim=1)
        colors_flat = generated.view(b, -1, h * w)

        if torch.isnan(semantic_flat).any() or torch.isnan(colors_flat).any():
            print("DEBUG: NaN detected in semantic/feature flattening.")

        color_sim = torch.bmm(colors_flat.transpose(1, 2), colors_flat)
        semantic_sim = torch.bmm(semantic_flat.transpose(1, 2), semantic_flat)
        loss = F.mse_loss(color_sim, semantic_sim)
        if torch.isnan(loss).any():
            print("DEBUG: NaN in Semantic Consistency Loss.")
        return loss


class ColorDiversityLoss(nn.Module):
    def __init__(self, num_clusters=8):
        super().__init__()
        self.num_clusters = num_clusters

    def forward(self, generated):
        generated = generated.float()
        b, c, h, w = generated.size()
        pixels = generated.view(b, 3, -1).transpose(1, 2)
        distances = torch.cdist(pixels, pixels)
        if torch.isnan(distances).any() or torch.isinf(distances).any():
            print("DEBUG: NaN/Inf detected in cdist computation.")
        min_distances = distances.topk(k=self.num_clusters, dim=1, largest=False)[0]
        loss = -torch.mean(min_distances)
        if torch.isnan(loss).any():
            print("DEBUG: NaN in Color Diversity Loss.")
        return loss
