import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torchvision.transforms as T
from sklearn.cluster import KMeans
import seaborn as sns
from collections import defaultdict
import logging
import cv2
from scipy import stats

logger = logging.getLogger(__name__)


class PhotochromAnalyzer:
    """Utilities for analyzing photochrom style transfer quality"""

    def __init__(self, analysis_dir: str = "analysis"):
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(exist_ok=True)

        # Standard transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.denorm = T.Compose([
            T.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        ])

    def analyze_color_distribution(
            self,
            generated_img: torch.Tensor,
            target_img: torch.Tensor,
            save_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """Analyze color distribution similarity between generated and target images"""
        # Denormalize if needed
        if generated_img.min() < 0:
            generated_img = self.denorm(generated_img)
        if target_img.min() < 0:
            target_img = self.denorm(target_img)

        # Convert to numpy and reshape
        gen_np = generated_img.cpu().numpy().transpose(1, 2, 0)
        target_np = target_img.cpu().numpy().transpose(1, 2, 0)

        # Convert to LAB color space for perceptual color analysis
        gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target_np, cv2.COLOR_RGB2LAB)

        # Compute color statistics
        metrics = {}

        # Earth Mover's Distance between color distributions
        for channel in range(3):
            gen_hist = cv2.calcHist([gen_lab], [channel], None, [256], [0, 256])
            target_hist = cv2.calcHist([target_lab], [channel], None, [256], [0, 256])

            # Normalize histograms
            gen_hist /= gen_hist.sum()
            target_hist /= target_hist.sum()

            # Compute EMD
            metrics[f'emd_channel_{channel}'] = cv2.EMD(
                gen_hist.astype(np.float32),
                target_hist.astype(np.float32),
                cv2.DIST_L2
            )[0]

        # Analyze color palette
        n_colors = 5
        gen_palette = self._extract_color_palette(gen_np, n_colors)
        target_palette = self._extract_color_palette(target_np, n_colors)

        # Plot comparison if save_path provided
        if save_path:
            self._plot_color_analysis(
                gen_np, target_np,
                gen_palette, target_palette,
                metrics,
                save_path
            )

        return metrics

    def _extract_color_palette(
            self,
            img: np.ndarray,
            n_colors: int = 5
    ) -> np.ndarray:
        """Extract dominant colors from image"""
        # Reshape image to list of pixels
        pixels = img.reshape(-1, 3)

        # Cluster colors
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(pixels)

        # Get colors by cluster centers
        colors = kmeans.cluster_centers_

        # Sort by cluster size
        counts = np.bincount(kmeans.labels_)
        sorted_idx = np.argsort(counts)[::-1]

        return colors[sorted_idx]

    def _plot_color_analysis(
            self,
            gen_img: np.ndarray,
            target_img: np.ndarray,
            gen_palette: np.ndarray,
            target_palette: np.ndarray,
            metrics: Dict[str, float],
            save_path: Path
    ):
        """Create visualization of color analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot images
        axes[0, 0].imshow(gen_img)
        axes[0, 0].set_title('Generated Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(target_img)
        axes[0, 1].set_title('Target Image')
        axes[0, 1].axis('off')

        # Plot color palettes
        def plot_palette(ax, palette, title):
            ax.imshow(palette.reshape(1, -1, 3))
            ax.set_title(title)
            ax.axis('off')

        plot_palette(axes[1, 0], gen_palette, 'Generated Palette')
        plot_palette(axes[1, 1], target_palette, 'Target Palette')

        # Add metrics as text
        metric_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        fig.text(0.5, 0.02, metric_text, ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()

    def analyze_spatial_consistency(
            self,
            generated_img: torch.Tensor,
            target_img: torch.Tensor,
            save_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """Analyze spatial consistency of colorization"""
        # Denormalize if needed
        if generated_img.min() < 0:
            generated_img = self.denorm(generated_img)
        if target_img.min() < 0:
            target_img = self.denorm(target_img)

        # Convert to numpy
        gen_np = generated_img.cpu().numpy().transpose(1, 2, 0)
        target_np = target_img.cpu().numpy().transpose(1, 2, 0)

        # Convert to grayscale for structure comparison
        gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_np, cv2.COLOR_RGB2GRAY)

        metrics = {}

        # Compute SSIM
        metrics['ssim'] = self._compute_ssim(gen_gray, target_gray)

        # Compute gradient similarity
        gen_grad = self._compute_gradients(gen_gray)
        target_grad = self._compute_gradients(target_gray)
        metrics['gradient_similarity'] = np.mean(
            np.abs(gen_grad - target_grad)
        )

        if save_path:
            self._plot_spatial_analysis(
                gen_np, target_np,
                gen_grad, target_grad,
                metrics,
                save_path
            )

        return metrics

    def _compute_ssim(
            self,
            img1: np.ndarray,
            img2: np.ndarray,
            window_size: int = 11
    ) -> float:
        """Compute Structural Similarity Index"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        window = cv2.getGaussianKernel(window_size, 1.5)
        window = np.outer(window, window)

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def _compute_gradients(self, img: np.ndarray) -> np.ndarray:
        """Compute image gradients using Sobel operator"""
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx ** 2 + sobely ** 2)

    def _plot_spatial_analysis(
            self,
            gen_img: np.ndarray,
            target_img: np.ndarray,
            gen_grad: np.ndarray,
            target_grad: np.ndarray,
            metrics: Dict[str, float],
            save_path: Path
    ):
        """Create visualization of spatial analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot original images
        axes[0, 0].imshow(gen_img)
        axes[0, 0].set_title('Generated Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(target_img)
        axes[0, 1].set_title('Target Image')
        axes[0, 1].axis('off')

        # Plot gradients
        axes[1, 0].imshow(gen_grad, cmap='viridis')
        axes[1, 0].set_title('Generated Gradients')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(target_grad, cmap='viridis')
        axes[1, 1].set_title('Target Gradients')
        axes[1, 1].axis('off')

        # Add metrics as text
        metric_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        fig.text(0.5, 0.02, metric_text, ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()

    def analyze_batch(
            self,
            generated_batch: torch.Tensor,
            target_batch: torch.Tensor,
            step: int,
            save_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """Analyze a batch of images and aggregate metrics"""
        if save_dir:
            save_dir = Path(save_dir) / f"step_{step}"
            save_dir.mkdir(parents=True, exist_ok=True)

        batch_metrics = defaultdict(list)

        for i in range(len(generated_batch)):
            # Analyze color distribution
            color_metrics = self.analyze_color_distribution(
                generated_batch[i],
                target_batch[i],
                save_dir / f"color_analysis_{i}.png" if save_dir else None
            )

            # Analyze spatial consistency
            spatial_metrics = self.analyze_spatial_consistency(
                generated_batch[i],
                target_batch[i],
                save_dir / f"spatial_analysis_{i}.png" if save_dir else None
            )

            # Aggregate metrics
            for k, v in {**color_metrics, **spatial_metrics}.items():
                batch_metrics[k].append(v)

        # Compute mean metrics
        return {k: np.mean(v) for k, v in batch_metrics.items()}


def plot_attention_maps(
        attention_weights: torch.Tensor,
        input_image: torch.Tensor,
        save_path: Optional[Path] = None
):
    """Visualize attention maps for debugging"""
    # Reshape attention weights if needed
    if len(attention_weights.shape) == 4:  # B, H, W, W
        attention_weights = attention_weights.mean(1)  # Average over heads

    # Get number of attention points to visualize
    n_points = min(5, attention_weights.size(1))

    fig, axes = plt.subplots(2, n_points, figsize=(4 * n_points, 8))

    # Plot original image
    if input_image.min() < 0:  # If normalized
        denorm = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        input_image = denorm(input_image)

    for i in range(n_points):
        # Plot reference point
        axes[0, i].imshow(input_image.cpu().permute(1, 2, 0))
        point_idx = np.random.randint(0, attention_weights.size(1))
        h = int(point_idx ** 0.5)
        w = point_idx - h * int(point_idx ** 0.5)
        axes[0, i].plot(w, h, 'rx', markersize=10)
        axes[0, i].set_title(f'Reference Point {i + 1}')
        axes[0, i].axis('off')

        # Plot attention map
        attention_map = attention_weights[0, point_idx].reshape(
            int(attention_weights.size(1) ** 0.5),
            int(attention_weights.size(1) ** 0.5)
        ).cpu()
        axes[1, i].imshow(attention_map, cmap='viridis')
        axes[1, i].set_title(f'Attention Map {i + 1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()


def visualize_feature_maps(
        feature_maps: torch.Tensor,
        save_path: Optional[Path] = None,
        max_features: int = 16
) -> None:
    """Visualize feature maps from intermediate layers"""
    # Take first image if batch
    if len(feature_maps.shape) == 4:
        feature_maps = feature_maps[0]

    # Select subset of features to visualize
    n_features = min(max_features, feature_maps.shape[0])
    features = feature_maps[:n_features]

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(n_features)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        if i < n_features:
            # Normalize feature map
            feature = features[i].cpu().numpy()
            feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)

            axes[i].imshow(feature, cmap='viridis')
            axes[i].set_title(f'Feature {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()


class PhotochromStyleAnalyzer:
    """Analyze photochrom-specific style characteristics"""

    def __init__(self):
        self.typical_colors = {
            'sky_blue': np.array([135, 206, 235]) / 255.,
            'grass_green': np.array([34, 139, 34]) / 255.,
            'mountain_brown': np.array([139, 69, 19]) / 255.,
            'water_blue': np.array([0, 105, 148]) / 255.,
            'stone_gray': np.array([128, 128, 128]) / 255.
        }

    def analyze_style_characteristics(
            self,
            image: torch.Tensor,
            save_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """Analyze photochrom-specific style characteristics"""
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image = image.cpu().numpy().transpose(1, 2, 0)

        # Ensure values are in [0, 1]
        if image.max() > 1:
            image = image / 255.

        metrics = {}

        # Analyze color palette similarity to typical photochrom colors
        metrics.update(self._analyze_color_similarity(image))

        # Analyze tonal characteristics
        metrics.update(self._analyze_tonal_characteristics(image))

        # Analyze texture characteristics
        metrics.update(self._analyze_texture_characteristics(image))

        if save_path:
            self._plot_style_analysis(image, metrics, save_path)

        return metrics

    def _analyze_color_similarity(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze similarity to typical photochrom colors"""
        metrics = {}

        # Convert to LAB color space for perceptual color comparison
        image_lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)

        for color_name, rgb_value in self.typical_colors.items():
            # Convert reference color to LAB
            ref_color = (rgb_value * 255).reshape(1, 1, 3).astype(np.uint8)
            ref_lab = cv2.cvtColor(ref_color, cv2.COLOR_RGB2LAB)[0, 0]

            # Compute color difference
            diff = np.sqrt(np.sum((image_lab - ref_lab) ** 2, axis=2))

            # Calculate percentage of pixels similar to this color
            similarity_threshold = 30  # Adjust based on desired sensitivity
            similar_pixels = np.sum(diff < similarity_threshold) / diff.size

            metrics[f'{color_name}_presence'] = similar_pixels

        return metrics

    def _analyze_tonal_characteristics(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze tonal characteristics typical of photochroms"""
        metrics = {}

        # Convert to grayscale for tonal analysis
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Analyze tonal distribution
        metrics['contrast'] = np.std(gray)
        metrics['brightness'] = np.mean(gray)

        # Analyze histogram characteristics
        metrics['tonal_range'] = np.percentile(gray, 95) - np.percentile(gray, 5)
        metrics['midtone_weight'] = np.sum(hist[64:192]) / np.sum(hist)

        return metrics

    def _analyze_texture_characteristics(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze texture characteristics typical of photochroms"""
        metrics = {}

        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Compute gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Analyze gradient characteristics
        metrics['texture_strength'] = np.mean(gradient_magnitude)
        metrics['texture_variance'] = np.std(gradient_magnitude)

        # Compute local binary patterns for texture analysis
        def get_lbp(img, points=8, radius=1):
            lbp = np.zeros_like(img)
            for i in range(points):
                theta = 2 * np.pi * i / points
                x = radius * np.cos(theta)
                y = -radius * np.sin(theta)

                # Bilinear interpolation
                fx = np.floor(x).astype(int)
                fy = np.floor(y).astype(int)
                cx = np.ceil(x).astype(int)
                cy = np.ceil(y).astype(int)

                ty = y - fy
                tx = x - fx

                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty

                neighbor = (w1 * np.roll(np.roll(img, fx, 1), fy, 0) +
                            w2 * np.roll(np.roll(img, cx, 1), fy, 0) +
                            w3 * np.roll(np.roll(img, fx, 1), cy, 0) +
                            w4 * np.roll(np.roll(img, cx, 1), cy, 0))

                lbp += (neighbor > img).astype(np.uint8) << i

            return lbp

        lbp = get_lbp(gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        lbp_hist = lbp_hist.flatten() / lbp_hist.sum()

        # Compute texture uniformity
        metrics['texture_uniformity'] = np.sum(lbp_hist ** 2)

        return metrics

    def _plot_style_analysis(
            self,
            image: np.ndarray,
            metrics: Dict[str, float],
            save_path: Path
    ) -> None:
        """Create visualization of style analysis"""
        fig = plt.figure(figsize=(15, 10))

        # Create grid layout
        gs = plt.GridSpec(2, 3, figure=fig)

        # Plot original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title('Analyzed Image')
        ax1.axis('off')

        # Plot color presence
        ax2 = fig.add_subplot(gs[0, 1])
        color_metrics = {k: v for k, v in metrics.items() if 'presence' in k}
        ax2.bar(range(len(color_metrics)), list(color_metrics.values()))
        ax2.set_xticks(range(len(color_metrics)))
        ax2.set_xticklabels([k.replace('_presence', '') for k in color_metrics.keys()],
                            rotation=45)
        ax2.set_title('Color Analysis')

        # Plot tonal distribution
        ax3 = fig.add_subplot(gs[0, 2])
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        ax3.hist(gray.ravel(), bins=256, range=(0, 256), density=True)
        ax3.set_title('Tonal Distribution')

        # Plot texture analysis
        ax4 = fig.add_subplot(gs[1, :])
        texture_metrics = {k: v for k, v in metrics.items()
                           if any(x in k for x in ['texture', 'contrast'])}
        ax4.bar(range(len(texture_metrics)), list(texture_metrics.values()))
        ax4.set_xticks(range(len(texture_metrics)))
        ax4.set_xticklabels(list(texture_metrics.keys()), rotation=45)
        ax4.set_title('Texture Analysis')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()


def analyze_training_batch(
        model: nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
        save_dir: Optional[Path] = None,
        step: Optional[int] = None
) -> Dict[str, float]:
    """Complete analysis of a training batch"""
    model.eval()
    bw_imgs, color_imgs = batch
    bw_imgs = bw_imgs.to(device)
    color_imgs = color_imgs.to(device)

    analyzer = PhotochromAnalyzer()
    style_analyzer = PhotochromStyleAnalyzer()

    with torch.no_grad():
        # Generate images
        generated_imgs = model(bw_imgs)

        # Basic quality metrics
        basic_metrics = analyzer.analyze_batch(
            generated_imgs,
            color_imgs,
            step or 0,
            save_dir / 'basic_analysis' if save_dir else None
        )

        # Style metrics
        style_metrics = defaultdict(list)
        for i in range(len(generated_imgs)):
            metrics = style_analyzer.analyze_style_characteristics(
                generated_imgs[i],
                save_dir / f'style_analysis_{i}.png' if save_dir else None
            )
            for k, v in metrics.items():
                style_metrics[f'style_{k}'].append(v)

        style_metrics = {k: np.mean(v) for k, v in style_metrics.items()}

    model.train()
    return {**basic_metrics, **style_metrics}