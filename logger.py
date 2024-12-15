import torch
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
import pandas as pd
import seaborn as sns
import socket
import os
from PIL import Image
import torchvision.transforms as T
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TrainingLogger:
    def __init__(
            self,
            log_dir: str = "experiments",
            run_name: Optional[str] = None,
            hyperparameters: Optional[Dict] = None
    ):
        # Create descriptive run name with timestamp if none provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"photochrom_run_{timestamp}"

        # Store hyperparameters
        self.hyperparameters = hyperparameters or {}

        # Create directory structure
        self.base_dir = Path(log_dir)
        self.run_dir = self.base_dir / run_name
        self.images_dir = self.run_dir / "progress_images"
        self.plots_dir = self.run_dir / "plots"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.analysis_dir = self.run_dir / "analysis"

        # Create all directories
        for directory in [
            self.run_dir, self.images_dir, self.plots_dir,
            self.tensorboard_dir, self.analysis_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.tensorboard_dir)

        # Initialize metrics storage
        self.metrics_data = []
        self.best_validation_loss = float('inf')
        self.patience_counter = 0

        # Save run metadata and hyperparameters
        self.save_metadata()
        logger.info(f"Logging to: {self.run_dir}")

        self.writer = SummaryWriter(self.tensorboard_dir)

    def save_metadata(self):
        """Save metadata and hyperparameters about the training run"""
        metadata = {
            'start_time': datetime.now().isoformat(),
            'hostname': socket.gethostname(),
            'python_version': os.sys.version,
            'pytorch_version': torch.__version__,
            'hyperparameters': self.hyperparameters
        }

        # Save as JSON
        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create human-readable summary
        with open(self.run_dir / 'run_summary.txt', 'w') as f:
            f.write(f"Photochrom Training Run Summary\n")
            f.write(f"============================\n")
            f.write(f"Started: {metadata['start_time']}\n")
            f.write(f"Host: {metadata['hostname']}\n")
            f.write(f"\nLoss Weights and Hyperparameters:\n")
            f.write(f"--------------------------------\n")
            for key, value in self.hyperparameters.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")

    def log_images(
            self,
            step: int,
            epoch: int,
            bw_images: torch.Tensor,
            generated_images: torch.Tensor,
            target_images: torch.Tensor,
            num_samples: int = 4
    ):
        """Save progress images showing model improvement"""

        # Denormalize images
        def denorm(x: torch.Tensor) -> torch.Tensor:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            return x * std + mean

        bw = denorm(bw_images)
        gen = generated_images
        target = denorm(target_images)

        # Convert UV to RGB if necessary
        if gen.size(1) == 2:  # If UV output
            # Create Y channel from grayscale input
            y = 0.299 * bw[:, 0] + 0.587 * bw[:, 1] + 0.114 * bw[:, 2]
            y = y.unsqueeze(1)
            # Combine Y with predicted UV
            yuv = torch.cat([y, gen], dim=1)
            # Convert to RGB
            gen = self.yuv_to_rgb(yuv)
            gen = denorm(gen)

        # Create comparison grid
        comparison = torch.cat([bw, gen, target], dim=0)
        grid = vutils.make_grid(
            comparison,
            nrow=bw.size(0),  # Use actual batch size instead of num_samples
            padding=2,
            normalize=True,
            value_range=(0, 1)
        )
        grid_image = to_pil_image(grid)

        # Create figure with labels
        plt.figure(figsize=(15, 8))
        plt.imshow(grid_image)
        plt.axis('off')

        # Add labels for each row
        labels = ['Input', 'Generated', 'Target']
        for idx, label in enumerate(labels):
            plt.text(
                -10,
                idx * (grid_image.size[1] / 3) + grid_image.size[1] / 6,
                label,
                rotation=90,
                verticalalignment='center'
            )

        plt.title(f'Progress at Step {step} (Epoch {epoch})', pad=20)

        # Save high-quality figure
        save_path = self.images_dir / f"progress_step_{step:07d}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()

        # Log to tensorboard
        self.writer.add_image('Progress/Comparison', grid, step)

        # Only save individual images if we have samples to save
        actual_samples = min(bw.size(0), num_samples)
        if actual_samples > 0:
            for i in range(actual_samples):
                sample_dir = self.analysis_dir / f"sample_{i}"
                sample_dir.mkdir(exist_ok=True)

                # Save each version of the image
                images = {
                    'input': bw[i],
                    'generated': gen[i],
                    'target': target[i]
                }

                for name, img in images.items():
                    img_path = sample_dir / f"{name}_step_{step:07d}.png"
                    to_pil_image(img).save(img_path)

    def log_metrics(
            self,
            step: int,
            epoch: int,
            metrics_dict: Dict[str, Union[float, Dict[str, float]]]
    ):
        """Log training metrics and create visualizations"""
        # Create flat dictionary for this step's metrics
        step_metrics = {
            'step': step,
            'epoch': epoch
        }

        # Process training metrics
        if 'train' in metrics_dict:
            for key, value in metrics_dict['train'].items():
                step_metrics[f'train_{key}'] = value

        # Process validation metrics
        if 'val' in metrics_dict:
            for key, value in metrics_dict['val'].items():
                step_metrics[f'val_{key}'] = value

            # Check for new best model
            if 'total_loss' in metrics_dict['val']:
                current_loss = metrics_dict['val']['total_loss']
                if current_loss < self.best_validation_loss:
                    self.best_validation_loss = current_loss
                    self.patience_counter = 0
                    # Save best model metrics
                    with open(self.run_dir / 'best_model_metrics.json', 'w') as f:
                        json.dump(metrics_dict['val'], f, indent=2)
                else:
                    self.patience_counter += 1

        # Add learning rate if present
        if 'learning_rate' in metrics_dict:
            step_metrics['learning_rate'] = metrics_dict['learning_rate']

        # Store metrics
        self.metrics_data.append(step_metrics)

        # Log to tensorboard with proper categorization
        for key, value in step_metrics.items():
            if key not in ['step', 'epoch']:
                category = key.split('_')[0].capitalize()
                metric_name = '_'.join(key.split('_')[1:])
                self.writer.add_scalar(f'{category}/{metric_name}', value, step)

        # Save metrics periodically
        if step % 1000 == 0:
            self.save_metrics()
            self.plot_loss_curves()
            self.create_training_summary()

    def plot_loss_curves(self, save: bool = True):
        """Plot detailed training progress curves"""
        if not self.metrics_data:
            return

        df = pd.DataFrame(self.metrics_data)

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Plot training and validation losses
        self._plot_losses(ax1, df)

        # Plot learning rate
        if 'learning_rate' in df.columns:
            self._plot_learning_rate(ax2, df)

        # Plot component losses
        self._plot_component_losses(ax3, df)

        plt.tight_layout()

        if save:
            plt.savefig(self.plots_dir / 'training_curves.png',
                        bbox_inches='tight', dpi=200)
            plt.close()
        else:
            plt.show()

    def _plot_losses(self, ax, df):
        """Plot main training and validation losses"""
        train_loss = df['train_total_loss'] if 'train_total_loss' in df.columns else None
        val_loss = df['val_total_loss'] if 'val_total_loss' in df.columns else None

        if train_loss is not None:
            ax.plot(df['step'], train_loss, label='Train Loss', alpha=0.8)
        if val_loss is not None:
            ax.plot(df['step'], val_loss, label='Validation Loss',
                    linestyle='--', alpha=0.8)

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_learning_rate(self, ax, df):
        """Plot learning rate curve"""
        ax.plot(df['step'], df['learning_rate'], label='Learning Rate')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    def _plot_component_losses(self, ax, df):
        """Plot individual loss components"""
        components = ['l1_loss', 'perceptual_loss', 'color_hist_loss']
        for comp in components:
            if f'train_{comp}' in df.columns:
                ax.plot(df['step'], df[f'train_{comp}'],
                        label=f'Train {comp}', alpha=0.8)

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def create_training_summary(self):
        """Create a comprehensive training summary"""
        if not self.metrics_data:
            return

        df = pd.DataFrame(self.metrics_data)

        # Calculate statistics
        stats = {
            'Total Steps': len(df),
            'Total Epochs': df['epoch'].max(),
            'Best Validation Loss': self.best_validation_loss,
            'Final Learning Rate': df['learning_rate'].iloc[-1] if 'learning_rate' in df else None,
            'Training Time (hours)': (
                                             pd.to_datetime(datetime.now()) -
                                             pd.to_datetime(self.hyperparameters.get('start_time', datetime.now()))
                                     ).total_seconds() / 3600
        }

        # Save summary
        with open(self.run_dir / 'training_summary.txt', 'w') as f:
            f.write("Training Summary\n")
            f.write("===============\n\n")

            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

    def yuv_to_rgb(self, yuv: torch.Tensor) -> torch.Tensor:
        """Convert YUV tensor to RGB"""
        # YUV to RGB conversion matrix
        transform = torch.tensor([
            [1.0, 0.0, 1.13983],
            [1.0, -0.39465, -0.58060],
            [1.0, 2.03211, 0.0]
        ]).to(yuv.device)

        # Reshape for matrix multiplication
        batch_size, _, height, width = yuv.shape
        yuv_reshaped = yuv.permute(0, 2, 3, 1).reshape(-1, 3)

        # Convert to RGB
        rgb = torch.mm(yuv_reshaped, transform.t())

        # Reshape back
        return rgb.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2)

    def save_metrics(self):
        """Save metrics to disk"""
        if not self.metrics_data:
            return

        # Save as CSV
        df = pd.DataFrame(self.metrics_data)
        df.to_csv(self.run_dir / 'metrics.csv', index=False)

        # Save as JSON
        with open(self.run_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics_data, f, indent=2)

    def create_progress_summary(self):
        """Create a summary image showing progress throughout training"""
        progress_images = sorted(self.images_dir.glob('progress_step_*.png'))

        if len(progress_images) > 0:
            num_samples = min(5, len(progress_images))
            sample_indices = np.linspace(
                0, len(progress_images) - 1, num_samples, dtype=int
            )
            selected_images = [progress_images[i] for i in sample_indices]

            plt.figure(figsize=(20, 6 * num_samples))
            for idx, img_path in enumerate(selected_images):
                plt.subplot(num_samples, 1, idx + 1)
                img = Image.open(img_path)
                plt.imshow(img)
                plt.axis('off')
                step = int(img_path.stem.split('_')[-1])
                plt.title(f'Step {step}')

            plt.tight_layout()
            plt

    def close(self):
        """Cleanup and save final summaries"""
        try:
            # Save final metrics
            self.save_metrics()

            # Create final visualizations
            if hasattr(self, 'metrics_data') and self.metrics_data:
                self.plot_loss_curves(save=True)
                self.create_progress_summary()

            # Close tensorboard writer
            if hasattr(self, 'writer'):
                self.writer.close()

        except Exception as e:
            logger.error(f"Error during logger cleanup: {str(e)}")

        logger.info("Training logger closed successfully")