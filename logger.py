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


class TrainingLogger:
    def __init__(self, log_dir="experiments", run_name=None, hyperparameters=None):
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

        # Create all directories
        for directory in [self.run_dir, self.images_dir, self.plots_dir, self.tensorboard_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.tensorboard_dir)

        # Initialize metrics storage
        self.metrics_data = []

        # Save run metadata and hyperparameters
        self.save_metadata()
        print(f"Logging to: {self.run_dir}")

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
            f.write(f"Training Run Summary\n")
            f.write(f"===================\n")
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

    def log_images(self, step, epoch, bw_images, generated_images, target_images, num_samples=4):
        """Save progress images showing model improvement"""

        # Denormalize images
        def denorm(img):
            return img.clamp_(-1, 1) * 0.5 + 0.5

        # Select a subset of images
        bw = bw_images[:num_samples]
        gen = generated_images[:num_samples]
        target = target_images[:num_samples]

        # Create a grid with three rows: BW, Generated, Target
        comparison = torch.cat([
            bw.repeat(1, 3, 1, 1),  # Repeat grayscale to 3 channels
            gen,
            target
        ], dim=0)

        # Make grid and convert to PIL
        grid = vutils.make_grid(denorm(comparison), nrow=num_samples, padding=2, normalize=False)
        grid_image = to_pil_image(grid)

        # Add text labels
        plt.figure(figsize=(15, 8))
        plt.imshow(grid_image)
        plt.axis('off')
        plt.title(f'Progress at Step {step} (Epoch {epoch})\nTop: Input | Middle: Generated | Bottom: Target',
                  pad=20)

        # Save high-quality figure
        save_path = self.images_dir / f"progress_step_{step:07d}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()

        # Log to tensorboard as well
        self.writer.add_image('Progress/Comparison', grid, step)

    def log_metrics(self, step, epoch, metrics_dict):
        """Log training metrics"""
        # Create a flat dictionary for this step's metrics
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

        # Add learning rate if present
        if 'learning_rate' in metrics_dict:
            step_metrics['learning_rate'] = metrics_dict['learning_rate']

        # Store the metrics
        self.metrics_data.append(step_metrics)

        # Log to tensorboard
        for key, value in step_metrics.items():
            if key not in ['step', 'epoch']:
                category = key.split('_')[0].capitalize()
                metric_name = '_'.join(key.split('_')[1:])
                self.writer.add_scalar(f'{category}/{metric_name}', value, step)

        # Save metrics periodically
        if step % 1000 == 0:
            self.save_metrics()
            self.plot_loss_curves()

    def plot_loss_curves(self, save=True):
        """Plot training progress"""
        if not self.metrics_data:
            return

        # Convert metrics to DataFrame for easier plotting
        df = pd.DataFrame(self.metrics_data)

        plt.figure(figsize=(12, 8))

        # Plot training metrics
        train_metrics = [col for col in df.columns if col.startswith('train_') and col.endswith('loss')]
        for metric in train_metrics:
            plt.plot(df['step'], df[metric], label=f"Train {metric.replace('train_', '')}", alpha=0.8)

        # Plot validation metrics
        val_metrics = [col for col in df.columns if col.startswith('val_') and col.endswith('loss')]
        for metric in val_metrics:
            plt.plot(df['step'], df[metric], label=f"Val {metric.replace('val_', '')}",
                     linestyle='--', alpha=0.8)

        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        if len(plt.gca().get_lines()) > 0:
            plt.legend()
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(self.plots_dir / 'loss_curves.png',
                        bbox_inches='tight', dpi=200)
            plt.close()
        else:
            plt.show()

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
            sample_indices = np.linspace(0, len(progress_images) - 1, num_samples, dtype=int)
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
            plt.savefig(self.plots_dir / 'training_progress_summary.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

    def close(self):
        """Cleanup and save final plots/summaries"""
        # Save final metrics
        self.save_metrics()

        # Create final plots
        self.plot_loss_curves()
        self.create_progress_summary()

        # Close tensorboard writer
        self.writer.close()