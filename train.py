import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from models import PhotochromGenerator, PerceptualLoss, ColorHistogramLoss, ColorAwareLoss, yuv_to_rgb
from logger import TrainingLogger
from analysis_utils import PhotochromAnalyzer, PhotochromStyleAnalyzer, analyze_training_batch
import os
import random
import logging
from typing import Dict, Tuple, List
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


class PhotochromDataset(Dataset):
    """Dataset for paired BW and photochrom images"""

    def __init__(self, processed_dir: str, image_size: int = 512, augment: bool = True):
        self.processed_dir = Path(processed_dir)
        self.augment = augment

        # Core transforms - maintaining alignment
        self.core_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # Normalization for pretrained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.bw_images = sorted(self.processed_dir.glob("*_bw.jpg"))
        logger.info(f"Found {len(self.bw_images)} images in {processed_dir}")

    def __len__(self) -> int:
        return len(self.bw_images)

    def augment_pair(self, bw: torch.Tensor, color: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply consistent augmentations to both images"""
        # Random horizontal flip
        if random.random() > 0.5:
            bw = TF.hflip(bw)
            color = TF.hflip(color)

        # Random rotation (-10 to 10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            bw = TF.rotate(bw, angle)
            color = TF.rotate(color, angle)

        return bw, color

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bw_path = self.bw_images[idx]
        color_path = self.processed_dir / f"{bw_path.stem.replace('_bw', '')}_color.jpg"

        bw_img = Image.open(bw_path).convert('L')
        color_img = Image.open(color_path).convert('RGB')

        # Convert to tensors
        bw_tensor = self.core_transform(bw_img)
        color_tensor = self.core_transform(color_img)

        # Apply augmentations if enabled
        if self.augment:
            bw_tensor, color_tensor = self.augment_pair(bw_tensor, color_tensor)

        # Convert BW to 3 channel and normalize
        bw_tensor = bw_tensor.repeat(3, 1, 1)
        bw_tensor = self.normalize(bw_tensor)

        # Normalize color image
        color_tensor = self.normalize(color_tensor)

        return bw_tensor, color_tensor


def validate_model(
        generator: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        logger: TrainingLogger,
        epoch: int,
        step: int,
        num_samples: int = 4
) -> Dict[str, float]:
    """Run validation with integrated analysis"""
    generator.eval()
    val_metrics = {
        'l1_loss': 0.0,
        'perceptual_loss': 0.0,
        'color_hist_loss': 0.0,
        'total_loss': 0.0
    }
    num_batches = 0

    # Initialize loss functions
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    color_hist_loss = ColorHistogramLoss().to(device)

    # Get all validation samples
    all_val_samples = []
    with torch.no_grad():
        for batch in val_loader:
            bw, color = batch[0].to(device), batch[1].to(device)
            all_val_samples.append((bw, color))

    # Create lists for visualization samples
    vis_bw = []
    vis_color = []

    # Get num_samples random indices
    total_samples = len(all_val_samples)
    if total_samples >= num_samples:
        indices = random.sample(range(total_samples), num_samples)
    else:
        indices = list(range(total_samples)) * (num_samples // total_samples + 1)
        indices = indices[:num_samples]

    # Collect the samples
    for idx in indices:
        vis_bw.append(all_val_samples[idx][0])
        vis_color.append(all_val_samples[idx][1])

    # Concatenate samples
    val_bw = torch.cat(vis_bw, dim=0)
    val_color = torch.cat(vis_color, dim=0)

    with torch.no_grad():
        # Generate and log sample images
        val_generated = generator(val_bw)
        logger.log_images(step, epoch, val_bw, val_generated, val_color)

        # Run analysis only on the visualization samples every N epochs
        if epoch % 5 == 0:  # Adjust frequency as needed
            analysis_dir = logger.analysis_dir / f"epoch_{epoch}"
            analysis_metrics = analyze_training_batch(
                generator,
                (val_bw, val_color),
                device,
                save_dir=analysis_dir
            )
            # Add analysis metrics to validation metrics
            for k, v in analysis_metrics.items():
                val_metrics[f'analysis_{k}'] = v

        # Calculate metrics over full validation set
        for bw_imgs, color_imgs in all_val_samples:
            generated_imgs = generator(bw_imgs)

            l1 = l1_loss(generated_imgs, color_imgs)
            perc, _ = perceptual_loss(generated_imgs, color_imgs)
            color = color_hist_loss(generated_imgs, color_imgs)
            total = l1 + 0.1 * perc + 0.05 * color

            val_metrics['l1_loss'] += l1.item()
            val_metrics['perceptual_loss'] += perc.item()
            val_metrics['color_hist_loss'] += color.item()
            val_metrics['total_loss'] += total.item()

            num_batches += 1

    # Calculate averages
    for k in val_metrics:
        if not k.startswith('analysis_'):  # Don't average analysis metrics
            val_metrics[k] /= num_batches

    generator.train()
    return val_metrics


def train_model(
        train_dir: str = "processed_images/synthetic_pairs",
        val_dir: str = "processed_images/real_pairs",
        batch_size: int = 1,
        num_epochs: int = 100,
        lr: float = 0.0002,
        image_size: int = 512,
        device: torch.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
) -> None:
    """Main training loop for photochrom model"""
    logger.info(f"Using device: {device}")
    logger.info(f"Starting training with batch size: {batch_size}")

    # Create GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # Define hyperparameters
    hyperparameters = {
        'loss_weights': {
            'l1': 1.0,
            'perceptual': 0.1,
            'color_histogram': 0.05,
            'style': 0.01
        },
        'learning_rate': lr,
        'batch_size': batch_size,
        'image_size': image_size,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingWarmRestarts',
        'architecture': 'PhotochromGenerator with attention'
    }

    # Initialize training logger
    train_logger = TrainingLogger(hyperparameters=hyperparameters)

    # Create datasets and dataloaders
    train_dataset = PhotochromDataset(train_dir, image_size=image_size, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    val_dataset = PhotochromDataset(val_dir, image_size=image_size, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # Reduced worker count based on warnings
        pin_memory=True
    )

    # Initialize models and losses
    generator = PhotochromGenerator(pretrained=True, debug=True).to(device)
    perceptual_loss = PerceptualLoss(style_weight=0.01).to(device)
    color_hist_loss = ColorHistogramLoss().to(device)
    color_aware_loss = ColorAwareLoss().to(device)

    # Load previous checkpoint if it exists
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    start_epoch = 0
    best_val_loss = float('inf')

    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        logger.info(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Resuming from epoch {start_epoch}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2  # Double the restart interval after each restart
    )

    if checkpoints and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    try:
        for epoch in range(start_epoch, num_epochs):
            total_loss_epoch = 0
            num_batches = 0
            generator.train()

            # Training loop
            for i, (bw_imgs, color_imgs) in enumerate(train_loader):
                bw_imgs = bw_imgs.to(device)
                color_imgs = color_imgs.to(device)

                # Forward pass with mixed precision where available
                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        generated_uv = generator(bw_imgs)
                        # Combine generated UV with Y channel from input
                        y_channel = bw_imgs[:, 0:1]  # Take first channel since R=G=B for grayscale
                        generated_yuv = torch.cat([y_channel, generated_uv], dim=1)

                        # Convert to RGB for loss calculations
                        generated_rgb = yuv_to_rgb(generated_yuv)

                        # Calculate losses
                        l1 = color_aware_loss(generated_uv, color_imgs)
                        perc, style_loss = perceptual_loss(generated_rgb, color_imgs)
                        color = color_hist_loss(generated_rgb, color_imgs)

                        # Combine losses
                        total_loss = (
                                l1 +
                                0.1 * perc +
                                0.05 * color +
                                (0.01 * style_loss if style_loss is not None else 0)
                        )
                else:
                    generated_uv = generator(bw_imgs)
                    y_channel = bw_imgs[:, 0:1]
                    generated_yuv = torch.cat([y_channel, generated_uv], dim=1)
                    generated_rgb = yuv_to_rgb(generated_yuv)

                    l1 = color_aware_loss(generated_uv, color_imgs)
                    perc, style_loss = perceptual_loss(generated_rgb, color_imgs)
                    color = color_hist_loss(generated_rgb, color_imgs)

                    total_loss = (
                            l1 +
                            0.1 * perc +
                            0.05 * color +
                            (0.01 * style_loss if style_loss is not None else 0)
                    )

                # Backward pass
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                total_loss_epoch += total_loss.item()
                num_batches += 1

                # Log progress
                if i % 500 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                        f"LR: {current_lr:.6f}, L1: {l1.item():.4f}, "
                        f"Perc: {perc.item():.4f}, Color: {color.item():.4f}, "
                        f"Total: {total_loss.item():.4f}"
                    )

                    metrics = {
                        'train': {
                            'l1_loss': l1.item(),
                            'perceptual_loss': perc.item(),
                            'color_hist_loss': color.item(),
                            'total_loss': total_loss.item(),
                            'learning_rate': current_lr
                        }
                    }
                    train_logger.log_metrics(i + epoch * len(train_loader), epoch, metrics)

            # End of epoch processing
            avg_loss = total_loss_epoch / num_batches

            # Run validation
            val_metrics = validate_model(
                generator, val_loader, device, train_logger,
                epoch, i + epoch * len(train_loader)
            )

            # Update learning rate
            scheduler.step(avg_loss)

            # Run periodic analysis
            if epoch % 10 == 0:
                logger.info("Running comprehensive analysis...")
                analysis_dir = train_logger.analysis_dir / f"epoch_{epoch}_comprehensive"
                analysis_dir.mkdir(exist_ok=True)

                val_batch = next(iter(val_loader))
                analysis_metrics = analyze_training_batch(
                    generator,
                    (val_batch[0].to(device), val_batch[1].to(device)),
                    device,
                    save_dir=analysis_dir,
                    step=epoch
                )

                val_metrics.update({f'analysis_{k}': v for k, v in analysis_metrics.items()})

            # Log end of epoch metrics
            metrics = {
                'train': {'epoch_loss': avg_loss},
                'val': val_metrics,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            train_logger.log_metrics(i + epoch * len(train_loader), epoch, metrics)

            # Save checkpoint
            val_loss = val_metrics['total_loss']
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)

            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss
            }, checkpoint_path)

            # Save best model separately
            if is_best:
                best_model_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, best_model_path)

            # Clean up old checkpoints (keep only latest two)
            old_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))[:-2]
            for ckpt in old_checkpoints:
                ckpt.unlink()

            logger.info(f"Completed epoch {epoch}, Average loss: {avg_loss:.4f}, "
                        f"Validation loss: {val_loss:.4f}"
                        f"{' (Best)' if is_best else ''}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted. Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': total_loss.item(),
            'best_val_loss': best_val_loss
        }, checkpoint_dir / "interrupted_checkpoint.pth")

    finally:
        # Create final analysis summary
        logger.info("Creating final analysis summary...")
        train_logger.create_training_summary()
        train_logger.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()