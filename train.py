import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from models import Generator, PerceptualLoss, ColorHistogramLoss
from logger import TrainingLogger
import os
import random


class PhotochromDataset(Dataset):
    def __init__(self, processed_dir, image_size=512):
        self.processed_dir = Path(processed_dir)

        # Simple resize and normalization only
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.color_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.bw_images = sorted(self.processed_dir.glob("*_bw.jpg"))
        print(f"Found {len(self.bw_images)} images in {processed_dir}")

    def __len__(self):
        return len(self.bw_images)

    def __getitem__(self, idx):
        bw_path = self.bw_images[idx]
        color_path = self.processed_dir / f"{bw_path.stem.replace('_bw', '')}_color.jpg"

        bw_img = Image.open(bw_path).convert('L')
        color_img = Image.open(color_path).convert('RGB')

        return self.transform(bw_img), self.color_transform(color_img)


def validate_model(generator, val_loader, device, logger, epoch, step, num_samples=4):
    """Run validation and log results"""
    generator.eval()
    val_metrics = {
        'l1_loss': 0.0,
        'perceptual_loss': 0.0,
        'color_hist_loss': 0.0,
        'total_loss': 0.0
    }
    num_batches = 0

    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    color_hist_loss = ColorHistogramLoss().to(device)

    # Get random batch for visualization
    val_batch = next(iter(val_loader))
    val_bw, val_color = val_batch[0][:num_samples].to(device), val_batch[1][:num_samples].to(device)

    with torch.no_grad():
        # Log sample images
        val_generated = generator(val_bw)
        logger.log_images(step, epoch, val_bw, val_generated, val_color)

        # Calculate metrics over full validation set
        for bw_imgs, color_imgs in val_loader:
            bw_imgs = bw_imgs.to(device)
            color_imgs = color_imgs.to(device)

            generated_imgs = generator(bw_imgs)

            l1 = l1_loss(generated_imgs, color_imgs)
            perc = perceptual_loss(generated_imgs, color_imgs)
            color = color_hist_loss(generated_imgs, color_imgs)
            total = l1 + 0.1 * perc + 0.05 * color

            val_metrics['l1_loss'] += l1.item()
            val_metrics['perceptual_loss'] += perc.item()
            val_metrics['color_hist_loss'] += color.item()
            val_metrics['total_loss'] += total.item()

            num_batches += 1

    # Calculate averages
    for k in val_metrics:
        val_metrics[k] /= num_batches

    generator.train()
    return val_metrics


def train_model(
        train_dir="processed_images/synthetic_pairs",
        val_dir="processed_images/real_pairs",
        batch_size=96,
        num_epochs=100,
        lr=0.0002,
        image_size=512,
        device="cuda"
):
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    print(f"Using device: {device}")
    print(f"Starting training with batch size: {batch_size}")

    hyperparameters = {
        'loss_weights': {
            'l1': 1.0,
            'perceptual': 0.1,
            'color_histogram': 0.05
        },
        'learning_rate': lr,
        'batch_size': batch_size,
        'image_size': image_size,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'architecture': 'UNet with semantic encoder and attention'
    }

    logger = TrainingLogger(hyperparameters=hyperparameters)

    # Create datasets and dataloaders
    train_dataset = PhotochromDataset(train_dir, image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = PhotochromDataset(val_dir, image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize models and losses
    generator = Generator().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    color_hist_loss = ColorHistogramLoss().to(device)
    l1_loss = nn.L1Loss()

    # Load previous checkpoint if it exists
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    start_epoch = 0

    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Optimizer with learning rate scheduling
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    try:
        for epoch in range(start_epoch, num_epochs):
            total_loss_epoch = 0
            num_batches = 0

            for i, (bw_imgs, color_imgs) in enumerate(train_loader):
                bw_imgs = bw_imgs.to(device)
                color_imgs = color_imgs.to(device)

                # Run forward pass with autocast
                with torch.amp.autocast('cuda'):  # Updated to new style
                    # Generate colorized images
                    generated_imgs = generator(bw_imgs)

                    # Calculate losses that support fp16
                    l1 = l1_loss(generated_imgs, color_imgs)
                    perc = perceptual_loss(generated_imgs, color_imgs)

                # Calculate color histogram loss in fp32
                with torch.cuda.amp.autocast(enabled=False):
                    color = color_hist_loss(generated_imgs.float(), color_imgs.float())

                # Combine losses with weights
                total_loss = l1 + 0.1 * perc + 0.05 * color

                # Update generator with gradient scaling
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Accumulate loss for epoch average
                total_loss_epoch += total_loss.item()
                num_batches += 1

                # Log progress
                if i % 100 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                          f"L1: {l1.item():.4f}, Perc: {perc.item():.4f}, "
                          f"Color: {color.item():.4f}, Total: {total_loss.item():.4f}")

                    # Log current metrics
                    metrics = {
                        'train': {
                            'l1_loss': l1.item(),
                            'perceptual_loss': perc.item(),
                            'color_hist_loss': color.item(),
                            'total_loss': total_loss.item()
                        }
                    }
                    logger.log_metrics(i + epoch * len(train_loader), epoch, metrics)

            # End of epoch processing
            avg_loss = total_loss_epoch / num_batches

            # Run validation
            val_metrics = validate_model(generator, val_loader, device, logger, epoch, i + epoch * len(train_loader))

            # Update learning rate
            scheduler.step(avg_loss)

            # Log end of epoch metrics
            metrics = {
                'train': {'epoch_loss': avg_loss},
                'val': val_metrics,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            logger.log_metrics(i + epoch * len(train_loader), epoch, metrics)

            # Save model checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'val_metrics': val_metrics,
            }, checkpoint_path)

            # Clean up old checkpoints (keep only the latest two)
            old_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))[:-2]
            for ckpt in old_checkpoints:
                ckpt.unlink()

            print(f"Completed epoch {epoch}, Average loss: {avg_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': total_loss.item(),
        }, checkpoint_dir / "interrupted_checkpoint.pth")

    finally:
        # Final cleanup
        logger.close()


if __name__ == "__main__":
    train_model()