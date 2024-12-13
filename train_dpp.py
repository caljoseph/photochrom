import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from models import Generator, PerceptualLoss, ColorHistogramLoss
from logger import TrainingLogger
import os
import random


# Your original PhotochromDataset unchanged
class PhotochromDataset(Dataset):
    def __init__(self, processed_dir, image_size=512):
        self.processed_dir = Path(processed_dir)

        # Simple transforms - maintaining exact alignment
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.color_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
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


# Your original validate_model unchanged
def validate_model(generator, val_loader, device, logger, epoch, step, num_samples=4):
    """Run validation and log results"""
    if isinstance(generator, DDP):
        generator = generator.module

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


def setup_ddp(rank, world_size):
    """Simple DDP setup"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Removed the timeout argument here
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train_model_ddp(
        rank,
        world_size,
        train_dir="processed_images/synthetic_pairs",
        val_dir="processed_images/real_pairs",
        batch_size=16,
        num_epochs=100,
        lr=0.0002,
        image_size=512,
):
    setup_ddp(rank, world_size)

    # Only create logger on main process
    logger = TrainingLogger(hyperparameters={
        'loss_weights': {
            'l1': 1.0,
            'perceptual': 0.1,
            'color_histogram': 0.05
        },
        'learning_rate': lr,
        'batch_size': batch_size * world_size,
        'image_size': image_size,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'architecture': 'UNet with semantic encoder and attention',
        'distributed_training': True,
        'num_gpus': world_size
    }) if rank == 0 else None

    # Create datasets with distributed sampler for training
    train_dataset = PhotochromDataset(train_dir, image_size=image_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=1,  # Reduced workers for DDP
        pin_memory=True
    )

    # Validation loader only on main process
    val_loader = None
    if rank == 0:
        val_dataset = PhotochromDataset(val_dir, image_size=image_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Initialize models and move to GPU
    generator = Generator(debug=False).to(rank)
    generator = DDP(generator, device_ids=[rank])

    perceptual_loss = PerceptualLoss().to(rank)
    color_hist_loss = ColorHistogramLoss().to(rank)
    l1_loss = nn.L1Loss()

    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler('cuda')

    try:
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            total_loss_epoch = 0
            num_batches = 0

            for i, (bw_imgs, color_imgs) in enumerate(train_loader):
                bw_imgs = bw_imgs.to(rank)
                color_imgs = color_imgs.to(rank)

                # Calculate most losses with mixed precision
                with torch.amp.autocast('cuda'):
                    generated_imgs = generator(bw_imgs)
                    l1 = l1_loss(generated_imgs, color_imgs)
                    perc = perceptual_loss(generated_imgs, color_imgs)

                # Calculate color histogram loss in full precision
                color = color_hist_loss(generated_imgs.float(), color_imgs.float())

                # Combine losses
                total_loss = l1 + 0.1 * perc + 0.05 * color

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss_epoch += total_loss.item()
                num_batches += 1

                if rank == 0 and i % 100 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                          f"Loss: {total_loss.item():.4f}")

                    metrics = {
                        'train': {
                            'l1_loss': l1.item(),
                            'perceptual_loss': perc.item(),
                            'color_hist_loss': color.item(),
                            'total_loss': total_loss.item()
                        }
                    }
                    logger.log_metrics(i + epoch * len(train_loader), epoch, metrics)

            # Only perform validation and logging on main process
            if rank == 0:
                avg_loss = total_loss_epoch / num_batches
                val_metrics = validate_model(generator, val_loader, rank, logger, epoch,
                                             i + epoch * len(train_loader))
                scheduler.step(avg_loss)

    finally:
        destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs!")
    import torch.multiprocessing as mp
    mp.spawn(train_model_ddp, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    main()
