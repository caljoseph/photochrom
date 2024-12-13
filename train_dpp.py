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


def validate_model(generator, val_loader, device, logger, epoch, step, num_samples=4):
    """Run validation and log results"""
    # Ensure we're using the base generator model, not DDP wrapper
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
        if logger is not None:
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
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup distributed training resources"""
    destroy_process_group()


def train_model_ddp(
        rank,  # GPU id for this process
        world_size,  # Total number of GPUs
        train_dir="processed_images/synthetic_pairs",
        val_dir="processed_images/real_pairs",
        batch_size=32,  # Increased batch size for multi-GPU
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
        'batch_size': batch_size * world_size,  # Global batch size
        'image_size': image_size,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'architecture': 'UNet with semantic encoder and attention',
        'distributed_training': True,
        'num_gpus': world_size
    }) if rank == 0 else None

    # Create datasets and dataloaders
    train_dataset = PhotochromDataset(train_dir, image_size=image_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Validation loader only needed on main process
    val_loader = None
    if rank == 0:
        val_dataset = PhotochromDataset(val_dir, image_size=image_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize models and losses
    generator = Generator(debug=False).to(rank)  # Move to GPU
    generator = DDP(generator, device_ids=[rank])

    perceptual_loss = PerceptualLoss().to(rank)
    color_hist_loss = ColorHistogramLoss().to(rank)
    l1_loss = nn.L1Loss()

    # Load previous checkpoint if it exists
    checkpoint_dir = Path("checkpoints")
    start_epoch = 0

    if rank == 0:  # Only check on main process
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))

        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=f'cuda:{rank}')
            generator.module.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

    # Ensure all processes start at same epoch
    if world_size > 1:
        start_epoch = torch.tensor(start_epoch, device=rank)
        torch.distributed.broadcast(start_epoch, 0)
        start_epoch = start_epoch.item()

    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()

    try:
        for epoch in range(start_epoch, num_epochs):
            train_sampler.set_epoch(epoch)  # Important for proper shuffling
            total_loss_epoch = 0
            num_batches = 0

            for i, (bw_imgs, color_imgs) in enumerate(train_loader):
                bw_imgs = bw_imgs.to(rank, non_blocking=True)
                color_imgs = color_imgs.to(rank, non_blocking=True)

                with autocast():
                    generated_imgs = generator(bw_imgs)
                    l1 = l1_loss(generated_imgs, color_imgs)
                    perc = perceptual_loss(generated_imgs, color_imgs)

                with autocast(enabled=False):
                    color = color_hist_loss(generated_imgs.float(), color_imgs.float())

                total_loss = l1 + 0.1 * perc + 0.05 * color

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss_epoch += total_loss.item()
                num_batches += 1

                if rank == 0 and i % 100 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                          f"L1: {l1.item():.4f}, Perc: {perc.item():.4f}, "
                          f"Color: {color.item():.4f}, Total: {total_loss.item():.4f}")

                    if logger:
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

                metrics = {
                    'train': {'epoch_loss': avg_loss},
                    'val': val_metrics,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                if logger:
                    logger.log_metrics(i + epoch * len(train_loader), epoch, metrics)

                # Save model checkpoint
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': generator.module.state_dict(),
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
        if rank == 0:
            print("\nTraining interrupted. Saving checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss.item(),
            }, checkpoint_dir / "interrupted_checkpoint.pth")

    finally:
        if logger:
            logger.close()
        cleanup_ddp()


def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs!")

    # Launch training processes
    import torch.multiprocessing as mp
    mp.spawn(
        train_model_ddp,
        args=(world_size,),
        nprocs=world_size,
    )


if __name__ == "__main__":
    main()