import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.amp import autocast, GradScaler
import numpy as np

from debug_utils import MemoryTracker
from models import PhotochromGenerator, PerceptualLoss, ColorHistogramLoss, ColorAwareLoss, rgb_to_yuv, yuv_to_rgb
from logger import TrainingLogger
import os
import random


class PhotochromDataset(Dataset):
    """Dataset for paired BW and photochrom images"""

    def __init__(self, processed_dir: str, image_size: int = 512, augment: bool = True):
        self.processed_dir = Path(processed_dir)
        self.augment = augment

        # Core transforms
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
        print(f"Found {len(self.bw_images)} images in {processed_dir}")

    def __len__(self) -> int:
        return len(self.bw_images)

    def __getitem__(self, idx: int) -> tuple:
        bw_path = self.bw_images[idx]
        color_path = self.processed_dir / f"{bw_path.stem.replace('_bw', '')}_color.jpg"

        bw_img = Image.open(bw_path).convert('L')
        color_img = Image.open(color_path).convert('RGB')

        # Convert to tensors
        bw_tensor = self.core_transform(bw_img)
        color_tensor = self.core_transform(color_img)

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
) -> dict:
    """Run validation with proper color space handling"""
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

    color_aware_loss = ColorAwareLoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    color_hist_loss = ColorHistogramLoss().to(device)

    # Get random batch for visualization
    val_batch = next(iter(val_loader))
    val_bw, val_color = val_batch[0][:num_samples].to(device), val_batch[1][:num_samples].to(device)

    with torch.no_grad():
        # Generate and log sample images
        generated_uv = generator(val_bw)
        y_channel = val_bw[:, 0:1]
        generated_yuv = torch.cat([y_channel, generated_uv], dim=1)
        generated_rgb = yuv_to_rgb(generated_yuv)

        logger.log_images(step, epoch, val_bw, generated_rgb, val_color)

        # Calculate metrics over full validation set
        for bw_imgs, color_imgs in val_loader:
            bw_imgs = bw_imgs.to(device)
            color_imgs = color_imgs.to(device)

            generated_uv = generator(bw_imgs)
            y_channel = bw_imgs[:, 0:1]
            generated_yuv = torch.cat([y_channel, generated_uv], dim=1)
            generated_rgb = yuv_to_rgb(generated_yuv)

            l1 = color_aware_loss(generated_uv, color_imgs)
            perc, _ = perceptual_loss(generated_rgb, color_imgs)
            color = color_hist_loss(generated_rgb, color_imgs)
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


def setup_ddp(rank: int, world_size: int):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train_model_ddp(
        rank: int,
        world_size: int,
        train_dir: str = "processed_images/synthetic_pairs",
        val_dir: str = "processed_images/real_pairs",
        batch_size: int = 8,
        num_epochs: int = 100,
        lr: float = 0.0002,
        image_size: int = 512,
):
    setup_ddp(rank, world_size)

    # Initialize memory tracker for this rank
    tracker = MemoryTracker(
        rank=rank,
        log_dir=f"memory_logs/rank_{rank}"
    )
    tracker.reset_peak_memory()
    tracker.log_memory("training_start")

    # Initialize logger on main process only
    logger = TrainingLogger(hyperparameters={
        'loss_weights': {
            'l1': 1.0,
            'perceptual': 0.1,
            'color_histogram': 0.05,
            'style': 0.01
        },
        'learning_rate': lr,
        'batch_size': batch_size * world_size,
        'image_size': image_size,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingWarmRestarts',
        'architecture': 'PhotochromGenerator with attention',
        'distributed_training': True,
        'num_gpus': world_size
    }) if rank == 0 else None

    # Create datasets with distributed sampler
    train_dataset = PhotochromDataset(train_dir, image_size=image_size, augment=True)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=1,
        pin_memory=True
    )

    tracker.log_memory("after_dataloader_init")

    # Validation loader only on main process
    val_loader = None
    if rank == 0:
        val_dataset = PhotochromDataset(val_dir, image_size=image_size, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
        )

    # Initialize models with memory tracking
    tracker.log_memory("before_model_init")
    generator = PhotochromGenerator(
        pretrained=True,
        debug=(rank == 0),
        tracker=tracker  # Pass tracker to the model
    ).to(rank)

    generator = DDP(generator, device_ids=[rank], find_unused_parameters=False)  # Removed find_unused_parameters
    tracker.log_memory("after_model_init")

    # Initialize losses
    color_aware_loss = ColorAwareLoss().to(rank)
    perceptual_loss = PerceptualLoss(style_weight=0.01, tracker=tracker).to(rank)
    color_hist_loss = ColorHistogramLoss().to(rank)
    tracker.log_memory("after_loss_init")

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    scaler = GradScaler('cuda')
    tracker.log_memory("after_optimizer_init")

    # Load checkpoint if exists
    start_epoch = 0
    if rank == 0:
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))

        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=f'cuda:{rank}')
            generator.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

    # Broadcast start_epoch to all processes
    start_epoch = torch.tensor(start_epoch, device=f'cuda:{rank}')
    torch.distributed.broadcast(start_epoch, 0)
    start_epoch = int(start_epoch.item())

    try:
        for epoch in range(start_epoch, num_epochs):
            tracker.reset_peak_memory()
            tracker.log_memory(f"epoch_{epoch}_start")

            train_sampler.set_epoch(epoch)
            total_loss_epoch = 0
            num_batches = 0

            for i, (bw_imgs, color_imgs) in enumerate(train_loader):
                if i % 100 == 0:  # Log memory periodically
                    tracker.log_memory(f"epoch_{epoch}_step_{i}_start")

                bw_imgs = bw_imgs.to(rank)
                color_imgs = color_imgs.to(rank)

                if i % 100 == 0:
                    tracker.log_tensor("input_images", bw_imgs)
                    tracker.log_tensor("target_images", color_imgs)

                with autocast(device_type='cuda'):
                    # Generate UV channels
                    generated_uv = generator(bw_imgs)
                    if i % 100 == 0:
                        tracker.log_tensor("generated_uv", generated_uv)
                        tracker.log_memory("after_generator")

                    # Combine with Y channel and convert to RGB
                    y_channel = bw_imgs[:, 0:1]
                    generated_yuv = torch.cat([y_channel, generated_uv], dim=1)
                    generated_rgb = yuv_to_rgb(generated_yuv)

                    if i % 100 == 0:
                        tracker.log_tensor("generated_rgb", generated_rgb)
                        tracker.log_memory("after_color_conversion")

                    # Calculate losses
                    l1 = color_aware_loss(generated_uv, color_imgs)
                    perc, style_loss = perceptual_loss(generated_rgb, color_imgs)
                    color = color_hist_loss(generated_rgb, color_imgs)

                    if i % 100 == 0:
                        tracker.log_memory("after_loss_computation")

                    total_loss = (
                            l1 +
                            0.1 * perc +
                            0.05 * color +
                            (0.01 * style_loss if style_loss is not None else 0)
                    )

                optimizer.zero_grad()

                if i % 100 == 0:
                    tracker.log_memory("before_backward")

                scaler.scale(total_loss).backward()

                if i % 100 == 0:
                    tracker.log_memory("after_backward")

                scaler.step(optimizer)
                scaler.update()

                if i % 100 == 0:
                    tracker.log_memory("after_optimizer_step")
                    tracker.clear_cache()  # Clear cache periodically

                total_loss_epoch += total_loss.item()
                num_batches += 1

                if rank == 0 and i % 100 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                          f"L1: {l1.item():.4f}, Perc: {perc.item():.4f}, "
                          f"Color: {color.item():.4f}, Total: {total_loss.item():.4f}")

                    metrics = {
                        'train': {
                            'l1_loss': l1.item(),
                            'perceptual_loss': perc.item(),
                            'color_hist_loss': color.item(),
                            'total_loss': total_loss.item(),
                            'learning_rate': optimizer.param_groups[0]['lr']
                        }
                    }
                    logger.log_metrics(i + epoch * len(train_loader), epoch, metrics)

            tracker.log_memory(f"epoch_{epoch}_end")

            # Only perform validation, logging, and checkpointing on main process
            if rank == 0:
                avg_loss = total_loss_epoch / num_batches
                val_metrics = validate_model(
                    generator,
                    val_loader,
                    rank,
                    logger,
                    epoch,
                    i + epoch * len(train_loader)
                )

                scheduler.step(avg_loss)

                # Log end of epoch metrics
                metrics = {
                    'train': {'epoch_loss': avg_loss},
                    'val': val_metrics,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                logger.log_metrics(i + epoch * len(train_loader), epoch, metrics)

                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': generator.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'val_metrics': val_metrics,
                }, checkpoint_path)

                # Clean up old checkpoints
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
        # Save final memory report
        tracker.save_final_report()
        tracker.report_memory_leak()
        destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs!")
    import torch.multiprocessing as mp
    mp.spawn(train_model_ddp, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    main()