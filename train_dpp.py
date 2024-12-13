import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from models import Generator, PerceptualLoss, ColorHistogramLoss
from logger import TrainingLogger
import os
import gc
import warnings

# Suppress the dataloader workers warning
warnings.filterwarnings("ignore", message="This DataLoader will create")

def get_num_workers():
    """Calculate appropriate number of workers based on system config"""
    try:
        # Check CPU count but limit it
        num_cpus = len(os.sched_getaffinity(0))
        return min(2, max(1, num_cpus // 4))  # Conservative worker count
    except:
        return 1


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
    """Memory-optimized validation"""
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

    # Create losses on device
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    color_hist_loss = ColorHistogramLoss().to(device)

    # Sample visualization with careful memory handling
    try:
        val_batch = next(iter(val_loader))
        with torch.cuda.amp.autocast():
            val_bw, val_color = val_batch[0][:num_samples].to(device), val_batch[1][:num_samples].to(device)
            with torch.no_grad():
                val_generated = generator(val_bw)
                if logger is not None:
                    logger.log_images(step, epoch, val_bw, val_generated, val_color)

            # Clear visualization tensors
            del val_bw, val_color, val_generated
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")

    # Calculate metrics with memory optimization
    with torch.no_grad():
        for bw_imgs, color_imgs in val_loader:
            bw_imgs = bw_imgs.to(device)
            color_imgs = color_imgs.to(device)

            with torch.cuda.amp.autocast():
                generated_imgs = generator(bw_imgs)
                l1 = l1_loss(generated_imgs, color_imgs)
                perc = perceptual_loss(generated_imgs, color_imgs)
                color = color_hist_loss(generated_imgs, color_imgs)
                total = l1 + 0.1 * perc + 0.05 * color

            val_metrics['l1_loss'] += l1.item()
            val_metrics['perceptual_loss'] += perc.item()
            val_metrics['color_hist_loss'] += color.item()
            val_metrics['total_loss'] += total.item()

            del generated_imgs, l1, perc, color, total
            num_batches += 1

    # Calculate averages
    for k in val_metrics:
        val_metrics[k] /= num_batches

    generator.train()
    return val_metrics


def setup_ddp(rank, world_size):
    """Initialize distributed training with error handling"""
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # Initialize process group with timeout
        init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.Store.TIMEOUT_DEFAULT
        )

        torch.cuda.set_device(rank)
    except Exception as e:
        print(f"DDP setup failed on rank {rank}: {e}")
        raise

def cleanup_ddp():
    """Cleanup distributed training resources"""
    destroy_process_group()

def train_model_ddp(
        rank,
        world_size,
        train_dir="processed_images/synthetic_pairs",
        val_dir="processed_images/real_pairs",
        batch_size=16,  # Reduced batch size for better memory management
        num_epochs=100,
        lr=0.0002,
        image_size=512,
        accumulation_steps=2  # Gradient accumulation for effective batch size
):
    # Setup DDP
    setup_ddp(rank, world_size)

    # Clear GPU memory before starting
    torch.cuda.empty_cache()
    gc.collect()

    # Determine number of workers
    num_workers = get_num_workers()

    # Only create logger on main process
    logger = None
    if rank == 0:
        logger = TrainingLogger(hyperparameters={
            'loss_weights': {
                'l1': 1.0,
                'perceptual': 0.1,
                'color_histogram': 0.05
            },
            'learning_rate': lr,
            'batch_size': batch_size * world_size * accumulation_steps,  # Global effective batch size
            'image_size': image_size,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'architecture': 'UNet with semantic encoder and attention',
            'distributed_training': True,
            'num_gpus': world_size,
            'accumulation_steps': accumulation_steps
        })

    # Create datasets and dataloaders with appropriate worker count
    train_dataset = PhotochromDataset(train_dir, image_size=image_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = None
    if rank == 0:
        val_dataset = PhotochromDataset(val_dir, image_size=image_size)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    # Initialize model and move to GPU
    generator = Generator(debug=False).to(rank)
    generator = DDP(generator, device_ids=[rank], find_unused_parameters=False)

    # Initialize losses
    perceptual_loss = PerceptualLoss().to(rank)
    color_hist_loss = ColorHistogramLoss().to(rank)
    l1_loss = nn.L1Loss()

    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda')

    # Load checkpoint if exists
    start_epoch = 0
    if rank == 0:
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))

        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=f'cuda:{rank}', weights_only=True)
            generator.module.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

    # Synchronize start epoch across processes
    if world_size > 1:
        start_epoch = torch.tensor(start_epoch, device=rank)
        torch.distributed.broadcast(start_epoch, 0)
        start_epoch = start_epoch.item()

    try:
        for epoch in range(start_epoch, num_epochs):
            train_sampler.set_epoch(epoch)
            total_loss_epoch = 0
            num_batches = 0
            optimizer.zero_grad()  # Zero gradients at start of epoch

            for i, (bw_imgs, color_imgs) in enumerate(train_loader):
                bw_imgs = bw_imgs.to(rank, non_blocking=True)
                color_imgs = color_imgs.to(rank, non_blocking=True)

                # Forward pass with automatic mixed precision
                with torch.cuda.amp.autocast():
                    generated_imgs = generator(bw_imgs)
                    l1 = l1_loss(generated_imgs, color_imgs)
                    perc = perceptual_loss(generated_imgs, color_imgs)
                    color = color_hist_loss(generated_imgs.float(), color_imgs.float())

                    total_loss = (l1 + 0.1 * perc + 0.05 * color) / accumulation_steps

                # Backward pass with gradient accumulation
                scaler.scale(total_loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Logging
                total_loss_epoch += total_loss.item() * accumulation_steps
                num_batches += 1

                if rank == 0 and i % 100 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                          f"Loss: {total_loss.item() * accumulation_steps:.4f}")

                    if logger:
                        metrics = {
                            'train': {
                                'l1_loss': l1.item(),
                                'perceptual_loss': perc.item(),
                                'color_hist_loss': color.item(),
                                'total_loss': total_loss.item() * accumulation_steps
                            }
                        }
                        logger.log_metrics(i + epoch * len(train_loader), epoch, metrics)

                # Clear some memory
                del generated_imgs, l1, perc, color, total_loss

            # End of epoch processing on main process
            if rank == 0:
                avg_loss = total_loss_epoch / num_batches

                val_metrics = validate_model(generator, val_loader, rank, logger, epoch,
                                             i + epoch * len(train_loader))

                scheduler.step(avg_loss)

                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': generator.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'val_metrics': val_metrics,
                }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                    _use_new_zipfile_serialization=True)

                # Clean up old checkpoints
                old_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))[:-2]
                for ckpt in old_checkpoints:
                    ckpt.unlink()

                print(f"Completed epoch {epoch}, Average loss: {avg_loss:.4f}")

            # Clear cache at end of epoch
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        raise
    finally:
        cleanup_ddp()
        if logger:
            logger.close()


def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs!")

    import torch.multiprocessing as mp
    mp.spawn(
        train_model_ddp,
        args=(world_size,),
        nprocs=world_size,
    )


if __name__ == "__main__":
    main()