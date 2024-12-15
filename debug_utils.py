import torch
import gc
import logging
from typing import Optional, Dict
from collections import defaultdict
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class MemoryTracker:
    def __init__(self, rank: int, log_dir: Optional[str] = None):
        self.rank = rank
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.memory_stats = defaultdict(list)
        self.shape_stats = defaultdict(list)
        self.start_time = time.time()

        if self.log_dir:
            self.log_file = self.log_dir / f"memory_stats_rank{rank}.json"

    def log_memory(self, location: str, extra_info: Optional[Dict] = None):
        """Log GPU memory usage at a specific location"""
        if not torch.cuda.is_available():
            return

        # Get memory stats
        allocated = torch.cuda.memory_allocated(self.rank) / 1024 ** 3  # GB
        reserved = torch.cuda.memory_reserved(self.rank) / 1024 ** 3  # GB
        max_allocated = torch.cuda.max_memory_allocated(self.rank) / 1024 ** 3

        # Get cached memory
        cached = torch.cuda.memory_reserved(self.rank) - torch.cuda.memory_allocated(self.rank)
        cached = cached / 1024 ** 3  # GB

        stats = {
            'location': location,
            'time': time.time() - self.start_time,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'cached_gb': cached,
            'max_allocated_gb': max_allocated
        }

        if extra_info:
            stats.update(extra_info)

        self.memory_stats[location].append(stats)

        # Log to console
        logger.info(
            f"[Rank {self.rank}] {location} - "
            f"Allocated: {allocated:.2f}GB, "
            f"Reserved: {reserved:.2f}GB, "
            f"Cached: {cached:.2f}GB, "
            f"Max: {max_allocated:.2f}GB"
        )

        # Save to file if directory provided
        if self.log_dir:
            with open(self.log_file, 'w') as f:
                json.dump(self.memory_stats, f, indent=2)

    def log_tensor(self, name: str, tensor: torch.Tensor):
        """Log tensor shape and memory usage"""
        shape = list(tensor.shape)
        memory = tensor.element_size() * tensor.nelement() / 1024 ** 3  # GB

        info = {
            'shape': shape,
            'memory_gb': memory,
            'dtype': str(tensor.dtype),
            'device': str(tensor.device)
        }

        self.shape_stats[name].append(info)

        logger.info(
            f"[Rank {self.rank}] Tensor {name}: "
            f"Shape {shape}, "
            f"Memory: {memory:.3f}GB, "
            f"Type: {tensor.dtype}"
        )

    def report_memory_leak(self):
        """Check for potential memory leaks"""
        gc.collect()
        torch.cuda.empty_cache()

        current_allocated = torch.cuda.memory_allocated(self.rank)
        if current_allocated > 0:
            logger.warning(
                f"[Rank {self.rank}] Potential memory leak detected. "
                f"Still allocated: {current_allocated / 1024 ** 3:.2f}GB"
            )

            # Get tensor statistics
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        logger.info(
                            f"Tensor of size {obj.size()} and type {obj.dtype} "
                            f"on device {obj.device} found"
                        )
                except:
                    pass

    def reset_peak_memory(self):
        """Reset peak memory stats"""
        torch.cuda.reset_peak_memory_stats(self.rank)

    def clear_cache(self):
        """Clear CUDA cache"""
        gc.collect()
        torch.cuda.empty_cache()

    def save_final_report(self):
        """Save final memory usage report"""
        if self.log_dir:
            report_file = self.log_dir / f"memory_report_rank{self.rank}.json"
            report = {
                'memory_stats': self.memory_stats,
                'shape_stats': self.shape_stats,
                'final_memory': {
                    'allocated': torch.cuda.memory_allocated(self.rank) / 1024 ** 3,
                    'reserved': torch.cuda.memory_reserved(self.rank) / 1024 ** 3,
                    'max_allocated': torch.cuda.max_memory_allocated(self.rank) / 1024 ** 3
                }
            }
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)