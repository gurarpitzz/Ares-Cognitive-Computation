"""Vision Data Loaders, Augmentations, Face Crop Dataset & Balanced Sampler.

Operationalizes:
- Frame-level dataset over face crops with video-level grouping.
- Class-balanced sampling for real vs fake videos.
- Video-level grouping and 1-2 FPS frame sampling.
- Data augmentations: horizontal flip, color jitter, Gaussian blur, JPEG re-compression, random erasing.
- Batch sample visualization (figures/batch_sample.png).
"""

from dataclasses import dataclass
import io
import random
from typing import Optional, Callable
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T
import torchvision.utils as vutils


class SimulatedJPEGRecompression:
    """Simulate compression artifacts by re-compressing PIL image to JPEG in-memory."""

    def __init__(self, min_quality: int = 30, max_quality: int = 100, p: float = 0.5) -> None:
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.min_quality, self.max_quality)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Pre-registered training augmentations for deepfake face crops."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        SimulatedJPEGRecompression(min_quality=30, max_quality=100, p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])


def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """Evaluation transforms: deterministic resize and normalization, no augmentation."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@dataclass
class VideoMetadata:
    video_id: str
    label: int  # 0 = real, 1 = fake
    split: str  # 'train', 'val', 'test'
    dataset: str  # 'ffpp', 'celebdf'
    frame_files: list[str]
    manipulation_type: str = "unknown"  # 'original', 'deepfakes', 'face2face', 'faceswap', 'neuraltextures', 'celeb_synthesis'


class FaceFrameDataset(Dataset):
    """Dataset serving individual face crop frames with video metadata."""

    def __init__(
        self,
        samples: list[dict],  # dict with 'image_path' (or PIL image), 'label', 'video_id', 'frame_idx'
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform or get_eval_transforms()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        item = self.samples[idx]
        img_src = item["image"]
        if isinstance(img_src, str):
            image = Image.open(img_src).convert("RGB")
        elif isinstance(img_src, Image.Image):
            image = img_src.convert("RGB")
        elif isinstance(img_src, np.ndarray):
            image = Image.fromarray(img_src).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(img_src)}")

        tensor = self.transform(image)
        label = torch.tensor(item["label"], dtype=torch.float32)
        video_id = item["video_id"]
        frame_idx = item.get("frame_idx", 0)
        return tensor, label, video_id, frame_idx


class BalancedClassBatchSampler(Sampler):
    """Batch sampler ensuring 50/50 balance between real (0) and fake (1) per batch.
    
    Addresses the ~4:1 fake:real imbalance in FaceForensics++ training data.
    """

    def __init__(self, labels: list[int], batch_size: int = 32, seed: int = 42) -> None:
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.rng = random.Random(seed)
        
        self.real_indices = [i for i, y in enumerate(labels) if y == 0]
        self.fake_indices = [i for i, y in enumerate(labels) if y == 1]
        
        if len(self.real_indices) == 0 or len(self.fake_indices) == 0:
            raise ValueError("Dataset must contain both real and fake samples for balanced sampling.")
            
        self.num_batches = max(len(self.real_indices), len(self.fake_indices)) // self.half_batch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        real_pool = list(self.real_indices)
        fake_pool = list(self.fake_indices)
        self.rng.shuffle(real_pool)
        self.rng.shuffle(fake_pool)
        
        r_ptr = 0
        f_ptr = 0
        for _ in range(self.num_batches):
            batch = []
            # Sample half real
            for _ in range(self.half_batch):
                if r_ptr >= len(real_pool):
                    self.rng.shuffle(real_pool)
                    r_ptr = 0
                batch.append(real_pool[r_ptr])
                r_ptr += 1
            # Sample half fake
            for _ in range(self.half_batch):
                if f_ptr >= len(fake_pool):
                    self.rng.shuffle(fake_pool)
                    f_ptr = 0
                batch.append(fake_pool[f_ptr])
                f_ptr += 1
                
            self.rng.shuffle(batch)
            yield batch


def visualize_batch_sample(
    dataloader,
    output_path: Path,
    n_images: int = 16,
) -> None:
    """Generate figures/batch_sample.png visualizing an augmented training batch."""
    for images, labels, video_ids, _ in dataloader:
        images = images[:n_images]
        # Denormalize for visualization: x * std + mean
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm = torch.clamp(images * std + mean, 0.0, 1.0)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vutils.save_image(denorm, str(output_path), nrow=4, padding=4)
        break
