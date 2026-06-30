import os
import random
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


class TumorClassificationDataset(Dataset):
    def __init__(self, filepaths: List[str], labels: List[int], transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class TumorSegmentationDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str], image_size: Tuple[int, int] = (256, 256), augment: bool = False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = get_segmentation_image(self.image_paths[idx], self.image_size)
        mask = get_segmentation_mask(self.mask_paths[idx], self.image_size)

        if self.augment:
            image, mask = self.random_flip(image, mask)
            image, mask = self.random_rotate(image, mask)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return image_tensor, mask_tensor

    def random_flip(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        return image, mask

    def random_rotate(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            image = np.rot90(image, k=angle // 90).copy()
            mask = np.rot90(mask, k=angle // 90).copy()
        return image, mask


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def list_image_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if is_image_file(filename):
                files.append(os.path.join(root, filename))
    return sorted(files)


def get_classification_transforms(image_size: Tuple[int, int] = (224, 224), mode: str = "train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_segmentation_image(image_path: str, image_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


def get_segmentation_mask(mask_path: str, image_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return mask[..., np.newaxis]


def split_paths(paths: List[str], labels: List[int], val_ratio: float = 0.2, seed: int = 42):
    combined = list(zip(paths, labels))
    random.Random(seed).shuffle(combined)
    split = int(len(combined) * (1 - val_ratio))
    train = combined[:split]
    val = combined[split:]
    return ([p for p, _ in train], [l for _, l in train], [p for p, _ in val], [l for _, l in val])


def build_classification_datasets(
    root_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    val_ratio: float = 0.2,
    seed: int = 42,
):
    classes = sorted([name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))])
    filepaths: List[str] = []
    labels: List[int] = []

    for label_index, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        for filepath in list_image_files(class_dir):
            filepaths.append(filepath)
            labels.append(label_index)

    train_paths, train_labels, val_paths, val_labels = split_paths(filepaths, labels, val_ratio=val_ratio, seed=seed)

    train_dataset = TumorClassificationDataset(train_paths, train_labels, transform=get_classification_transforms(image_size, "train"))
    val_dataset = TumorClassificationDataset(val_paths, val_labels, transform=get_classification_transforms(image_size, "val"))
    return train_dataset, val_dataset, classes


def pair_images_and_masks(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    images = sorted([p for p in list_image_files(image_dir)])
    masks = sorted([p for p in list_image_files(mask_dir)])
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}
    pairs: List[Tuple[str, str]] = []

    for image_path in images:
        base = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = mask_map.get(base)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


def build_segmentation_datasets(
    image_dir: str,
    mask_dir: str,
    image_size: Tuple[int, int] = (256, 256),
    val_ratio: float = 0.2,
    seed: int = 42,
):
    pairs = pair_images_and_masks(image_dir, mask_dir)
    if len(pairs) == 0:
        raise ValueError("No image-mask pairs found. Check image_dir and mask_dir paths.")

    random.Random(seed).shuffle(pairs)
    split = int(len(pairs) * (1 - val_ratio))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    train_dataset = TumorSegmentationDataset([x for x, _ in train_pairs], [y for _, y in train_pairs], image_size=image_size, augment=True)
    val_dataset = TumorSegmentationDataset([x for x, _ in val_pairs], [y for _, y in val_pairs], image_size=image_size, augment=False)
    return train_dataset, val_dataset
