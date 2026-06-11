import argparse
import os
import sys
from typing import Dict, List

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from preprocessing import build_classification_datasets, build_segmentation_datasets, TumorClassificationDataset, TumorSegmentationDataset
from train import UNet, get_classifier


def dice_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = torch.sigmoid(preds)
    preds = (preds >= threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item()
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def evaluate_classification(model, dataloader):
    model.eval()
    predictions: List[int] = []
    references: List[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
            predictions.extend(preds)
            references.extend(labels.cpu().numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(references, predictions),
        "precision": precision_score(references, predictions, average="binary", zero_division=0),
        "recall": recall_score(references, predictions, average="binary", zero_division=0),
        "f1_score": f1_score(references, predictions, average="binary", zero_division=0),
    }
    return metrics


def evaluate_segmentation(model, dataloader):
    model.eval()
    dice_scores: List[float] = []
    losses: List[float] = []

    with torch.no_grad():
        for inputs, masks in dataloader:
            outputs = model(inputs)
            dice_scores.append(dice_score(outputs, masks))
            losses.append(F.binary_cross_entropy_with_logits(outputs, masks).item())

    return {
        "dice": float(np.mean(dice_scores)),
        "bce_loss": float(np.mean(losses)),
    }


def print_metrics(metrics: Dict[str, float]):
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def load_checkpoint(path: str, task: str, device: torch.device, num_classes: int = 2):
    checkpoint = torch.load(path, map_location=device)
    if task == "classification":
        model = get_classifier(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = UNet(in_channels=3, out_channels=1)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate classification or segmentation models.")
    parser.add_argument("--task", choices=["classification", "segmentation"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/classification")
    parser.add_argument("--image-dir", type=str, default="data/segmentation/images")
    parser.add_argument("--mask-dir", type=str, default="data/segmentation/masks")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "classification":
        train_dataset, val_dataset, classes = build_classification_datasets(
            args.data_dir,
            image_size=(args.image_size, args.image_size),
            val_ratio=args.val_split,
            seed=args.seed,
        )
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        model = load_checkpoint(args.model_path, task="classification", device=device, num_classes=len(classes))
        metrics = evaluate_classification(model, dataloader)
        print("Classification evaluation")
        print_metrics(metrics)

    else:
        _, val_dataset = build_segmentation_datasets(
            args.image_dir,
            args.mask_dir,
            image_size=(args.image_size, args.image_size),
            val_ratio=args.val_split,
            seed=args.seed,
        )
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        model = load_checkpoint(args.model_path, task="segmentation", device=device)
        metrics = evaluate_segmentation(model, dataloader)
        print("Segmentation evaluation")
        print_metrics(metrics)


if __name__ == "__main__":
    main()
