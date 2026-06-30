import argparse
import os
import sys
from typing import List

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessing import build_classification_datasets
from train import get_classifier
from evaluate import evaluate_classification


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    noisy = image + np.random.normal(scale=sigma, size=image.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy


def apply_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    blurred = cv2.GaussianBlur((image * 255).astype(np.uint8), (kernel_size, kernel_size), 0)
    return blurred.astype(np.float32) / 255.0


def apply_jpeg_compression(image: np.ndarray, quality: int = 20) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode('.jpg', (image * 255).astype(np.uint8), encode_param)
    if not success:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return decoded.astype(np.float32) / 255.0


def corrupt_image(image: np.ndarray, corruption_type: str, severity: float) -> np.ndarray:
    if corruption_type == "noise":
        return add_gaussian_noise(image, sigma=severity)
    if corruption_type == "blur":
        kernel_size = max(3, int(severity * 25) | 1)
        return apply_blur(image, kernel_size=kernel_size)
    if corruption_type == "jpeg":
        quality = max(10, int((1 - severity) * 90))
        return apply_jpeg_compression(image, quality=quality)
    return image


def apply_corruption_to_batch(batch, corruption_type: str, severity: float):
    images, labels = batch
    corrupted_images = []
    for image in images:
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        image = np.clip(image, 0.0, 1.0)
        corrupted = corrupt_image(image, corruption_type, severity)
        normalized = (corrupted - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        corrupted_images.append(torch.from_numpy(normalized.transpose(2, 0, 1)).float())
    return torch.stack(corrupted_images), labels


def evaluate_robustness(model, dataloader, corruption_types: List[str], severities: List[float], device: torch.device):
    results = {}
    model.to(device)
    for corruption in corruption_types:
        results[corruption] = {}
        for severity in severities:
            metrics_list = []
            for batch in dataloader:
                inputs, labels = apply_corruption_to_batch(batch, corruption, severity)
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    metrics_list.append((preds.cpu(), labels.cpu()))

            predictions = []
            references = []
            for preds, labels in metrics_list:
                predictions.extend(preds.numpy().tolist())
                references.extend(labels.numpy().tolist())
            accuracy = np.mean(np.array(predictions) == np.array(references))
            results[corruption][severity] = accuracy
            print(f"Corruption={corruption}, severity={severity:.3f}, accuracy={accuracy:.4f}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run robustness evaluation on classification models.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/classification")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_dataset, classes = build_classification_datasets(
        args.data_dir,
        image_size=(args.image_size, args.image_size),
        val_ratio=args.val_split,
        seed=args.seed,
    )
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    checkpoint = torch.load(args.model_path, map_location=device)
    model = get_classifier(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    corruption_types = ["noise", "blur", "jpeg"]
    severities = [0.02, 0.05, 0.1]
    results = evaluate_robustness(model, dataloader, corruption_types, severities, device)
    print("Robustness summary:")
    for corruption, severity_map in results.items():
        for severity, accuracy in severity_map.items():
            print(f"{corruption} @ {severity:.2f} -> accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
