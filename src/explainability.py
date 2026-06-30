import os
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, inputs: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        outputs = self.model(inputs)
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()

        self.model.zero_grad()
        target = outputs[0, class_idx]
        target.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (inputs.shape[-1], inputs.shape[-2]))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = heatmap_color.astype(np.float32) * alpha + image.astype(np.float32) * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return image


def enable_mc_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predict_with_uncertainty(model: torch.nn.Module, inputs: torch.Tensor, n_samples: int = 20) -> dict:
    model.eval()
    enable_mc_dropout(model)
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            predictions.append(probs)

    predictions = np.stack(predictions, axis=0)
    mean_prob = predictions.mean(axis=0)[0]
    uncertainty = predictions.std(axis=0).mean()
    predicted_class = int(mean_prob.argmax())
    return {
        "predicted_class": predicted_class,
        "mean_probability": mean_prob.tolist(),
        "uncertainty": float(uncertainty),
    }


def save_heatmap(output_dir: str, filename: str, image: np.ndarray, heatmap: np.ndarray) -> str:
    os.makedirs(output_dir, exist_ok=True)
    overlay = overlay_heatmap(image, heatmap)
    result_path = os.path.join(output_dir, filename)
    Image.fromarray(overlay).save(result_path)
    return result_path
    return result_path
