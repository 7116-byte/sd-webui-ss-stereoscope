from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .depth_anything_v2.dpt import DepthAnythingV2


@dataclass(frozen=True)
class DepthModelSpec:
    repo_id: str
    filename: str
    encoder: str
    features: int
    out_channels: list[int]


MODEL_SPECS = {
    "Depth Anything V2 - Small": DepthModelSpec(
        repo_id="depth-anything/Depth-Anything-V2-Small",
        filename="depth_anything_v2_vits.pth",
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
    ),
    "Depth Anything V2 - Base": DepthModelSpec(
        repo_id="depth-anything/Depth-Anything-V2-Base",
        filename="depth_anything_v2_vitb.pth",
        encoder="vitb",
        features=128,
        out_channels=[96, 192, 384, 768],
    ),
    "Depth Anything V2 - Large": DepthModelSpec(
        repo_id="depth-anything/Depth-Anything-V2-Large",
        filename="depth_anything_v2_vitl.pth",
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    ),
}


def ensure_model_file(spec: DepthModelSpec, models_dir: Path) -> Path:
    model_dir = models_dir / "depthanything"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / spec.filename
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    endpoints = []
    configured_endpoint = os.environ.get("HF_ENDPOINT")
    if configured_endpoint:
        endpoints.append(configured_endpoint)
    endpoints.extend(["https://hf-mirror.com", None])

    errors = []
    for endpoint in endpoints:
        try:
            print(
                "[SS Stereoscope] Downloading depth model "
                f"{spec.filename} from {spec.repo_id}"
                + (f" via {endpoint}" if endpoint else "")
            )
            return Path(
                hf_hub_download(
                    repo_id=spec.repo_id,
                    filename=spec.filename,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                    endpoint=endpoint,
                    etag_timeout=30,
                )
            )
        except Exception as exc:
            errors.append(f"{endpoint or 'default'}: {exc}")

    raise RuntimeError(
        "Failed to download Depth Anything V2 weights. "
        f"Download {spec.filename} manually from https://huggingface.co/{spec.repo_id} "
        f"and place it in {model_dir}. Errors: {' | '.join(errors)}"
    )


def resize_for_depth(image: np.ndarray, input_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    scale = float(input_size) / float(min(height, width))
    resized_h = max(14, int(round(height * scale / 14.0) * 14))
    resized_w = max(14, int(round(width * scale / 14.0) * 14))

    try:
        import cv2

        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    except Exception:
        from PIL import Image

        resized = np.array(Image.fromarray(image).resize((resized_w, resized_h), Image.Resampling.BICUBIC))

    return resized, (height, width)


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


class DirectDepthAnythingV2:
    def __init__(self, device: torch.device, models_dir: Path, input_size: int = 518):
        self.device = device
        self.models_dir = models_dir
        self.input_size = input_size
        self.model_name: str | None = None
        self.model: DepthAnythingV2 | None = None

    def load(self, model_name: str):
        if self.model_name == model_name and self.model is not None:
            return

        self.unload()
        spec = MODEL_SPECS[model_name]
        model_path = ensure_model_file(spec, self.models_dir)

        print(f"[SS Stereoscope] Loading Depth Anything V2 weights: {model_path}")
        model = DepthAnythingV2(
            encoder=spec.encoder,
            features=spec.features,
            out_channels=spec.out_channels,
        )
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)

        self.model_name = model_name
        self.model = model

    def unload(self):
        self.model_name = None
        self.model = None

    def predict(self, image: np.ndarray, model_name: str) -> np.ndarray:
        self.load(model_name)
        assert self.model is not None

        resized, original_size = resize_for_depth(image, self.input_size)
        tensor = image_to_tensor(resized, self.device)

        with torch.inference_mode():
            depth = self.model(tensor)
            depth = F.interpolate(
                depth[:, None],
                size=original_size,
                mode="bilinear",
                align_corners=True,
            )[0, 0]

        return depth.detach().float().cpu().numpy()
