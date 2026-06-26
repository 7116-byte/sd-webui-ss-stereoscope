from __future__ import annotations

import gc
import traceback
from dataclasses import dataclass

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import devices, scripts, shared

try:
    import cv2
except Exception:
    cv2 = None

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except Exception:
    AutoImageProcessor = None
    AutoModelForDepthEstimation = None


TITLE = "SS Stereoscope"


MODEL_IDS = {
    "Depth Anything V2 - Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth Anything V2 - Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth Anything V2 - Large": "depth-anything/Depth-Anything-V2-Large-hf",
    "Depth Anything V3 - Small": "depth-anything/DA3-Small",
    "Depth Anything V3 - Base": "depth-anything/DA3-Base",
    "Depth Anything V3 - Large": "depth-anything/DA3-Large",
}


def _resampling_nearest():
    return getattr(getattr(Image, "Resampling", Image), "NEAREST")


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    if depth_max - depth_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - depth_min) / (depth_max - depth_min)


def _gradient_depth(width: int, height: int) -> np.ndarray:
    row = np.linspace(1.0, 0.0, width, dtype=np.float32)
    return np.tile(row, (height, 1))


def _blur_depth(depth: np.ndarray, blur_radius: int) -> np.ndarray:
    if blur_radius <= 0:
        return depth

    radius = int(blur_radius)
    kernel_radius = max(1, radius) // 2
    kernel_size = kernel_radius * 2 + 1

    if cv2 is not None:
        return cv2.blur(depth, (kernel_size, kernel_size))

    padded = np.pad(depth, kernel_radius, mode="edge")
    output = np.zeros_like(depth, dtype=np.float32)
    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            output[y, x] = np.mean(padded[y : y + kernel_size, x : x + kernel_size])
    return output


class DepthEstimator:
    def __init__(self):
        self.model_name = None
        self.processor = None
        self.model = None
        self.device = devices.device

    def load(self, model_name: str):
        if self.model_name == model_name and self.model is not None and self.processor is not None:
            return

        self.unload()

        if AutoImageProcessor is None or AutoModelForDepthEstimation is None:
            raise RuntimeError("transformers depth-estimation classes are unavailable")

        model_id = MODEL_IDS[model_name]
        print(f"[SS Stereoscope] Loading depth model: {model_id}")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name

    def unload(self):
        self.processor = None
        self.model = None
        self.model_name = None
        gc.collect()
        devices.torch_gc()

    def predict(self, image: Image.Image, model_name: str, blur_radius: int) -> np.ndarray:
        width, height = image.size

        try:
            self.load(model_name)
            assert self.processor is not None
            assert self.model is not None

            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth = prediction.detach().float().cpu().numpy()
            depth = 1.0 - _normalize_depth(depth)
            return _blur_depth(depth, blur_radius)
        except Exception:
            print("[SS Stereoscope] Depth model failed; using fallback gradient depth.")
            print(traceback.format_exc())
            return _blur_depth(_gradient_depth(width, height), blur_radius)


@dataclass
class StereoscopeResult:
    sbs: Image.Image
    depth: Image.Image


class StereoscopeEngine:
    def __init__(self):
        self.depth_estimator = DepthEstimator()

    def create_sbs(
        self,
        image: Image.Image,
        depth_scale: float,
        blur_radius: int,
        fill_radius: int,
        invert_depth: bool,
        mode: str,
        model_name: str,
    ) -> StereoscopeResult:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size

        depth = self.depth_estimator.predict(rgb_image, model_name, blur_radius)
        if invert_depth:
            depth = 1.0 - depth

        depth_img = Image.fromarray(np.clip(depth * 255.0, 0, 255).astype(np.uint8), mode="L")
        depth_img = depth_img.resize((width, height), _resampling_nearest())

        img_array = np.array(rgb_image, dtype=np.uint8)
        depth_array = np.array(depth_img, dtype=np.float32)

        sbs_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        sbs_image[:, :width] = img_array
        sbs_image[:, width:] = img_array

        # Original ComfyUI_SSStereoscope scaling: depth values are 0..255.
        depth_scaling = float(depth_scale) / float(width)
        pixel_shifts = (depth_array * depth_scaling).astype(np.int32)
        pixel_shifts = np.clip(pixel_shifts, 0, max(width - 1, 0))

        target_eye_offset = 0 if mode == "Parallel" else width
        fill_radius = int(np.clip(fill_radius, 1, 32))

        for x in range(width - 1, -1, -1):
            source_pixels = img_array[:, x, :]
            shifts = pixel_shifts[:, x]
            target_x = x + shifts

            for fill_offset in range(fill_radius):
                fill_x = target_x + fill_offset
                valid_mask = (fill_x >= 0) & (fill_x < width)
                if not np.any(valid_mask):
                    continue

                valid_rows = np.where(valid_mask)[0]
                valid_x = fill_x[valid_mask]
                sbs_image[valid_rows, valid_x + target_eye_offset] = source_pixels[valid_rows]

        depth_rgb = Image.merge("RGB", (depth_img, depth_img, depth_img))
        return StereoscopeResult(Image.fromarray(sbs_image, mode="RGB"), depth_rgb)


ENGINE = StereoscopeEngine()


class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self._image_index = 0

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    def ui(self, is_img2img):
        with gr.Accordion(TITLE, open=False):
            enabled = gr.Checkbox(False, label="Enable")
            model_name = gr.Dropdown(
                choices=list(MODEL_IDS.keys()),
                value="Depth Anything V2 - Small",
                label="Depth model",
            )
            mode = gr.Radio(["Parallel", "Cross-eyed"], value="Parallel", label="Viewing mode")
            depth_scale = gr.Slider(1.0, 100.0, value=40.0, step=0.5, label="Depth scale")
            blur_radius = gr.Slider(0, 51, value=3, step=1, label="Depth blur radius")
            fill_radius = gr.Slider(1, 32, value=10, step=1, label="Edge fill radius")
            invert_depth = gr.Checkbox(False, label="Invert depth")
            save_depth = gr.Checkbox(True, label="Save depth map")
            unload_model = gr.Checkbox(False, label="Unload depth model after each image")

        self.infotext_fields = [
            (enabled, "SS Stereoscope enabled"),
            (model_name, "SS Stereoscope model"),
            (mode, "SS Stereoscope mode"),
            (depth_scale, "SS Stereoscope depth scale"),
            (blur_radius, "SS Stereoscope blur radius"),
            (fill_radius, "SS Stereoscope fill radius"),
            (invert_depth, "SS Stereoscope invert depth"),
        ]

        return [enabled, model_name, mode, depth_scale, blur_radius, fill_radius, invert_depth, save_depth, unload_model]

    def before_process(
        self,
        p,
        enabled,
        model_name,
        mode,
        depth_scale,
        blur_radius,
        fill_radius,
        invert_depth,
        save_depth,
        unload_model,
    ):
        self._image_index = 0
        if not enabled:
            return

        p.extra_generation_params["SS Stereoscope"] = True
        p.extra_generation_params["SS Stereoscope model"] = model_name
        p.extra_generation_params["SS Stereoscope mode"] = mode
        p.extra_generation_params["SS Stereoscope depth scale"] = depth_scale
        p.extra_generation_params["SS Stereoscope blur radius"] = blur_radius
        p.extra_generation_params["SS Stereoscope fill radius"] = fill_radius
        p.extra_generation_params["SS Stereoscope invert depth"] = invert_depth

    def postprocess_image(
        self,
        p,
        pp: scripts.PostprocessImageArgs,
        enabled,
        model_name,
        mode,
        depth_scale,
        blur_radius,
        fill_radius,
        invert_depth,
        save_depth,
        unload_model,
    ):
        if not enabled:
            return

        index = self._image_index
        self._image_index += 1

        print(f"[SS Stereoscope] Processing txt2img output #{index + 1}")
        result = ENGINE.create_sbs(
            pp.image,
            depth_scale=depth_scale,
            blur_radius=int(blur_radius),
            fill_radius=int(fill_radius),
            invert_depth=bool(invert_depth),
            mode=mode,
            model_name=model_name,
        )

        if save_depth:
            from modules import images

            seeds = getattr(p, "seeds", None) or [getattr(p, "seed", None)]
            prompts = getattr(p, "prompts", None) or [getattr(p, "prompt", None)]
            seed = seeds[min(index, len(seeds) - 1)] if seeds else None
            prompt = prompts[min(index, len(prompts) - 1)] if prompts else None
            images.save_image(
                result.depth,
                p.outpath_samples,
                "",
                seed=seed,
                prompt=prompt,
                extension=shared.opts.samples_format,
                info=getattr(p, "infotext", lambda *_args, **_kwargs: None)(index) if hasattr(p, "infotext") else None,
                p=p,
                suffix="-ss-depth",
            )

        pp.image = result.sbs

        if unload_model:
            ENGINE.depth_estimator.unload()
