from __future__ import annotations

import gc
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import devices, scripts, shared

try:
    import cv2
except Exception:
    cv2 = None

EXTENSION_DIR = Path(__file__).resolve().parents[1]
if str(EXTENSION_DIR) not in sys.path:
    sys.path.insert(0, str(EXTENSION_DIR))

from ss_stereoscope_depth.direct_depth import DirectDepthAnythingV2, MODEL_SPECS


TITLE = "SS Stereoscope"
MODEL_IDS = list(MODEL_SPECS.keys())


def _resampling_nearest():
    return getattr(getattr(Image, "Resampling", Image), "NEAREST")


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    if depth_max - depth_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - depth_min) / (depth_max - depth_min)


class DepthModelError(RuntimeError):
    pass


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
        self.device = devices.device
        self.backend = DirectDepthAnythingV2(
            device=self.device,
            models_dir=Path(shared.models_path),
            input_size=518,
        )

    def load(self, model_name: str):
        if self.model_name == model_name:
            return

        self.backend.load(model_name)
        self.model_name = model_name

    def unload(self):
        self.backend.unload()
        self.model_name = None
        gc.collect()
        devices.torch_gc()

    def predict(self, image: Image.Image, model_name: str, blur_radius: int) -> np.ndarray:
        try:
            self.load(model_name)
            depth = self.backend.predict(np.array(image.convert("RGB")), model_name)
            depth = 1.0 - _normalize_depth(depth)
            return _blur_depth(depth, blur_radius)
        except DepthModelError:
            raise
        except Exception as exc:
            print("[SS Stereoscope] Depth model failed.")
            print(traceback.format_exc())
            raise DepthModelError("Depth model failed; see console for traceback.") from exc


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
        sbs = Image.fromarray(sbs_image, mode="RGB")
        sbs.info.update(image.info)
        depth_rgb.info.update(image.info)
        return StereoscopeResult(sbs, depth_rgb)


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
                choices=MODEL_IDS,
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
        try:
            result = ENGINE.create_sbs(
                pp.image,
                depth_scale=depth_scale,
                blur_radius=int(blur_radius),
                fill_radius=int(fill_radius),
                invert_depth=bool(invert_depth),
                mode=mode,
                model_name=model_name,
            )
        except DepthModelError as exc:
            print(f"[SS Stereoscope] Skipped: {exc}")
            return

        infotext = self.create_infotext(p, index)
        if infotext:
            result.sbs.info["parameters"] = infotext
            result.depth.info["parameters"] = infotext

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
                info=infotext,
                p=p,
                existing_info=result.depth.info,
                suffix="-ss-depth",
            )

        pp.image = result.sbs

        if unload_model:
            ENGINE.depth_estimator.unload()

    @staticmethod
    def create_infotext(p, index: int) -> str | None:
        try:
            from modules.processing import create_infotext

            return create_infotext(
                p,
                p.prompts,
                p.seeds,
                p.subseeds,
                use_main_prompt=False,
                index=index,
                all_negative_prompts=p.negative_prompts,
            )
        except Exception:
            print("[SS Stereoscope] Failed to create infotext for depth map.")
            print(traceback.format_exc())
            return None
