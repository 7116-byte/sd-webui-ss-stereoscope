# SD WebUI SS Stereoscope

Txt2img postprocess extension for AUTOMATIC1111 WebUI. It generates a Depth Anything map for each txt2img result and replaces the output with a parallel-eye SBS image.

## Usage

1. Restart WebUI after installing or editing this extension.
2. Open txt2img.
3. Expand `SS Stereoscope`.
4. Enable it and generate an image.

The final txt2img image becomes SBS output. If `Save depth map` is enabled, the depth image is also saved with the `-ss-depth` suffix.

## Defaults

- Viewing mode: `Parallel`
- Depth scale: `40`
- Depth blur radius: `3`
- Edge fill radius: `10`
- Depth model: `Depth Anything V2 - Small`
- Depth device: `CPU (low VRAM)`

## Notes

Depth Anything V2 is vendored as a direct torch backend. The extension does not auto-upgrade WebUI's shared `transformers` dependency because doing so can break gallery and PNG infotext features.

Depth weights are stored in `models/depthanything` and download automatically on first use. If Hugging Face is slow, download `depth_anything_v2_vits.pth` manually from `depth-anything/Depth-Anything-V2-Small` and place it in that folder.

The default depth device is CPU to avoid competing with the loaded Stable Diffusion model for VRAM. Use `CUDA (fast)` only when there is enough free VRAM, or `Auto` to try CUDA and fall back to CPU on out-of-memory.

The extension keeps standard WebUI PNG infotext clean, so generated SBS images can still be sent back to txt2img from gallery tools.

## Attribution

This extension ports the txt2img-friendly parts of the SBS workflow inspired by `SamSeenX/ComfyUI_SSStereoscope`, with defaults oriented toward parallel-eye output.

It also borrows practical parameter defaults from `7116-byte/ParallelVrGallery`.
