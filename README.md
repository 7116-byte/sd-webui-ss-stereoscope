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

## Notes

Depth Anything V2 needs a newer `transformers` release than some WebUI bundles ship with. This extension does not auto-upgrade WebUI's shared Python dependencies because doing so can break gallery and PNG infotext features. If the bundled dependency is too old, depth generation is skipped and the console explains why.

The extension keeps standard WebUI PNG infotext clean, so generated SBS images can still be sent back to txt2img from gallery tools.

## Attribution

This extension ports the txt2img-friendly parts of the SBS workflow inspired by `SamSeenX/ComfyUI_SSStereoscope`, with defaults oriented toward parallel-eye output.

It also borrows practical parameter defaults from `7116-byte/ParallelVrGallery`.
