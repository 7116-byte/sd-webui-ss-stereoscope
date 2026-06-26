from importlib.metadata import PackageNotFoundError, version

import launch


def parse_version(value: str) -> tuple[int, ...]:
    parts = []
    for part in value.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


try:
    transformers_version = version("transformers")
except PackageNotFoundError:
    transformers_version = "0"

if parse_version(transformers_version) < (4, 51, 0) or parse_version(transformers_version) >= (5, 0, 0):
    launch.run_pip("install -U \"transformers>=4.51.0,<5\"", "requirements for SS Stereoscope")

if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python", "requirements for SS Stereoscope")
