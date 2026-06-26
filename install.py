import launch


if not launch.is_installed("transformers"):
    launch.run_pip("install transformers", "requirements for SS Stereoscope")

if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python", "requirements for SS Stereoscope")
