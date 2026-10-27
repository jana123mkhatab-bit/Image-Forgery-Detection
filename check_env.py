"""Run this on your laptop: python check_env.py"""
import importlib
import platform
import shutil
import sys

print("python", sys.version.split()[0], "|", platform.platform())
try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu:", p.name, f"{p.total_memory / 1e9:.1f} GB")
    else:
        print("-> CPU-only torch. Reinstall the CUDA build from https://pytorch.org")
except ImportError:
    print("torch NOT installed")
for m in ["segmentation_models_pytorch", "timm", "datasets", "cv2", "sklearn", "PIL", "yaml", "gradio", "open_clip", "diffusers"]:
    try:
        importlib.import_module(m); print("ok     ", m)
    except ImportError:
        print("MISSING", m)
print(f"free disk here: {shutil.disk_usage('.').free / 1e9:.0f} GB")
