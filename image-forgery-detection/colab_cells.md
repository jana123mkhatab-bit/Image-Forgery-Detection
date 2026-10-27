# Colab launcher (paste each block into its own cell)

Runtime > Change runtime type > GPU (T4). The notebook holds no logic: it
clones the repo, installs requirements, and runs `train.py`.

## Cell 1: mount Drive (checkpoints and logs live there)
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2: get the code (re-run to update)
```python
import os
if not os.path.exists('/content/Image-Forgery-Detection'):
    !git clone https://github.com/jana123mkhatab-bit/Image-Forgery-Detection.git /content/Image-Forgery-Detection
%cd /content/Image-Forgery-Detection
!git pull
!pip install -q -r requirements.txt
```

## Cell 3: check the GPU
```python
!python check_env.py
```

## Cell 4: train (resumes automatically after a disconnect)
```python
!python train.py --config configs/default.yaml \
    --out_dir /content/drive/MyDrive/forgery/runs/baseline_r34_srm
```

Notes:
- Datasets are streamed from the Hugging Face Hub; nothing is downloaded up front.
- Pretrained weights (ResNet, CLIP, Stable Diffusion) are fetched by the Python
  libraries into Colab's temporary disk each session.
- If a dataset needs a Hugging Face login, run `from huggingface_hub import notebook_login; notebook_login()`.
