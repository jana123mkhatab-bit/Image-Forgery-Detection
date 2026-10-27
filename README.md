# Image Forgery Detection

Detect whether an image is authentic, edited (Photoshop-style splice / copy-move)
or AI-generated, and localize the edited pixels.

- `baselines.py`, `run_baselines.py`: classical methods (ELA, noise, JPEG ghost, copy-move)
- `forgery_synth.py`: on-the-fly splice / copy-move / degradation generator
- `data_stream.py`: streams datasets from the Hugging Face Hub (nothing stored on disk)
- `model.py`: RGB + SRM-residual input, ResNet U-Net, mask head + 3-class head
- `train.py`: mixed-precision training with automatic resume
- `colab_cells.md`: the Colab launcher

## Laptop
```
python -m venv .venv && .venv\Scripts\activate
# install the CUDA build of torch from pytorch.org, then:
pip install -r requirements.txt
python check_env.py
```

## Colab
See `colab_cells.md`.
