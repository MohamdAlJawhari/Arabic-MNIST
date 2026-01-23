# Arabic-MNIST

Arabic letter classification with a MobileNetV3-Small model, trained on an
ImageFolder-style dataset and served through a Gradio demo that also visualizes
intermediate feature maps.

## 🎥 Demo

You can watch the demo directly from this repository:

[![Watch the demo](https://img.youtube.com/vi/q7U2XbBlIvs/0.jpg)](demo/Arabic-MNIST.mp4)

[▶️ Arabic-MNIST Demo](demo/Arabic-MNIST.mp4)

Or on YouTube:

[📺 Watch on YouTube](https://youtu.be/q7U2XbBlIvs)


## Objective

Build and serve a compact Arabic letter classifier, trained on the OCR-Arabic
dataset from https://github.com/ahmedsaeedsaid/OCR-Arabic/tree/master/dataset.

## What is in src/

- `src/config.py`: Training configuration (paths, image size, padding, epochs, lr).
- `src/data.py`: Dataset loading, train/val split, and torchvision transforms.
- `src/model.py`: MobileNetV3-Small adapted for 1-channel input and custom head.
- `src/train.py`: Training loop, validation, and checkpoint saving.
- `src/infer.py`: Checkpoint loading and preprocessing for inference.
- `src/app.py`: Gradio UI for drawing or uploading a letter and viewing feature maps.
- `src/viz.py`: Helper to render feature maps into a grid image.
- `src/utils.py`: Seeding, device selection, and checkpoint serialization.

## Model and training details (The src/ folder)

### Model (src/model.py)

- Backbone: torchvision `mobilenet_v3_small` (ImageNet pretrained by default).
- Input: 1×128×128 grayscale (see transforms in `src/data.py`). The first conv layer is replaced to accept 1 channel (`model.features[0][0]`); when using pretrained weights, the new conv is initialized by averaging the original RGB filters.
- Feature extractor: `model.features` is a sequence of 13 blocks:
  - `features[0]`: `Conv2dNormActivation` (stem)
  - `features[1..11]`: 11× `InvertedResidual` blocks
  - `features[12]`: `Conv2dNormActivation` (final 1×1 conv to 576 channels)
- Classifier head: global average pooling (`model.avgpool`) + `model.classifier`:
  - Linear(576 → 1024) → Hardswish → Dropout(p=0.2) → Linear(1024 → `num_classes`)
  - `num_classes` is inferred from the `dataset/` folder names (ImageFolder classes).

### Layers / shapes (for a 128×128 input)

The model downsamples spatially while increasing channels. Example output shapes (N=1):

- Input: 1×128×128
- `features[0]`: 16×64×64
- `features[1]`: 16×32×32
- `features[2]`: 24×16×16
- `features[4]`: 40×8×8
- `features[9]`: 96×4×4
- `features[12]`: 576×4×4 → avgpool → 576×1×1 → classifier → `num_classes` logits

### Feature map visualization (src/app.py + src/viz.py)

- The Gradio app runs a forward pass and captures intermediate activations from `model.features` at indices `{1, 3, 6}`.
- Each captured tensor is rendered as a grid image (`src/viz.py`), showing up to 12 channels (4 columns) with per-channel min/max normalization.

### Optimisation / training (src/train.py + src/config.py)

- Loss: `nn.CrossEntropyLoss`.
- Optimizer: `torch.optim.AdamW` with `lr=1e-3` (from `Config.lr`) and `weight_decay=1e-4`.
- Mixed precision: optional AMP via `torch.autocast(...)` + `torch.cuda.amp.GradScaler` when `Config.use_amp=True` and CUDA is available.
- Training: end-to-end fine-tuning (`optimizer` over `model.parameters()`), reporting loss + accuracy each epoch.
- Model selection: best checkpoint is saved based on validation accuracy.
- Default hyperparameters: `img_size=128`, `padding=12`, `batch_size=64`, `epochs=20`, `lr=1e-3`, `val_split=0.2`, `seed=42`, `use_amp=True`.

### Data preprocessing (src/data.py + src/infer.py)

- Split: stratified train/val split (`val_split=0.2`) using `train_test_split(..., stratify=targets)`.
- Train transforms: Grayscale → Pad(12) → RandomAffine(degrees=18, translate=(0.12, 0.12), scale=0.85–1.15, shear=7, fill=0) → RandomPerspective(distortion_scale=0.2, p=0.2) → Resize(128×128) → Normalize(mean=0.5, std=0.5).
- Val/inference transforms: Grayscale → Pad(12) → Resize(128×128) → Normalize(mean=0.5, std=0.5).
- Sketchpad input: if the background is mostly white, the image is inverted so the letter is bright on a dark background (see `preprocess_any_image` in `src/infer.py`).

## Data layout

Place the dataset under `dataset/` using ImageFolder layout:

```
dataset/
  <class_name>/
    img1.png
    img2.png
  <class_name_2>/
    ...
```

The training script will split a validation set automatically.

## Train

```
python -m src.train
```

Checkpoints are saved under `models/` (default: `models/mobilenet_arabic_letters.pt`).
Adjust hyperparameters in `src/config.py` as needed.

## Run the demo app

```
python -m src.app
```

The app loads the checkpoint from `models/mobilenet_arabic_letters.pt`, accepts
either a sketchpad drawing or an uploaded image, and shows feature map grids
from early layers.

## Dependencies

```
pip install -r requirements.txt
```
