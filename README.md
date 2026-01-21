# Arabic-MNIST

Arabic letter classification with a MobileNetV3-Small model, trained on an
ImageFolder-style dataset and served through a Gradio demo that also visualizes
intermediate feature maps.

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

Checkpoints are saved under `runs/` (default: `runs/mobilenet_arabic_letters.pt`).
Adjust hyperparameters in `src/config.py` as needed.

## Run the demo app

```
python -m src.app
```

The app loads the checkpoint from `runs/mobilenet_arabic_letters.pt`, accepts
either a sketchpad drawing or an uploaded image, and shows feature map grids
from early layers.

## Dependencies

```
pip install -r requirements.txt
```
