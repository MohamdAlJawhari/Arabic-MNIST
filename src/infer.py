import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from .model import build_mobilenet
from .utils import get_device

def load_checkpoint(ckpt_path: str):
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    img_size = ckpt["img_size"]
    padding = ckpt["padding"]

    model = build_mobilenet(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Pad(padding, fill=0),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    return model, class_names, val_tf, device

def preprocess_any_image(img, val_tf, device):
    """
    img can be:
    - numpy array from Gradio
    - PIL image
    """
    if img is None:
        return None

    if isinstance(img, np.ndarray):
        # Handle RGBA
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        pil = Image.fromarray(img.astype("uint8"))
    else:
        pil = img

    pil = pil.convert("L")
    arr = np.array(pil)

    # If background is white (sketch), invert so it becomes black background with white strokes
    if arr.mean() > 127:
        pil = ImageOps.invert(pil)

    x = val_tf(pil).unsqueeze(0).to(device)
    return x

@torch.no_grad()
def predict(img, model, class_names, val_tf, device):
    x = preprocess_any_image(img, val_tf, device)
    if x is None:
        return {}, None

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    probs_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    pred_idx = int(probs.argmax())
    return probs_dict, class_names[pred_idx]
    