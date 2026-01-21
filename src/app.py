import numpy as np
import gradio as gr
import torch

from .infer import load_checkpoint, predict, preprocess_any_image
from .viz import feature_maps_to_grid

CKPT_PATH = "runs/mobilenet_arabic_letters.pt"

model, class_names, val_tf, device = load_checkpoint(CKPT_PATH)

# Hook: capture some internal features (simple approach: grab early feature maps)
# MobileNetV3 features is a Sequential; we’ll take outputs at a few points.
@torch.no_grad()
def forward_with_features(x):
    feats = []
    out = x
    # pick a few indices to visualize (tweak later)
    watch = {1, 3, 6}
    for i, layer in enumerate(model.features):
        out = layer(out)
        if i in watch:
            feats.append(out)
    logits = model.classifier(model.avgpool(out).flatten(1))
    return logits, feats

def unwrap_sketchpad(img):
    if isinstance(img, dict):
        # Prefer composite (actual drawing)
        for key in ["composite", "image", "background"]:
            if key in img and img[key] is not None:
                return np.array(img[key])
        return None
    return img

@torch.no_grad()
def predict_and_viz(img):
    img = unwrap_sketchpad(img)
    if img is None:
        return {}, None, None, None

    x = preprocess_any_image(img, val_tf, device)
    if x is None:
        return {}, None, None, None

    logits, feats = forward_with_features(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    probs_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    # create feature grids
    grids = []
    for f in feats:
        grids.append(feature_maps_to_grid(f, max_maps=12, cols=4))

    # ensure we always return 3 images
    while len(grids) < 3:
        grids.append(None)

    return probs_dict, grids[0], grids[1], grids[2]

def main():
    with gr.Blocks(title="Arabic Letter Classifier (MobileNet)") as demo:
        gr.Markdown("# Arabic Letter Classifier (MobileNetV3)")

        with gr.Row():
            sketch = gr.Sketchpad(label="Draw an Arabic letter")
            upload = gr.Image(type="numpy", image_mode="L", label="Or upload an image")

        label = gr.Label(num_top_classes=5, label="Prediction (Top-5)")

        with gr.Row():
            f1 = gr.Image(type="numpy", image_mode="L", label="Feature maps (stage 1)")
            f2 = gr.Image(type="numpy", image_mode="L", label="Feature maps (stage 2)")
            f3 = gr.Image(type="numpy", image_mode="L", label="Feature maps (stage 3)")

        btn1 = gr.Button("Predict from Drawing")
        btn2 = gr.Button("Predict from Upload")

        btn1.click(fn=predict_and_viz, inputs=sketch, outputs=[label, f1, f2, f3])
        btn2.click(fn=predict_and_viz, inputs=upload, outputs=[label, f1, f2, f3])

    demo.launch()

if __name__ == "__main__":
    main()
