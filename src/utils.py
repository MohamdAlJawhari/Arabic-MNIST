import random
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(path, model, class_names, cfg):
    ckpt = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "img_size": cfg.img_size,
        "padding": cfg.padding,
    }
    torch.save(ckpt, path)
