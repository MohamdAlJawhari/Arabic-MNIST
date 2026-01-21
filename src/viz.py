import numpy as np
import torch

def feature_maps_to_grid(fmap: torch.Tensor, max_maps: int = 12, cols: int = 4):
    fmap = fmap.detach().cpu()
    fmap = fmap[0]  # (C,H,W)
    C, H, W = fmap.shape
    n = min(C, max_maps)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    grid = np.zeros((rows * H, cols * W), dtype=np.uint8)

    for i in range(n):
        fm = fmap[i].numpy()
        mn, mx = fm.min(), fm.max()
        if mx > mn:
            fm = (fm - mn) / (mx - mn)
        else:
            fm = np.zeros_like(fm)
        fm = (fm * 255).astype(np.uint8)

        r, c = divmod(i, cols)
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = fm
    return grid
