import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]  # PIL image, int label
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def build_transforms(img_size: int, padding: int):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Pad(padding, fill=0),  # black border
        transforms.RandomAffine(
            degrees=18,
            translate=(0.12, 0.12),
            scale=(0.85, 1.15),
            shear=7,
            fill=0,
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Pad(padding, fill=0),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return train_tf, val_tf

def build_loaders(data_dir: Path, img_size: int, padding: int,
                  batch_size: int, num_workers: int, val_split: float, seed: int):
    raw = datasets.ImageFolder(root=str(data_dir), transform=None)
    class_names = raw.classes

    indices = np.arange(len(raw))
    targets = [raw.samples[i][1] for i in indices]

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=targets,
        random_state=seed
    )

    train_tf, val_tf = build_transforms(img_size, padding)

    train_ds = TransformedSubset(raw, train_idx, transform=train_tf)
    val_ds = TransformedSubset(raw, val_idx, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_names, val_tf
