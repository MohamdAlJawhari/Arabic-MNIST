import argparse
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import Config
from .data import build_loaders
from .model import build_mobilenet
from .utils import seed_everything, get_device, save_checkpoint

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=loss.item())

    return total_loss / total, correct / total

def main():
    cfg = Config()
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    device = get_device()
    print("Device:", device)

    train_loader, val_loader, class_names, _ = build_loaders(
        cfg.data_dir, cfg.img_size, cfg.padding,
        cfg.batch_size, cfg.num_workers, cfg.val_split, cfg.seed
    )

    model = build_mobilenet(num_classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type=="cuda"))

    best_acc = 0.0
    ckpt_path = cfg.runs_dir / cfg.ckpt_name

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, cfg.use_amp)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc*100:.2f}%")

        if va_acc > best_acc:
            best_acc = va_acc
            save_checkpoint(ckpt_path, model, class_names, cfg)
            print(f"  ✅ saved best -> {ckpt_path} (val acc {best_acc*100:.2f}%)")

    print("Done. Best val acc:", round(best_acc*100, 2), "%")

if __name__ == "__main__":
    main()
