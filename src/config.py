from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "dataset"
    runs_dir: Path = project_root / "runs"

    img_size: int = 128          # MobileNet likes bigger than 64; 128 is a good start
    padding: int = 12            # empty frame
    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 20
    lr: float = 1e-3
    seed: int = 42

    val_split: float = 0.2
    use_amp: bool = True         # mixed precision if CUDA

    # checkpoint
    ckpt_name: str = "mobilenet_arabic_letters.pt"
