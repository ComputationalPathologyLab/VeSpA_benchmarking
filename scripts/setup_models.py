from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
SAM_REPO_DIR = REPO_ROOT / "external" / "segment-anything"
YOLO_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-seg.pt"

SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def clone_sam_repo() -> None:
    SAM_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    if SAM_REPO_DIR.exists():
        print(f"SAM repository already exists at {SAM_REPO_DIR}")
        return
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/facebookresearch/segment-anything",
            str(SAM_REPO_DIR),
        ],
        check=True,
    )


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"File already exists at {destination}")
        return
    print(f"Downloading {url} -> {destination}")
    urlretrieve(url, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SAM repository and checkpoints.")
    parser.add_argument("--sam-model-type", choices=sorted(SAM_CHECKPOINT_URLS), default="vit_b")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--with-yolo", action="store_true", help="Also download yolov8n-seg.pt into models/.")
    args = parser.parse_args()

    if not args.skip_clone:
        clone_sam_repo()

    checkpoint_url = SAM_CHECKPOINT_URLS[args.sam_model_type]
    filename = checkpoint_url.rsplit("/", 1)[-1]
    download_file(checkpoint_url, MODELS_DIR / filename)
    if args.with_yolo:
        download_file(YOLO_WEIGHTS_URL, MODELS_DIR / "yolov8n-seg.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
