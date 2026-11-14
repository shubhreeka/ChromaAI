import argparse
import random
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

URL = "http://images.cocodataset.org/zips/val2017.zip"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"


def download(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    zip_path = RAW / "val2017.zip"
    download(URL, zip_path)
    image_dir = RAW / "val2017"
    if not image_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW)

    images = sorted(image_dir.glob("*.jpg"))
    random.Random(args.seed).shuffle(images)
    if args.limit:
        images = images[: args.limit]

    n_test = max(1, int(len(images) * 0.15))
    n_val = max(1, int(len(images) * 0.15))
    parts = {
        "test": images[:n_test],
        "val": images[n_test : n_test + n_val],
        "train": images[n_test + n_val :],
    }
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, items in parts.items():
        (SPLITS / f"{name}.txt").write_text("\n".join(str(p) for p in items), encoding="utf-8")
        print(name, len(items))


if __name__ == "__main__":
    main()
