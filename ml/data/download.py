"""Download and verify the LEVIR-CD dataset.

Source: the Hugging Face mirror ``satellite-image-deep-learning/LEVIR-CD`` at a pinned
revision, using the same URLs and MD5 checksums that TorchGeo's ``LEVIRCD`` dataset class
uses. The archives contain the authors' official train / val / test split.

Dataset: H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset
for Remote Sensing Image Change Detection", Remote Sensing 12(10):1662, 2020.
Licence: academic use only, commercial use prohibited (https://justchenhao.github.io/LEVIR/).

Usage:  python -m ml.data.download --root data/raw/LEVIR-CD
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

_REV = "6a6bb0a5b389403d81c05e33bf08bc0b9e5f13a6"
_BASE = f"https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/{_REV}"

ARCHIVES: dict[str, dict[str, str]] = {
    "train": {"url": f"{_BASE}/train.zip", "md5": "a638e71f480628652dea78d8544307e4"},
    "val": {"url": f"{_BASE}/val.zip", "md5": "f7b857978524f9aa8c3bf7f94e3047a4"},
    "test": {"url": f"{_BASE}/test.zip", "md5": "07d5dd89e46f5c1359e2eca746989ed9"},
}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()  # noqa: S324 - integrity check against a published checksum
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "terravision-dataset-fetch"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        next_report = 0
        while block := response.read(1 << 20):
            out.write(block)
            done += len(block)
            if total and done >= next_report:
                print(f"  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
                next_report += total // 10 or 1
    tmp.replace(dest)


def _safe_extract(archive: Path, target: Path) -> None:
    """Extract ``archive`` into ``target`` refusing any member that escapes it."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            resolved = (target / member.filename).resolve()
            if not resolved.is_relative_to(target):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(target)


def fetch(root: Path, keep_archives: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for split, meta in ARCHIVES.items():
        split_dir = root / split
        if (split_dir / "A").is_dir() and any((split_dir / "A").iterdir()):
            print(f"[{split}] already extracted, skipping")
            continue
        archive = root / f"{split}.zip"
        if not archive.exists():
            print(f"[{split}] downloading {meta['url']}")
            _download(meta["url"], archive)
        print(f"[{split}] verifying md5")
        actual = md5sum(archive)
        if actual != meta["md5"]:
            raise RuntimeError(f"{archive} md5 mismatch: expected {meta['md5']}, got {actual}")
        print(f"[{split}] extracting")
        split_dir.mkdir(exist_ok=True)
        _safe_extract(archive, split_dir)
        # Some archives nest a top-level folder named after the split; flatten it.
        nested = split_dir / split
        if nested.is_dir() and not (split_dir / "A").exists():
            for child in nested.iterdir():
                child.rename(split_dir / child.name)
            nested.rmdir()
        if not keep_archives:
            archive.unlink()
    print("done")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/raw/LEVIR-CD"))
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args(argv)
    fetch(args.root, args.keep_archives)
    return 0


if __name__ == "__main__":
    sys.exit(main())
