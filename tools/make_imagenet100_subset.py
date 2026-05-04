#!/usr/bin/env python3
"""Create an ImageNet-100 ImageFolder subset by symlinking selected synsets.

Usage:
  python tools/make_imagenet100_subset.py \
    --imagenet-root /data/imagenet \
    --synsets configs/imagenet100_synsets.txt \
    --out /data/imagenet100

Expected source:
  /data/imagenet/train/<wnid>/*.JPEG
  /data/imagenet/val/<wnid>/*.JPEG
Target:
  /data/imagenet100/train/<wnid> -> symlink
  /data/imagenet100/val/<wnid> -> symlink
"""
import argparse
from pathlib import Path
import os
import shutil


def link_or_copy(src: Path, dst: Path, copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copytree(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imagenet-root", required=True)
    p.add_argument("--synsets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--copy", action="store_true", help="copy directories instead of symlinking")
    args = p.parse_args()
    src_root = Path(args.imagenet_root)
    out = Path(args.out)
    synsets = [x.strip() for x in open(args.synsets, "r", encoding="utf-8") if x.strip() and not x.startswith("#")]
    if len(synsets) != 100:
        print(f"Warning: synset file contains {len(synsets)} classes, not 100")
    for split in ["train", "val"]:
        for wnid in synsets:
            src = src_root / split / wnid
            if not src.exists():
                print(f"Missing: {src}")
                continue
            link_or_copy(src, out / split / wnid, copy=args.copy)
    print(f"Created ImageNet subset at {out}")


if __name__ == "__main__":
    main()
