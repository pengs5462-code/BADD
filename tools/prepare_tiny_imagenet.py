#!/usr/bin/env python3
import argparse
from baddlab.datasets import prepare_tiny_imagenet_val


def main():
    p = argparse.ArgumentParser("Prepare official Tiny-ImageNet val folder for ImageFolder")
    p.add_argument("--root", required=True, help="path to tiny-imagenet-200")
    args = p.parse_args()
    prepare_tiny_imagenet_val(args.root)


if __name__ == "__main__":
    main()
