# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


    
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, "train" if is_train else "val")
    dataset = datasets.ImageFolder(root, transform=transform)

    n_cls = getattr(args, "subset_classes", 0)
    frac = getattr(args, "subset_frac", 1.0)
    if n_cls or frac < 1.0:
        dataset = _subset(dataset, n_cls, frac, is_train)
    return dataset


def _subset(dataset, n_cls, frac, is_train):
    """First n_cls classes, first frac of the images in each -- for the rho scan.

    Deterministic (no shuffling), so a baseline arm and a regularized arm see exactly
    the same images.  Class labels keep their original ids; the head stays 1000-way.
    """
    from torch.utils.data import Subset
    from collections import defaultdict

    by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        if n_cls and label >= n_cls:
            continue
        by_class[label].append(idx)

    keep = []
    for label, idxs in by_class.items():
        k = len(idxs) if frac >= 1.0 else max(1, int(len(idxs) * frac))
        keep.extend(idxs[:k])
    keep.sort()
    print("[subset] {}: {} images over {} classes (of {} / {})".format(
        "train" if is_train else "val", len(keep), len(by_class),
        len(dataset.samples), len(dataset.classes)))
    return Subset(dataset, keep)

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        # if args.three_aug:

        #     secondary_tfl = transforms.RandomChoice([gray_scale(p=1.0),
        #                                       Solarization(p=1.0),
        #                                       GaussianBlur(p=1.0)])
        #     transform = transforms.Compose([transform,secondary_tfl])
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 232
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
import torch
from torchvision import transforms

import numpy as np
from torchvision import datasets, transforms
import random



from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
