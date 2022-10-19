# -*- coding: utf-8 -*-
# ============================================================================
"""Base dataset generator definition."""
import random
import cv2
import numpy as np

import mindspore.ops as P
from mindspore import Tensor
from mindspore.common import dtype


class BaseDataset:
    """Base dataset generator class."""
    def __init__(self,
                 ignore_label=-1,
                 num_classes=19,
                 base_size=2048,
                 crop_size=None,
                 downsample_rate=1,
                 scale_factor=16,
                 mean=None,
                 std=None):

        self.num_classes = num_classes
        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1. / downsample_rate

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        """Transform data format of images."""
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        """Transform data format of labels."""
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, shape, padvalue):
        """Pad an image."""
        pad_image = image.copy()
        pad_h = max(shape[0] - h, 0)
        pad_w = max(shape[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label):
        """Crop a feature at a random location."""
        h = image.shape[0]
        w = image.shape[1]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        return image, label

    def multi_scale_aug(self, image, label=None, rand_scale=1.0, rand_crop=True):
        """Augment feature into different scales."""
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h = image.shape[0]
        w = image.shape[1]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label

    def gen_sample(self, image, label, multi_scale=False, is_flip=False):
        """Data preprocessing."""
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)
        else:
            image, label = self.rand_crop(image, label)
        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(label, None,
                               fx=self.downsample_rate,
                               fy=self.downsample_rate,
                               interpolation=cv2.INTER_NEAREST)
        return image, label

