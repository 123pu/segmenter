# -*- coding: utf-8 -*-
# ============================================================================
"""Dataset generators."""

import mindspore.dataset.engine as de

from src.data.dataset import Cityscapes

'''crop_size=args.crop_size,
            batch_size=args.batch_size,
            data_root=args.data_root,
            ignore_label=args.ignore_label,
            rank=args.rank,
            group_size=args.group_size,'''

def create_scapes_dataset(kwards, is_train=True, shuffle=False):
    """
    create train/test dataset
    """
    data_root = kwards['data_root']
    batch_size = kwards['batch_size']
    crop_size = kwards['crop_size']
    ignore_label = kwards['ignore_label']
    rank = kwards['rank']
    group_size = kwards['group_size']
    num_classes = 19
    if is_train:
        multi_scale = True
        flip = True
        crop_size = (crop_size, crop_size)
    else:
        multi_scale = False
        flip = False
        crop_size = (crop_size, crop_size)
    if data_root is None:
        return crop_size, num_classes
    dataset = Cityscapes(data_root,
                         num_samples=None,
                         num_classes=19,
                         multi_scale=multi_scale,
                         flip=flip,
                         ignore_label=ignore_label,
                         base_size=2048,
                         crop_size=crop_size,
                         downsample_rate=1,
                         scale_factor=16,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         is_train=is_train)
    datasets = de.GeneratorDataset(dataset, column_names=["image", "label"], num_parallel_workers = 1,
                                   num_shards=group_size,
                                   shard_id=rank,
                                  shuffle=shuffle)
    datasets = datasets.batch(batch_size, drop_remainder=True)

    return dataset, datasets
