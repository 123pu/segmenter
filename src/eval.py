# -*- coding: utf-8 -*-

# ============================================================================
"""Segmenter inference."""
import timeit
import argparse
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from src.data.dataloader import create_scapes_dataset
from src import config
from src.nets.factory import create_segmenter
import moxing as mox

class BuildEvalNetwork(nn.Cell):
    def __init__(self, network, input_format="NCHW"):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)
        self.transpose = ops.Transpose()
        self.format = input_format

    def construct(self, input_data):
        if self.format == "NHWC":
            input_data = self.transpose(input_data, (0, 3, 1, 2))
        output = self.network(input_data)
        output = self.softmax(output)
        return output

def get_confusion_matrix(label, pred, shape, num_class, ignore=-1):
    """Calcute the confusion matrix by given label and pred."""
    output = pred.asnumpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint16)
    seg_gt = np.asarray(label.asnumpy()[:, :shape[-2], :shape[-1]], dtype=np.int32)
    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]
    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))
    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    confusion_matrix = label_count.reshape((num_class, num_class))

    return confusion_matrix

def inference_cityscapes(net, dataset, dataloader, num_classes, scales, flip=False):
    """Inference with Cityscapes."""
    confusion_matrix = np.zeros((num_classes, num_classes))
    count = 0
    for data in dataloader:
        image, label = data

        shape = label.shape
        if len(scales)>1:
            from src.data.dataset import multi_scale_inference
            pred = multi_scale_inference(dataset, net, image, scales, flip)
        else:
            pred = net(image)
        confusion_matrix += get_confusion_matrix(label, pred, shape, num_classes, 255)
        count += 1
        print("image:{:4d} saved".format(count))
    return confusion_matrix, count

def main():
    args = config.load_config()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')

    # parameter config
    model_cfg = args["model"]["vit_large_patch16_384"]
    eval_cfg = args["evaluation"]['segmantic']
    dataset_cfg = args["dataset"]['cityscapes']
    decoder_cfg = args["decoder"]["mask_transformer"]

    # model config
    crop_size = dataset_cfg["crop_size"]  # 768
    world_batch_size = dataset_cfg["batch_size"]
    model_cfg["image_size"] = (crop_size, crop_size)  # (768, 768)
    model_cfg["backbone"] = "vit_large_patch16_384"  #
    model_cfg["dropout"] = 0.1
    model_cfg["drop_path_rate"] = 0.1
    decoder_cfg["name"] = 'mask_transformer'
    model_cfg["decoder"] = decoder_cfg

    # eval config
    ckpt_path = eval_cfg['ckpt_path']
    data_root = eval_cfg['data_root']
    scales = eval_cfg['scales']
    flip = eval_cfg['flip']
    input_format = eval_cfg['input_format']
    n_cls = dataset_cfg['num_classes']

    # experiment config
    batch_size = world_batch_size // 1
    eval_freq = eval_cfg['eval_freq']

    variant = dict(
        world_batch_size=batch_size,
        version="normal",
        dataset_kwargs=dict(
            crop_size=crop_size,
            batch_size=eval_freq,
            data_root=data_root,
            ignore_label=255,
            rank=0,
            group_size=1,
        ),
        net_kwargs=model_cfg,
    )

    # eval dataset
    dataset_kwargs = variant['dataset_kwargs']
    # create model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = n_cls
    network = create_segmenter(net_kwargs)
    eval_net = BuildEvalNetwork(network, input_format)

    """Inference process."""
    dataset, datasets = create_scapes_dataset(dataset_kwargs, is_train=False)

    param_dict = load_checkpoint(ckpt_file_name= ckpt_path)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)

    # Calculate results
    start = timeit.default_timer()
    confusion_matrix, count = inference_cityscapes(eval_net, dataset, datasets, n_cls, scales, flip)

    end = timeit.default_timer()
    total_time = end - start
    avg_time = total_time / count
    print("Number of samples: {:4d}, total time: {:4.2f}s, average time: {:4.2f}s".format(
       count, total_time, avg_time))

    mean_iou = miou(confusion_matrix, ckpt_path)
    # Show results
    print(ckpt_path,"miou:", mean_iou)
    return mean_iou

def miou(hist, ckpt_path):
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    print(ckpt_path,"iou array: \n", iou)
    miou = np.nanmean(iou)
    return miou


if __name__ == '__main__':
    mox.file.copy_parallel(src_url='s3://seg-contains/city_data', dst_url='./city_data')
    mox.file.copy_parallel(src_url='s3://seg-contains/models', dst_url='./models')
    main()

