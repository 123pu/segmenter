# -*- coding: utf-8 -*-
import pathlib

import argparse
import mindspore.nn as nn
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.common import set_seed
from mindspore import context
from src.loss import loss
from src import config
from src.nets.factory import create_segmenter
from src.utils.optimizer import create_optimizer
from src.data.dataloader import create_scapes_dataset
from src.evaluate.eval_acc import StepLossAccInfo
import moxing as mox


set_seed(1)

# 动静态图设置
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')


def parse_args():
    parser = argparse.ArgumentParser('mindspore segmenter training')
    parser.add_argument("--output_dir", type=str, default='./segm_output', help='where training log and ckpts saved')
    parser.add_argument("--data_root", type=str, default='./city_data/Cityscapes', help='data root path')
    parser.add_argument("--model", type=str, default='segmenter', help='select model')
    parser.add_argument('--num_classes', type=int, default=19, help='number of classes')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    # dataset
    parser.add_argument("--dataset", type=str, default="cityscapes", help='dataset path')
    parser.add_argument("--im_size", default=None, type=int, help="dataset resize size")
    parser.add_argument("--crop_size", default=None, type=int)
    parser.add_argument("--backbone", default="vit_large_patch16_384", type=str)
    parser.add_argument("--decoder", default="mask_transformer", type=str)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--drop_path", default=0.1, type=float)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--base_lr", default=None, type=float)
    parser.add_argument("--lr_decay_step", default=None, type=float)
    parser.add_argument("--lr_decay_rate", default=None, type=float)
    parser.add_argument('--lr_type', type=str, default='poly', help='type of learning rate')
    parser.add_argument("--eval_freq", default=None, type=int)
    parser.add_argument('--loss_scale', type=float, default=3072.0, help='loss scale')
    parser.add_argument("--device_target", type=str, default='Ascend', help='Target device type')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    parser.add_argument('--save_steps', type=int, default=2975, help='steps interval for saving')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='max checkpoint for saving')
    args, _ = parser.parse_known_args()
    return args

def train_net():
    args = parse_args()
    cfg = config.load_config()
    model_cfg = cfg["model"][args.backbone]
    dataset_cfg = cfg["dataset"][args.dataset]
    if "mask_transformer" in args.decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][args.decoder]
    # model config
    if not args.im_size:
        args.im_size = dataset_cfg["im_size"]   # 1024
    if not args.crop_size:
        args.crop_size = dataset_cfg.get("crop_size", args.im_size)   # 768

    model_cfg["image_size"] = (args.crop_size, args.crop_size)  #(768, 768)
    model_cfg["backbone"] = args.backbone  #
    model_cfg["dropout"] = args.dropout
    model_cfg["drop_path_rate"] = args.drop_path
    decoder_cfg["name"] = args.decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    args.base_lr = dataset_cfg["base_lr"]
    args.lr_decay_step = dataset_cfg['lr_decay_step']
    args.lr_decay_rate = dataset_cfg['lr_decay_step']
    if args.batch_size:
        world_batch_size = args.batch_size
    if args.epochs is None:
        args.epochs = num_epochs
    if args.eval_freq is None:
        args.eval_freq = dataset_cfg.get("eval_freq", 1)

    # experiment config
    batch_size = world_batch_size // 1
    args.batch_size = batch_size
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=args.resume,
        dataset_kwargs=dict(
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            data_root=args.data_root,
            ignore_label=args.ignore_label,
            rank=args.rank,
            group_size=args.group_size,
        ),
        algorithm_kwargs=dict(
            batch_size=args.batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=args.eval_freq,
        ),
        net_kwargs=model_cfg,
    )

    # dataset
    # train
    dataset_kwargs = variant["dataset_kwargs"]

    _, dataset = create_scapes_dataset(dataset_kwargs, shuffle=True)
    _, eval_dataset = create_scapes_dataset(dataset_kwargs, is_train=False)
    n_cls = args.num_classes
    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = n_cls
    network = create_segmenter(net_kwargs)

    # loss
    loss_ = loss.SoftmaxCrossEntropyLoss(n_cls, args.ignore_label)
    loss_.add_flags_recursive(fp32=True)

    # optimizer
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]

    iters_per_epoch = dataset.get_dataset_size()
    total_train_steps = iters_per_epoch * num_epochs
    opt = create_optimizer(args, network, total_train_steps)

    # 损失缩放系数不变管理器
    manager_loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    amp_level = "O2" if args.device_target == "GPU" else "O3"
    model = Model(
                  network,
                  loss_fn=loss_,
                  optimizer=opt,
                  amp_level=amp_level,
                  loss_scale_manager=manager_loss_scale,
                  metrics={"accuracy"},
                  )

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    steps_loss = {"step": [], "loss_value": []}
    steps_eval = {"step": [], "acc": []}
    step_loss_acc_info = StepLossAccInfo(model, eval_dataset, steps_loss, steps_eval)
    cbs = [time_cb, loss_cb, step_loss_acc_info]

    # save checkpoint
    if args.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.save_steps,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=args.model, directory=args.output_dir, config=config_ck)
        cbs.append(ckpoint_cb)

    # train for one epoch
    print('start train!!!')
    model.train(args.epochs, dataset, callbacks=cbs, dataset_sink_mode=(args.device_target != "CPU"))
    # 关联训练完成后的模型路径
    # segm_output 模型导出路径
    mox.file.copy_parallel(src_url='./segm_output', dst_url='s3://cityscapes-seg/segm_output')

if __name__ == "__main__":
    # 关联数据路径
    # cityscapes-seg 创建桶名称
    mox.file.copy_parallel(src_url='s3://cityscapes-seg/city_data', dst_url='./city_data')
    train_net()
