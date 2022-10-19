

import mindspore.nn as nn
from src.utils import learning_rates


def create_optimizer(args, train_net, total_train_steps):
    if args.lr_type == "cos":
        lr_iter = learning_rates.cosine_lr(args.base_lr, total_train_steps, total_train_steps)
    elif args.lr_type == "poly":
        lr_iter = learning_rates.poly_lr(args.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    elif args.lr_type == 'exp':
        lr_iter = learning_rates.exponential_lr(args.base_lr, args.lr_decay_step, args.lr_decay_rate,
                                                total_train_steps, staircase=True)
    else:
        raise ValueError('unknown learning rate type')
    if args.optimizer == 'sgd':
        opt = nn.SGD(params=train_net.trainable_params(),
                     learning_rate=lr_iter,
                     momentum=0.9,
                     weight_decay=0.0001,
                     loss_scale=args.loss_scale)
    elif args.optimizer == 'adam':
        opt = nn.Adam(params=train_net.trainable_params(), learning_rate=lr_iter, weight_decay=0.0)
    else:
        opt = nn.Momentum(params=train_net.trainable_params(), learning_rate=lr_iter, momentum=0.9)
    return opt











