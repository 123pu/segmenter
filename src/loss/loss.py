# -*- coding: utf-8 -*-
# ============================================================================

from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore as ms
from mindspore.ops import operations as P


class SoftmaxCrossEntropyLoss(nn.Cell):
    def __init__(self, num_cls=19, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        # self.ce = nn.BCEWithLogitsLoss()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        labels_int = self.cast(labels, ms.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)   # (589824,)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        logits_n = logits_.astype('float32')
        one_hot_labels_n = one_hot_labels.astype('float32')
        loss = self.ce(logits_n, one_hot_labels_n)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights)+ 1e-3)
        return loss
