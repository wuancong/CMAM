import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from fastreid.modeling.losses import CrossEntropyLoss
from torch.nn import Parameter
from fastreid.utils.one_hot import one_hot

class CM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum, save_tensor=True):
        ctx.features = features
        ctx.momentum = momentum
        if save_tensor:
            ctx.save_for_backward(inputs, targets)
        else:
            ctx.save_for_backward(torch.tensor([]), torch.tensor([]))
        # print('inputs, targets', inputs.shape, ctx.features.shape)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        # print(grad_outputs.dtype, ctx.features.dtype)
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features.type_as(grad_outputs))
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None


def cm(inputs, indexes, features, momentum=0.5, save_tensor=True):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), save_tensor)


class CM_Hard(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features.type_as(grad_outputs))

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))




class ClusterMemory(nn.Module, ABC):
    def __init__(self,cfg, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.cfg = cfg
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def norm_forward(self, features, targets, weight):
        sim_mat = F.linear(F.normalize(features), F.normalize(weight) )
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        targets = one_hot(targets, self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n
        return pred_class_logits, targets

    def forward(self, inputs, targets, save_tensor=True):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum, save_tensor=save_tensor)

        outputs /= self.temp
        if self.cfg.UL.CLUSTER.LSCE:
            loss = CrossEntropyLoss(self.cfg)(outputs, None, targets)['loss_cls']
        else:
            loss = F.cross_entropy(outputs, targets)
        return loss, outputs

    def update(self, inputs, targets):
        for x, y in zip(inputs, targets):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()


class CircleMemory(nn.Module):
    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        self.cfg = cfg

    def forward(self, features, targets):
        # print("??", self.weight.shape, features.shape, targets.shape)
        ori_targets = targets
        sim_mat = F.linear(F.normalize(features), F.normalize(self.weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        targets = one_hot(targets, self._num_classes)
        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        if self.cfg.UL.CLUSTER.LSCE:
            loss = CrossEntropyLoss(self.cfg)(pred_class_logits, None, ori_targets)['loss_cls']
        else:
            loss = F.cross_entropy(pred_class_logits, ori_targets)

        return loss, pred_class_logits


    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )

