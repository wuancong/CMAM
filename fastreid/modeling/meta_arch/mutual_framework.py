import torch
from torch import nn

from .build import META_ARCH_REGISTRY
import torch.nn.functional as F
import numpy as np


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist2 = dist - 2 * torch.matmul(x, y.t())
    dist = dist2.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


@META_ARCH_REGISTRY.register()
class NEWBASE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self.avg_clsloss = 0
        if self._cfg.MODEL.BASE == 'BASE':
            from .model_backbone import BASE
        elif self._cfg.MODEL.BASE == 'BASE_UDA':
            from .model_backbone_uda import BASE
        self.model1 = BASE(cfg).cuda()
        self.model2 = BASE(cfg).cuda()
        self.pixel_mean = self.model1.pixel_mean
        self.pixel_std = self.model1.pixel_std

    def forward(self, batched_inputs, input_key='images', use_model_idx=[1, 2]):

        batched_inputs = {key: value.to('cuda') if torch.is_tensor(value) else value
                          for key, value in batched_inputs.items()}

        if not self.training:
            pred_feat = self.inference(batched_inputs, input_key=input_key, use_model_idx=use_model_idx)
            return pred_feat, batched_inputs["targets"], batched_inputs["camid"], batched_inputs['img_path']

        model1, model2 = self.model1, self.model2

        self.camids = batched_inputs['camid']

        targets = batched_inputs["targets"].long()
        cams = batched_inputs['camid']

        # batched_inputs['images']: original RGB images
        # batched_inputs['images1']: transformed grayscale images
        images_rgb = self.preprocess_image(batched_inputs['images'], 'RGB')
        images_channel = self.preprocess_image(batched_inputs['images'], 'CHANNEL')
        if batched_inputs.get('images1') is not None:
            images_transformed = self.preprocess_image(batched_inputs['images1'], 'RGB')
            combine_weight = 0.2
            images_transformed = combine_weight * images_transformed + (1 - combine_weight) * images_channel
        else:
            images_transformed = images_channel

        if self._cfg.MUTUAL.TYPE == 'ASYMMETRIC':
            out_m1_rgb = model1.train_forward(images_rgb, targets, cams, head_idx=1)
            out_m1_transformed = model1.train_forward(images_transformed, targets, cams, head_idx=2)
            out_m2_transformed = model2.train_forward(images_transformed, targets, cams, head_idx=2)
        elif self._cfg.MUTUAL.TYPE == 'SYMMETRIC':
            try:
                out_m1_rgb = model1.train_forward(images_rgb, targets, cams, head_idx=1)
                out_m1_transformed = None
                out_m2_transformed = model2.train_forward(images_transformed, targets, cams, head_idx=2)
            except:
                bug_state = {
                    'images_rgb': images_rgb,
                    'targets': targets,
                    'cams': cams,
                    'images_transformed': images_transformed,
                }
                torch.save(bug_state, './logs_release/regdb_bug_state.pth')
        else:
            raise ValueError('Unsupported MUTUAL.TYPE')

        return {
            'out_m1_rgb': out_m1_rgb,
            'out_m1_transformed': out_m1_transformed,
            'out_m2_transformed': out_m2_transformed,
        }

    def inference(self, batched_inputs, input_key='images', use_model_idx=[1, 2]):
        assert not self.training
        images_rgb = self.preprocess_image(batched_inputs['images'], 'RGB')
        if 1 in use_model_idx:
            pred_feat1 = self.model1.test_forward(images_rgb)
        else:
            pred_feat1 = None

        images_channel = self.preprocess_image(batched_inputs['images'], 'CHANNEL')
        if 2 in use_model_idx:
            pred_feat2 = self.model2.test_forward(images_channel)
        else:
            pred_feat2 = None
        if pred_feat1 is None:
            pred_feat1 = torch.zeros_like(pred_feat2)
        if pred_feat2 is None:
            pred_feat2 = torch.zeros_like(pred_feat1)
        pred_feat = torch.cat((pred_feat1, pred_feat2), dim=1)

        return pred_feat

    def preprocess_image(self, images_in, input_type):
        # input_type: "RGB" (original image) or "CHANNEL" (repeat one random channel as grayscale image)
        images = (images_in - self.pixel_mean) / self.pixel_std
        images = self.adjust_images(images, input_type)
        return images

    def adjust_images(self, images, input):
        if input == 'CHANNEL':
            N = len(images)
            for i in range(N):
                a, b, c = np.random.choice(3, 3, replace=False)
                images[i, b] = images[i, a]
                images[i, c] = images[i, a]
        elif input == 'RGB':
            images = images
        else:
            assert False
        return images

    def compute_kl_loss_stage1(self, p, q):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def compute_kl_loss_stage2(self, p, q):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        p_loss = p_loss.sum()
        return p_loss

    def compute_sim_loss_stage1(self, a_feat, b_feat):
        a_feat = F.normalize(a_feat)
        b_feat = F.normalize(b_feat)
        a_sim = a_feat.mm(a_feat.t())
        b_sim = b_feat.mm(b_feat.t())
        loss = ((a_sim - b_sim) ** 2).sum()
        return loss

    def compute_sim_loss_stage2(self, a_feat, b_feat):
        a_feat = F.normalize(a_feat)
        b_feat = F.normalize(b_feat)
        a_sim = euclidean_dist(a_feat, a_feat)
        b_sim = euclidean_dist(b_feat, b_feat)
        loss = ((a_sim - b_sim) ** 2).sum()
        return loss

    def losses(self, outputs):
        loss_dict = {}
        out_name = 'm1_rgb'
        loss_dict.update(self.model1.losses(outputs[f'out_{out_name}'], prefix=out_name))
        out_name = 'm1_transformed'
        loss_dict.update(self.model1.losses(outputs[f'out_{out_name}'], prefix=out_name))
        out_name = 'm2_transformed'
        loss_dict.update(self.model2.losses(outputs[f'out_{out_name}'], prefix=out_name))

        kl_scale = self._cfg.MODEL.LOSSES.KL.SCALE
        if kl_scale > 0:
            kl_loss12 = self.compute_kl_loss_stage1(outputs['out_m2_transformed']['logits'], outputs['out_m1_rgb']['logits'].detach())
            kl_loss21 = self.compute_kl_loss_stage1(outputs['out_m1_transformed']['logits'], outputs['out_m2_transformed']['logits'].detach())
            loss_dict['kl_loss12'] = {'value': kl_loss12, 'weight': kl_scale}
            loss_dict['kl_loss21'] = {'value': kl_loss21, 'weight': kl_scale}

        sim_scale = self._cfg.MODEL.LOSSES.SIM.SCALE
        if sim_scale > 0:
            sim_loss12 = self.compute_sim_loss_stage1(outputs['out_m1_rgb']['feats'].detach(),
                                             outputs['out_m2_transformed']['feats'])
            sim_loss21 = self.compute_sim_loss_stage1(outputs['out_m2_transformed']['feats'].detach(),
                                             outputs['out_m1_transformed']['feats'])
            loss_dict['sim_loss12'] = {'value': sim_loss12, 'weight': sim_scale}
            loss_dict['sim_loss21'] = {'value': sim_loss21, 'weight': sim_scale}

        return loss_dict

    def uda_losses(self, outputs, memories):
        momemtum = 0.999
        loss_dict = {}
        loss1, logit1 = memories['share'](outputs['out_m1_rgb']['feats'], outputs['out_m1_rgb']['targets'])
        loss2, logit2 = memories['share2'](outputs['out_m2_transformed']['feats'], outputs['out_m2_transformed']['targets'])

        loss_dict['m1_cls_loss'] = {'value': loss1, 'weight': 1.0}
        loss_dict['m2_cls_loss'] = {'value': loss2, 'weight': 1.0}

        clsloss_total = 0.0
        clsloss_total += loss1.item() + loss2.item()
        self.avg_clsloss = (1 - momemtum) * self.avg_clsloss + momemtum * clsloss_total

        kl_scale = self._cfg.MODEL.LOSSES.KL.SCALE
        if kl_scale > 1e-7:
            kl_loss = self.compute_kl_loss_stage2(logit1, logit2) + self.compute_kl_loss_stage2(logit2, logit1)
            loss_dict['kl_loss'] = {'value': kl_loss, 'weight': kl_scale}

        sim_scale = self._cfg.MODEL.LOSSES.SIM.SCALE
        if sim_scale > 1e-7:
            sim_loss = self.compute_sim_loss_stage2(outputs['out_m1_rgb']['feats'], outputs['out_m2_transformed']['feats'])
            loss_dict['sim_loss'] = {'value': sim_loss, 'weight': sim_scale}

        return loss_dict