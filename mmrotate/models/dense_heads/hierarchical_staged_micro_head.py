import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_HEADS


def sigmoid_focal_loss(logits,
                       targets,
                       alpha=0.25,
                       gamma=2.0,
                       reduction='mean'):
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    return loss.mean()


@ROTATED_HEADS.register_module()
class HierarchicalStagedMicroHead(nn.Module):
    """Lightweight micro-object branch inserted between neck and heads.

    The module predicts center/offset/size targets on shallow FPN features and
    uses the center heatmap as a spatial attention map to enhance those same
    shallow features before they are consumed by the original detection heads.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=128,
                 feat_levels=(0, 1),
                 feat_strides=(4, 8),
                 context_level=2,
                 use_context=False,
                 small_area_thr=1024.0,
                 small_size_thr=-1.0,
                 heatmap_radius=1,
                 center_loss_weight=0.2,
                 offset_loss_weight=0.05,
                 size_loss_weight=0.05,
                 enhance_weight=0.5,
                 detach_heatmap=True,
                 use_feat_enhance=True,
                 use_inference_enhance=True,
                 loss_type='focal',
                 enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.feat_levels = tuple(feat_levels)
        self.feat_strides = tuple(feat_strides)
        self.context_level = context_level
        self.use_context = use_context
        self.small_area_thr = float(small_area_thr)
        self.small_size_thr = float(small_size_thr)
        self.heatmap_radius = int(heatmap_radius)
        self.center_loss_weight = float(center_loss_weight)
        self.offset_loss_weight = float(offset_loss_weight)
        self.size_loss_weight = float(size_loss_weight)
        self.enhance_weight = float(enhance_weight)
        self.detach_heatmap = detach_heatmap
        self.use_feat_enhance = use_feat_enhance
        self.use_inference_enhance = use_inference_enhance
        self.loss_type = loss_type

        if self.loss_type not in ('focal', 'bce'):
            raise ValueError(f'Unsupported loss_type: {self.loss_type}')

        self.level_blocks = nn.ModuleList()
        self.center_heads = nn.ModuleList()
        self.offset_heads = nn.ModuleList()
        self.size_heads = nn.ModuleList()
        self.context_projs = nn.ModuleList()

        for _ in self.feat_levels:
            self.level_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feat_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                    nn.ReLU(inplace=True)))
            self.center_heads.append(nn.Conv2d(feat_channels, 1, 1))
            self.offset_heads.append(nn.Conv2d(feat_channels, 2, 1))
            self.size_heads.append(nn.Conv2d(feat_channels, 2, 1))
            self.context_projs.append(nn.Conv2d(in_channels, feat_channels, 1))

    def forward(self, feats):
        """Predict micro-object targets on selected feature levels.

        Args:
            feats (list[Tensor] | tuple[Tensor]): each feature is [B, C, H, W].

        Returns:
            dict: center/offset/size predictions aligned with feat_levels.
        """
        feats = list(feats)
        pred_dict = dict(center_logits=[], offset_preds=[], size_preds=[])

        for pred_idx, feat_level in enumerate(self.feat_levels):
            feat = feats[feat_level]
            hidden = self.level_blocks[pred_idx](feat)

            if self.use_context and 0 <= self.context_level < len(feats):
                context_feat = feats[self.context_level]
                context_feat = self.context_projs[pred_idx](context_feat)
                context_feat = F.interpolate(
                    context_feat,
                    size=hidden.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
                hidden = hidden + context_feat

            pred_dict['center_logits'].append(self.center_heads[pred_idx](hidden))
            pred_dict['offset_preds'].append(self.offset_heads[pred_idx](hidden))
            pred_dict['size_preds'].append(self.size_heads[pred_idx](hidden))

        return pred_dict

    def enhance_feats(self, feats, pred_dict, enable=None):
        """Apply heatmap-guided enhancement to shallow features.

        Args:
            feats (list[Tensor] | tuple[Tensor]): original FPN features.
            pred_dict (dict): output of :meth:`forward`.
            enable (bool | None): optional runtime override.

        Returns:
            list[Tensor]: feature list with the same length and shapes.
        """
        feats = list(feats)
        should_enhance = self.use_feat_enhance if enable is None else enable
        if not should_enhance:
            return feats

        for pred_idx, feat_level in enumerate(self.feat_levels):
            attn = pred_dict['center_logits'][pred_idx].sigmoid()
            if self.detach_heatmap:
                attn = attn.detach()
            feats[feat_level] = feats[feat_level] * (1 + self.enhance_weight * attn)

        return feats

    def get_bbox_center_size_area(self, gt_bboxes):
        """Parse HBB / OBB / polygon boxes into centers, sizes and areas."""
        if gt_bboxes.numel() == 0:
            empty = gt_bboxes.new_zeros((0, 2))
            return empty, empty, gt_bboxes.new_zeros((0, ))

        if gt_bboxes.size(-1) == 4:
            x1, y1, x2, y2 = gt_bboxes.unbind(dim=-1)
            widths = x2 - x1
            heights = y2 - y1
            centers = torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dim=-1)
            sizes = torch.stack((widths, heights), dim=-1)
            areas = widths * heights
        elif gt_bboxes.size(-1) == 5:
            centers = gt_bboxes[:, :2]
            sizes = gt_bboxes[:, 2:4]
            areas = sizes[:, 0] * sizes[:, 1]
        elif gt_bboxes.size(-1) == 8:
            points = gt_bboxes.view(-1, 4, 2)
            centers = points.mean(dim=1)
            x_coords = points[..., 0]
            y_coords = points[..., 1]
            widths = x_coords.max(dim=1).values - x_coords.min(dim=1).values
            heights = y_coords.max(dim=1).values - y_coords.min(dim=1).values
            sizes = torch.stack((widths, heights), dim=-1)
            shifted_x = torch.roll(x_coords, shifts=-1, dims=1)
            shifted_y = torch.roll(y_coords, shifts=-1, dims=1)
            areas = 0.5 * torch.abs(
                (x_coords * shifted_y - y_coords * shifted_x).sum(dim=1))
        else:
            raise ValueError(f'Unsupported bbox shape: {gt_bboxes.shape}')

        return centers, sizes, areas

    def _select_small_targets(self, gt_bboxes):
        centers, sizes, areas = self.get_bbox_center_size_area(gt_bboxes)
        if centers.numel() == 0:
            return centers, sizes, areas, gt_bboxes.new_zeros((0, ), dtype=torch.bool)

        widths = sizes[:, 0]
        heights = sizes[:, 1]
        valid_mask = (widths > 0) & (heights > 0) & (areas > 0)
        if self.small_size_thr > 0:
            small_mask = torch.maximum(widths, heights) <= self.small_size_thr
        else:
            small_mask = areas <= self.small_area_thr
        keep_mask = valid_mask & small_mask
        return centers[keep_mask], sizes[keep_mask], areas[keep_mask], keep_mask

    def _draw_heatmap(self, center_target, center_x, center_y):
        if self.heatmap_radius <= 0:
            center_target[center_y, center_x] = 1
            return

        height, width = center_target.shape
        radius = self.heatmap_radius
        sigma = max(radius / 2.0, 1e-6)
        for delta_y in range(-radius, radius + 1):
            target_y = center_y + delta_y
            if target_y < 0 or target_y >= height:
                continue
            for delta_x in range(-radius, radius + 1):
                target_x = center_x + delta_x
                if target_x < 0 or target_x >= width:
                    continue
                dist2 = float(delta_x * delta_x + delta_y * delta_y)
                value = math.exp(-dist2 / (2 * sigma * sigma))
                center_target[target_y, target_x] = max(
                    center_target[target_y, target_x], value)

    def _build_level_targets(self,
                             pred_shape,
                             stride,
                             gt_bboxes,
                             img_metas,
                             dtype,
                             device):
        batch_size, _, feat_h, feat_w = pred_shape
        center_target = torch.zeros(
            (batch_size, 1, feat_h, feat_w), dtype=dtype, device=device)
        offset_target = torch.zeros(
            (batch_size, 2, feat_h, feat_w), dtype=dtype, device=device)
        size_target = torch.zeros(
            (batch_size, 2, feat_h, feat_w), dtype=dtype, device=device)
        pos_mask = torch.zeros(
            (batch_size, 1, feat_h, feat_w), dtype=dtype, device=device)

        num_micro_gt = 0
        for batch_idx in range(batch_size):
            if batch_idx >= len(gt_bboxes):
                continue
            centers, sizes, _, _ = self._select_small_targets(gt_bboxes[batch_idx])
            if centers.numel() == 0:
                continue

            num_micro_gt += centers.size(0)
            for target_idx in range(centers.size(0)):
                center_x = centers[target_idx, 0] / stride
                center_y = centers[target_idx, 1] / stride
                feat_x = int(torch.floor(center_x).item())
                feat_y = int(torch.floor(center_y).item())

                if feat_x < 0 or feat_x >= feat_w or feat_y < 0 or feat_y >= feat_h:
                    continue

                self._draw_heatmap(center_target[batch_idx, 0], feat_x, feat_y)
                pos_mask[batch_idx, 0, feat_y, feat_x] = 1
                offset_target[batch_idx, 0, feat_y, feat_x] = center_x - feat_x
                offset_target[batch_idx, 1, feat_y, feat_x] = center_y - feat_y
                size_target[batch_idx, 0, feat_y, feat_x] = torch.log(
                    sizes[target_idx, 0] / stride + 1e-6)
                size_target[batch_idx, 1, feat_y, feat_x] = torch.log(
                    sizes[target_idx, 1] / stride + 1e-6)

        return center_target, offset_target, size_target, pos_mask, num_micro_gt

    def loss(self,
             pred_dict,
             gt_bboxes,
             gt_labels,
             img_metas,
             feat_strides=None):
        """Compute micro-object losses on the selected feature levels."""
        center_logits = pred_dict['center_logits']
        offset_preds = pred_dict['offset_preds']
        size_preds = pred_dict['size_preds']
        feat_strides = tuple(self.feat_strides if feat_strides is None else feat_strides)

        dtype = center_logits[0].dtype
        device = center_logits[0].device
        loss_center = center_logits[0].new_zeros(())
        loss_offset = center_logits[0].new_zeros(())
        loss_size = center_logits[0].new_zeros(())

        total_gt = sum(len(each) for each in gt_bboxes)
        if total_gt == 0:
            return dict(
                loss_micro_center=loss_center,
                loss_micro_offset=loss_offset,
                loss_micro_size=loss_size)

        total_micro_gt = 0
        for level_idx, stride in enumerate(feat_strides):
            center_target, offset_target, size_target, pos_mask, num_micro_gt = \
                self._build_level_targets(
                    center_logits[level_idx].shape,
                    stride,
                    gt_bboxes,
                    img_metas,
                    dtype,
                    device)
            total_micro_gt += num_micro_gt
            if num_micro_gt == 0:
                continue

            if self.loss_type == 'focal':
                level_center_loss = sigmoid_focal_loss(
                    center_logits[level_idx], center_target, reduction='mean')
            else:
                level_center_loss = F.binary_cross_entropy_with_logits(
                    center_logits[level_idx], center_target, reduction='mean')

            pos_count = pos_mask.sum().clamp_min(1.0)
            level_offset_loss = F.l1_loss(
                offset_preds[level_idx] * pos_mask,
                offset_target * pos_mask,
                reduction='sum') / pos_count
            level_size_loss = F.l1_loss(
                size_preds[level_idx] * pos_mask,
                size_target * pos_mask,
                reduction='sum') / pos_count

            loss_center = loss_center + level_center_loss
            loss_offset = loss_offset + level_offset_loss
            loss_size = loss_size + level_size_loss

        if total_micro_gt == 0:
            return dict(
                loss_micro_center=center_logits[0].new_zeros(()),
                loss_micro_offset=center_logits[0].new_zeros(()),
                loss_micro_size=center_logits[0].new_zeros(()))

        return dict(
            loss_micro_center=loss_center * self.center_loss_weight,
            loss_micro_offset=loss_offset * self.offset_loss_weight,
            loss_micro_size=loss_size * self.size_loss_weight)