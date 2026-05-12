import os
import os.path as osp
import sys

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path = [
    path for path in sys.path
    if osp.abspath(path or os.getcwd()) != REPO_ROOT
]

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from mmrotate.models.builder import build_head


def main():
    head = build_head(
        dict(
            type='HierarchicalStagedMicroHead',
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
            loss_type='focal'))

    feats = [
        torch.randn(2, 256, 128, 128, requires_grad=True),
        torch.randn(2, 256, 64, 64, requires_grad=True),
        torch.randn(2, 256, 32, 32, requires_grad=True),
    ]
    gt_bboxes = [
        torch.tensor([[12.0, 14.0, 28.0, 30.0]], dtype=torch.float32),
        torch.tensor([[80.0, 96.0, 18.0, 16.0, 0.0]], dtype=torch.float32),
    ]
    gt_labels = [
        torch.tensor([1], dtype=torch.long),
        torch.tensor([2], dtype=torch.long),
    ]
    img_metas = [
        dict(img_shape=(512, 512, 3), batch_input_shape=(512, 512)),
        dict(img_shape=(512, 512, 3), batch_input_shape=(512, 512)),
    ]

    pred_dict = head(feats)
    enhanced_feats = head.enhance_feats(feats, pred_dict)
    losses = head.loss(pred_dict, gt_bboxes, gt_labels, img_metas)

    assert len(pred_dict['center_logits']) == 2
    assert pred_dict['center_logits'][0].shape == (2, 1, 128, 128)
    assert pred_dict['center_logits'][1].shape == (2, 1, 64, 64)
    assert pred_dict['offset_preds'][0].shape == (2, 2, 128, 128)
    assert pred_dict['size_preds'][1].shape == (2, 2, 64, 64)
    assert len(enhanced_feats) == len(feats)
    for source_feat, enhanced_feat in zip(feats, enhanced_feats):
        assert source_feat.shape == enhanced_feat.shape

    polygon = torch.tensor(
        [[10.0, 10.0, 18.0, 10.0, 18.0, 20.0, 10.0, 20.0]],
        dtype=torch.float32)
    polygon_centers, polygon_sizes, polygon_areas = head.get_bbox_center_size_area(
        polygon)
    assert polygon_centers.shape == (1, 2)
    assert polygon_sizes.shape == (1, 2)
    assert polygon_areas.shape == (1, )

    total_loss = sum(losses.values()) + sum(feat.sum() * 0 for feat in enhanced_feats)
    total_loss.backward()
    for key, value in losses.items():
        assert value.ndim == 0, f'{key} is not scalar: {value.shape}'

    print('Micro head smoke test passed.')
    print({key: float(value.detach().cpu()) for key, value in losses.items()})


if __name__ == '__main__':
    main()