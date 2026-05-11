import argparse
import os
import os.path as osp
import sys

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
# NOTE: BUILD-ONLY IMPORT PATHS - Import the installed mmcv-full package first,
# then append the repository root so MMRotate code is loaded from this workspace
# without shadowing the packaged mmcv distribution.
sys.path = [
    path for path in sys.path
    if osp.abspath(path or os.getcwd()) != REPO_ROOT
]

from mmcv import Config

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmrotate.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build sparse and soft MoE configs without training.')
    parser.add_argument(
        '--configs',
        nargs='+',
        default=[
            'configs/SM3Det/SM3Det_convnext_t.py',
            'configs/M3ADet/M3ADet_convnext_t_soft_moe.py'
        ],
        help='Config paths to instantiate.')
    parser.add_argument(
        '--disable-pretrained',
        action='store_true',
        help='Set model.backbone.init_cfg=None before build.')
    return parser.parse_args()


def summarize_model(config_path, model):
    backbone = model.backbone
    summary = {
        'config': config_path,
        'detector': type(model).__name__,
        'backbone': type(backbone).__name__,
        'moe_type': getattr(backbone, 'moe_type', None),
        'num_stages': len(getattr(backbone, 'stages', []))
    }

    soft_block = None
    for stage_idx, stage in enumerate(getattr(backbone, 'stages', [])):
        for block_idx, block in enumerate(stage):
            if type(getattr(block, 'ffn', None)).__name__ == 'WindowSlotSoftMoE2D':
                soft_block = (stage_idx, block_idx)
                break
        if soft_block is not None:
            break
    summary['soft_block'] = soft_block
    return summary


def main():
    args = parse_args()
    for config_path in args.configs:
        cfg = Config.fromfile(config_path)
        if args.disable_pretrained:
            # NOTE: BUILD-ONLY VALIDATION - Allow config instantiation checks to
            # run even when pretrained checkpoints are unavailable locally.
            cfg.model.backbone.init_cfg = None
        model = build_detector(cfg.model)
        summary = summarize_model(config_path, model)
        print(summary)


if __name__ == '__main__':
    main()