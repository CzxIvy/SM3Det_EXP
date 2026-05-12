_base_ = ['./ablation_soft_moe_convnext_t_soft_base.py']

model = dict(
    backbone=dict(
        soft_moe_cfg=dict(
            # NOTE: WINDOW SIZE ABLATION - The soft MoE module already handles
            # non-divisible H/W by padding and reverse cropping internally.
            window_size=14)))