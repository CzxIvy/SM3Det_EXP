_base_ = ['./ablation_soft_moe_convnext_t_soft_base.py']

model = dict(
    backbone=dict(
        soft_moe_cfg=dict(
            aux_loss_weight=0.001)))