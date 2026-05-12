_base_ = ['./ablation_soft_moe_convnext_t_soft_base.py']

model = dict(
    backbone=dict(
        MoE_Block_inds=[[], [], [], [0, 2]]))