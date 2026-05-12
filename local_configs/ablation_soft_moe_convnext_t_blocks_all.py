_base_ = ['./ablation_soft_moe_convnext_t_soft_base.py']

model = dict(
    backbone=dict(
        MoE_Block_inds=[[0, 1, 2], [0, 1, 2], [i for i in range(9)], [0, 1, 2]]))