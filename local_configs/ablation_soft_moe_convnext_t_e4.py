_base_ = ['./ablation_soft_moe_convnext_t_soft_base.py']

model = dict(
    backbone=dict(
        num_experts=4,
        MoE_cfg=dict(
            num_experts=4,
            top_k=3,
            noisy_gating=True,
            gating='cosine'),
        soft_moe_cfg=dict(
            num_experts=4)))