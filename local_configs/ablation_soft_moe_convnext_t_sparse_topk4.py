_base_ = ['./ablation_soft_moe_convnext_t_sparse_baseline.py']

model = dict(
    backbone=dict(
        top_k=4,
        MoE_cfg=dict(
            num_experts=8,
            top_k=4,
            noisy_gating=True,
            gating='cosine')))