_base_ = ['../SM3Det/SM3Det_convnext_t.py']

model = dict(
    backbone=dict(
        # NOTE: SOFT MOE CONFIG SWITCH - Reuse the original SM3Det ConvNeXt-T
        # insertion points and detector setup, and only swap the MoE branch.
        moe_type='window_slot_soft_moe',
        num_experts=8,
        top_k=3,
        MoE_cfg=dict(
            num_experts=8,
            top_k=3,
            noisy_gating=True,
            gating='cosine'),
        soft_moe_cfg=dict(
            window_size=7,
            temperature=1.0,
            hidden_ratio=4,
            residual_alpha_init=0.0,
            use_shared_expert=True,
            aux_loss_weight=0.01)))