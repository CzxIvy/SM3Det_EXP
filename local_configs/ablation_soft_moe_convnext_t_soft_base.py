_base_ = ['../configs/M3ADet/M3ADet_convnext_t_soft_moe.py']

model = dict(
    backbone=dict(
        # NOTE: SOFT MOE ABLATION ENTRY - Keep all Window Slot Soft MoE
        # experiment switches under model.backbone so detector routing stays
        # unchanged across ablations.
        moe_type='window_slot_soft_moe',
        soft_moe_cfg=dict(
            window_size=7,
            temperature=1.0,
            hidden_ratio=4,
            residual_alpha_init=0.0,
            use_shared_expert=True,
            aux_loss_weight=0.01)))