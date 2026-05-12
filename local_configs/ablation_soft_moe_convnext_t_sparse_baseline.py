_base_ = ['./SM3Det_convnext_t.py']

model = dict(
    backbone=dict(
        # NOTE: ABLATION SWITCH - Keep the sparse baseline explicit so only the
        # MoE branch selection changes across sparse vs. soft experiments.
        moe_type='sparse'))