_base_ = ['./ablation_micro_head_convnext_t_micro_base.py']

model = dict(
    micro_head=dict(
        enhance_weight=0.25))