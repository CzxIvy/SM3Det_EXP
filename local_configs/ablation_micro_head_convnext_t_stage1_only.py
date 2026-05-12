_base_ = ['./ablation_micro_head_convnext_t_micro_base.py']

model = dict(
    micro_head=dict(
        # NOTE: STAGE DECOUPLING ABLATION - Keep the micro supervision branch
        # but disable feature enhancement to isolate stage-1 candidate sensing.
        use_feat_enhance=False,
        use_inference_enhance=False))