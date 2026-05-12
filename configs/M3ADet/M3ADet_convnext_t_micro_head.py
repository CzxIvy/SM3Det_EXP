_base_ = ['../SM3Det/SM3Det_convnext_t.py']

model = dict(
    micro_head=dict(
        enable=True,
        type='HierarchicalStagedMicroHead',
        in_channels=256,
        feat_channels=128,
        feat_levels=(0, 1),
        # NOTE: The detector infers the actual per-modality strides from the
        # feature shapes at runtime. These defaults match the RGB/IFR shallow
        # FPN levels and act as a fallback for standalone micro-head tests.
        feat_strides=(4, 8),
        context_level=2,
        use_context=False,
        small_area_thr=1024.0,
        small_size_thr=-1.0,
        heatmap_radius=1,
        center_loss_weight=0.2,
        offset_loss_weight=0.05,
        size_loss_weight=0.05,
        enhance_weight=0.5,
        detach_heatmap=True,
        use_feat_enhance=True,
        use_inference_enhance=True,
        loss_type='focal'))