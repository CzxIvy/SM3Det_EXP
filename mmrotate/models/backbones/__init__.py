# Copyright (c) OpenMMLab. All rights reserved.
from .re_resnet import ReResNet
from .lsknet import LSKNet
from .van import VAN
from .convnext_moe import ConvNeXt_moe_MultiInput, ConvNeXt_moe
from .van_moe import VAN_moe, VAN_moe_MultiInput 
from .lsk_moe import LSKNet_moe_MultiInput
from .convnext_moe_DA import ConvNeXt_DA_MultiInput
from .swin_moe import SwinTransformer_MoE 
from .window_slot_soft_moe import WindowSlotSoftMoE2D
try:
    # NOTE: OPTIONAL BACKBONE IMPORTS - Keep unrelated optional vision
    # transformer dependencies from blocking ConvNeXt MoE build validation.
    from .intern_vit import InternViT
except ImportError:
    InternViT = None

try:
    from .vit_adapter import InternViTAdapter
except ImportError:
    InternViTAdapter = None
__all__ = ['ReResNet','LSKNet', 'ConvNeXt_moe_MultiInput', 'ConvNeXt_DA_MultiInput',
           'ConvNeXt_moe', 'VAN_moe', 'VAN_moe_MultiInput', 'VAN', 'LSKNet_moe_MultiInput','SwinTransformer_MoE',
           'WindowSlotSoftMoE2D']

if InternViT is not None:
    __all__.append('InternViT')

if InternViTAdapter is not None:
    __all__.append('InternViTAdapter')
