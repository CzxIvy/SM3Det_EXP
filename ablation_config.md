# Window Slot Soft MoE Ablation Guide

## 1. 开关写在哪里

所有 Window Slot Soft MoE 的消融都建议写在配置文件的 model.backbone 下，不要改 detector 代码。

- 总开关：model.backbone.moe_type
- soft MoE 超参：model.backbone.soft_moe_cfg
- 稀疏 MoE 对照超参：model.backbone.MoE_cfg
- 插入位置：model.backbone.MoE_Block_inds

当前代码约定：

- moe_type='sparse' 时走原始 sparse MoE 路径
- moe_type='window_slot_soft_moe' 时走 Window Slot Soft MoE 路径
- soft 路径训练日志应出现 loss_soft_moe_aux
- sparse 路径训练日志应出现 loss_moe_aux

## 2. 哪些参数对 soft 路径真正生效

对 Window Slot Soft MoE 当前实现真正生效的参数：

- temperature
- num_experts
- MoE_Block_inds
- window_size
- hidden_ratio
- use_shared_expert
- aux_loss_weight

需要特别注意：

- top_k 不是当前 Window Slot Soft MoE 的 active knob
- top_k 只会影响 sparse MoE 路径
- 所以 top_k 更适合作为 sparse 对照实验，而不是 soft 主实验

## 3. 已整理好的配置模板

### 3.1 基线

- sparse 基线：[local_configs/ablation_soft_moe_convnext_t_sparse_baseline.py](local_configs/ablation_soft_moe_convnext_t_sparse_baseline.py)
- soft 基线：[local_configs/ablation_soft_moe_convnext_t_soft_base.py](local_configs/ablation_soft_moe_convnext_t_soft_base.py)

### 3.2 soft 主消融

- temperature=0.5：[local_configs/ablation_soft_moe_convnext_t_temp05.py](local_configs/ablation_soft_moe_convnext_t_temp05.py)
- temperature=2.0：[local_configs/ablation_soft_moe_convnext_t_temp20.py](local_configs/ablation_soft_moe_convnext_t_temp20.py)
- num_experts=4：[local_configs/ablation_soft_moe_convnext_t_e4.py](local_configs/ablation_soft_moe_convnext_t_e4.py)
- num_experts=16：[local_configs/ablation_soft_moe_convnext_t_e16.py](local_configs/ablation_soft_moe_convnext_t_e16.py)
- 仅最后一层插入：[local_configs/ablation_soft_moe_convnext_t_blocks_last1.py](local_configs/ablation_soft_moe_convnext_t_blocks_last1.py)
- 最后三层插入：[local_configs/ablation_soft_moe_convnext_t_blocks_last3.py](local_configs/ablation_soft_moe_convnext_t_blocks_last3.py)
- 各 stage 偶数块插入：[local_configs/ablation_soft_moe_convnext_t_blocks_even.py](local_configs/ablation_soft_moe_convnext_t_blocks_even.py)
- 所有块插入：[local_configs/ablation_soft_moe_convnext_t_blocks_all.py](local_configs/ablation_soft_moe_convnext_t_blocks_all.py)

### 3.3 sparse 对照消融

- sparse top_k=1：[local_configs/ablation_soft_moe_convnext_t_sparse_topk1.py](local_configs/ablation_soft_moe_convnext_t_sparse_topk1.py)
- sparse top_k=4：[local_configs/ablation_soft_moe_convnext_t_sparse_topk4.py](local_configs/ablation_soft_moe_convnext_t_sparse_topk4.py)

### 3.4 之前已经补过的 soft 变体

- window_size=14：[local_configs/ablation_soft_moe_convnext_t_ws14.py](local_configs/ablation_soft_moe_convnext_t_ws14.py)
- no shared expert：[local_configs/ablation_soft_moe_convnext_t_no_shared.py](local_configs/ablation_soft_moe_convnext_t_no_shared.py)
- aux_loss_weight=0.001：[local_configs/ablation_soft_moe_convnext_t_aux001.py](local_configs/ablation_soft_moe_convnext_t_aux001.py)
- hidden_ratio=2：[local_configs/ablation_soft_moe_convnext_t_hidden2.py](local_configs/ablation_soft_moe_convnext_t_hidden2.py)

## 4. 推荐的消融顺序

建议按下面顺序做，避免一次改太多因素：

1. sparse 基线 vs soft 基线
2. temperature
3. num_experts
4. window_size / hidden_ratio / shared expert / aux loss weight
5. MoE_Block_inds
6. sparse top_k 对照

这样更容易判断收益来自 soft 路由本身，还是来自容量、插入深度、或 sparse 门控强度。

## 5. 每次做消融的标准步骤

### 步骤 1：先做 build-only

先确认配置能实例化：

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/misc/check_moe_build.py --disable-pretrained --configs local_configs/ablation_soft_moe_convnext_t_soft_base.py
```

如果你要一次检查多份配置，可以把多个配置路径都写到 --configs 后面。

### 步骤 2：再做 smoke train

前提：

- data/SOI_Det 数据树已经就位
- 如果本地没有 pretrained，就临时加 model.backbone.init_cfg=None

示例：

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_soft_base.py \
	--work-dir work_dirs/ablation_soft_base \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

如果只想做 1 iter smoke：

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_soft_base.py \
	--work-dir work_dirs/ablation_soft_base_smoke \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None runner.max_iters=1 data.samples_per_gpu=1 data.workers_per_gpu=0 log_config.interval=1 checkpoint_config.interval=1
```

### 步骤 3：检查日志关键项

soft 配置要重点看：

- 是否出现 loss_soft_moe_aux
- 是否没有出现 loss_moe_aux

sparse 配置要重点看：

- 是否出现 loss_moe_aux
- 是否没有出现 loss_soft_moe_aux

两者都要确认：

- DSO 的 reweight_losses 没有把这两个 aux loss 当成任务损失

### 步骤 4：记录实验对照表

建议每次记录：

- 配置文件名
- moe_type
- temperature
- num_experts
- top_k
- MoE_Block_inds
- window_size
- hidden_ratio
- use_shared_expert
- aux_loss_weight
- build 是否通过
- smoke train 是否通过
- 是否看到对应 aux loss
- 最终指标

## 6. 常用命令模板

### soft 基线

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_soft_base.py \
	--work-dir work_dirs/ablation_soft_base \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

### temperature 消融

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_temp05.py \
	--work-dir work_dirs/ablation_soft_temp05 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_temp20.py \
	--work-dir work_dirs/ablation_soft_temp20 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

### num_experts 消融

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_e4.py \
	--work-dir work_dirs/ablation_soft_e4 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_e16.py \
	--work-dir work_dirs/ablation_soft_e16 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

### MoE_Block_inds 消融

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_blocks_last1.py \
	--work-dir work_dirs/ablation_soft_blocks_last1 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_blocks_all.py \
	--work-dir work_dirs/ablation_soft_blocks_all \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

### sparse top_k 对照

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_sparse_topk1.py \
	--work-dir work_dirs/ablation_sparse_topk1 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_soft_moe_convnext_t_sparse_topk4.py \
	--work-dir work_dirs/ablation_sparse_topk4 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

## 7. 一个实用建议

如果你接下来要大批量跑实验，建议始终保留一个最小公共基线：

- sparse_baseline
- soft_base

所有新实验都只相对 soft_base 改一个变量。

这样后续看结果时，不会把 branch 差异和超参差异混在一起。

# Hierarchical Staged Micro Head Ablation Guide

## 1. 开关写在哪里

所有 micro head 的消融都建议写在配置文件的 model.micro_head 下，不要改 detector 代码。

- 总开关：model.micro_head.enable
- 分层/分阶段超参：model.micro_head.*
- 原始 SM3Det 基线不需要写 micro_head

当前代码约定：

- 原始配置 [configs/SM3Det/SM3Det_convnext_t.py](configs/SM3Det/SM3Det_convnext_t.py) 不启用 micro head
- 新配置 [configs/M3ADet/M3ADet_convnext_t_micro_head.py](configs/M3ADet/M3ADet_convnext_t_micro_head.py) 启用 micro head
- micro head 的辅助监督会生成 sar_loss_micro_*、rgb_loss_micro_*、ifr_loss_micro_*
- 这些 micro losses 不进入 DSO 的 reweight_losses

## 2. 哪些参数对 micro head 真正生效

当前实现里，下面这些参数是真正 active 的：

- feat_channels
- feat_levels
- use_context
- context_level
- small_area_thr
- small_size_thr
- heatmap_radius
- center_loss_weight
- offset_loss_weight
- size_loss_weight
- enhance_weight
- detach_heatmap
- use_feat_enhance
- use_inference_enhance
- loss_type

需要特别注意：

- feat_strides 在配置里保留为默认值，但 detector 实际会根据当前模态的 feature shape 和 batch_input_shape 动态推断 stride
- 这是为了兼容 SAR 分支使用 start_level=1，而 RGB/IFR 使用默认 start_level=0 的现有 SM3Det 结构

## 3. 已整理好的配置模板

### 3.1 基线

- 原始 SM3Det 基线：[local_configs/ablation_micro_head_convnext_t_baseline.py](local_configs/ablation_micro_head_convnext_t_baseline.py)
- micro head 基线：[local_configs/ablation_micro_head_convnext_t_micro_base.py](local_configs/ablation_micro_head_convnext_t_micro_base.py)

### 3.2 分阶段消融

- 仅 Stage 1 辅助监督，不做特征增强：[local_configs/ablation_micro_head_convnext_t_stage1_only.py](local_configs/ablation_micro_head_convnext_t_stage1_only.py)
- 训练时增强，推理时关闭增强：[local_configs/ablation_micro_head_convnext_t_no_infer_enhance.py](local_configs/ablation_micro_head_convnext_t_no_infer_enhance.py)

### 3.3 分层特征消融

- 只用最浅层特征：[local_configs/ablation_micro_head_convnext_t_levels_p2_only.py](local_configs/ablation_micro_head_convnext_t_levels_p2_only.py)
- 打开 context 融合：[local_configs/ablation_micro_head_convnext_t_context_on.py](local_configs/ablation_micro_head_convnext_t_context_on.py)

### 3.4 小目标定义消融

- 更严格的小目标阈值 16x16：[local_configs/ablation_micro_head_convnext_t_area256.py](local_configs/ablation_micro_head_convnext_t_area256.py)
- 更宽松的小目标阈值 48x48：[local_configs/ablation_micro_head_convnext_t_area2304.py](local_configs/ablation_micro_head_convnext_t_area2304.py)

### 3.5 Heatmap 监督消融

- 无邻域扩散，只监督中心点：[local_configs/ablation_micro_head_convnext_t_radius0.py](local_configs/ablation_micro_head_convnext_t_radius0.py)
- 更大邻域半径：[local_configs/ablation_micro_head_convnext_t_radius2.py](local_configs/ablation_micro_head_convnext_t_radius2.py)

### 3.6 特征增强强度消融

- enhance_weight=0.25：[local_configs/ablation_micro_head_convnext_t_enhance025.py](local_configs/ablation_micro_head_convnext_t_enhance025.py)
- enhance_weight=0.75：[local_configs/ablation_micro_head_convnext_t_enhance075.py](local_configs/ablation_micro_head_convnext_t_enhance075.py)

### 3.7 Micro branch 容量消融

- feat_channels=64：[local_configs/ablation_micro_head_convnext_t_feat64.py](local_configs/ablation_micro_head_convnext_t_feat64.py)
- feat_channels=192：[local_configs/ablation_micro_head_convnext_t_feat192.py](local_configs/ablation_micro_head_convnext_t_feat192.py)

## 4. 推荐的消融顺序

建议按下面顺序做，避免一次改太多因素：

1. 原始 SM3Det 基线 vs micro head 基线
2. stage1_only / no_infer_enhance
3. feat_levels / context_on
4. small_area_thr
5. heatmap_radius
6. enhance_weight
7. feat_channels

这样更容易区分收益来自：

- 小目标辅助监督本身
- heatmap 引导的浅层增强
- 特征层级选择
- 小目标定义阈值
- branch 容量与增强强度

## 5. 每次做消融的标准步骤

### 步骤 1：先做 build-only

先确认配置能实例化：

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/misc/check_moe_build.py --disable-pretrained --configs local_configs/ablation_micro_head_convnext_t_micro_base.py
```

### 步骤 2：再做 micro head 独立 smoke

这个 smoke 不依赖真实数据集：

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/misc/test_micro_head_smoke.py
```

### 步骤 3：再做最小训练 smoke

前提：

- data/SOI_Det 数据树已经就位
- 如果本地没有 pretrained，就临时加 model.backbone.init_cfg=None

示例：

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_micro_head_convnext_t_micro_base.py \
	--work-dir work_dirs/ablation_micro_base \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

如果只想做 1 iter smoke：

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_micro_head_convnext_t_micro_base.py \
	--work-dir work_dirs/ablation_micro_base_smoke \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None runner.max_iters=1 data.samples_per_gpu=1 data.workers_per_gpu=0 log_config.interval=1 checkpoint_config.interval=1
```

### 步骤 4：检查日志关键项

micro 配置要重点看：

- 是否出现 sar_loss_micro_center / sar_loss_micro_offset / sar_loss_micro_size
- 是否出现 rgb_loss_micro_center / rgb_loss_micro_offset / rgb_loss_micro_size
- 是否出现 ifr_loss_micro_center / ifr_loss_micro_offset / ifr_loss_micro_size

同时确认：

- 原始 sar_loss_cls / sar_loss_bbox / sar_loss_dfl 仍然存在
- 原始 rgb / ifr 的 rpn 与 roi loss 仍然存在
- DSO 的 reweight_losses 没有包含任何 *_loss_micro_* 键

### 步骤 5：记录实验对照表

建议每次记录：

- 配置文件名
- 是否启用 micro head
- feat_levels
- use_context
- small_area_thr
- heatmap_radius
- enhance_weight
- feat_channels
- use_feat_enhance
- use_inference_enhance
- build 是否通过
- smoke train 是否通过
- 是否看到对应 micro loss
- 最终指标

## 6. 常用命令模板

### micro head 基线

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_micro_head_convnext_t_micro_base.py \
	--work-dir work_dirs/ablation_micro_base \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

### stage1_only 消融

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_micro_head_convnext_t_stage1_only.py \
	--work-dir work_dirs/ablation_micro_stage1_only \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

### feature hierarchy 消融

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_micro_head_convnext_t_levels_p2_only.py \
	--work-dir work_dirs/ablation_micro_levels_p2_only \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_micro_head_convnext_t_context_on.py \
	--work-dir work_dirs/ablation_micro_context_on \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

### small object threshold 消融

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_micro_head_convnext_t_area256.py \
	--work-dir work_dirs/ablation_micro_area256 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_micro_head_convnext_t_area2304.py \
	--work-dir work_dirs/ablation_micro_area2304 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

### enhancement strength 消融

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_micro_head_convnext_t_enhance025.py \
	--work-dir work_dirs/ablation_micro_enhance025 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

```bash
cd /root/autodl-tmp/SM3Det_EXP
PYTHONPATH=. python tools/train.py local_configs/ablation_micro_head_convnext_t_enhance075.py \
	--work-dir work_dirs/ablation_micro_enhance075 \
	--no-validate \
	--cfg-options model.backbone.init_cfg=None
```

## 7. 一个实用建议

如果你接下来要批量跑 micro head 消融，建议始终保留两个公共对照：

- 原始 SM3Det baseline
- micro_base

所有新实验都只相对 micro_base 改一个变量。

这样后续看结果时，不会把“是否启用 micro head”的收益和“micro head 内部超参变化”的收益混在一起。

## 8. 批量运行脚本

我已经补了一份批量运行脚本：

- [tools/misc/run_micro_head_ablation.sh](tools/misc/run_micro_head_ablation.sh)

它支持四种模式：

- list：打印当前 micro head 全部 ablation 配置
- build：一次性做 build-only
- smoke：一次性做 1 iter smoke train
- train：一次性串行正式训练

### 8.1 先看有哪些配置

```bash
cd /root/autodl-tmp/SM3Det_EXP
bash tools/misc/run_micro_head_ablation.sh --mode list
```

### 8.2 一次性做 build-only

```bash
cd /root/autodl-tmp/SM3Det_EXP
bash tools/misc/run_micro_head_ablation.sh --mode build
```

### 8.3 一次性做 1 iter smoke train

```bash
cd /root/autodl-tmp/SM3Det_EXP
bash tools/misc/run_micro_head_ablation.sh --mode smoke --gpus 1
```

### 8.4 一次性串行正式训练

```bash
cd /root/autodl-tmp/SM3Det_EXP
bash tools/misc/run_micro_head_ablation.sh --mode train --gpus 1
```

### 8.5 只跑子集

例如只跑 micro_base、area256 和 feat64：

```bash
cd /root/autodl-tmp/SM3Det_EXP
bash tools/misc/run_micro_head_ablation.sh --mode smoke --run micro_base,area256,feat64
```

### 8.6 多卡训练

如果想走分布式单机多卡：

```bash
cd /root/autodl-tmp/SM3Det_EXP
bash tools/misc/run_micro_head_ablation.sh --mode train --gpus 2
```

脚本会在 gpus > 1 时自动切到 [tools/dist_train.sh](tools/dist_train.sh)。

### 8.7 透传额外参数

如果你想传额外训练参数，可以在 -- 后面继续追加：

```bash
cd /root/autodl-tmp/SM3Det_EXP
bash tools/misc/run_micro_head_ablation.sh --mode smoke --run micro_base -- --seed 0
```

### 8.8 默认行为说明

- 默认会给训练注入 model.backbone.init_cfg=None，避免本地没有 pretrained 时直接卡住
- 如果你本地 pretrained 已经齐全，可以加 --keep-pretrained 取消这个覆盖
- smoke 模式会自动附加：runner.max_iters=1、data.samples_per_gpu=1、data.workers_per_gpu=0、log_config.interval=1、checkpoint_config.interval=1
- work_dir 默认写到 work_dirs/micro_head_ablation/<ablation_name>
