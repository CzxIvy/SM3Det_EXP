# Soft MoE 与 Hierarchical Staged Micro Head 实验章节

## 1. 实验目标与验证思路

本章旨在从实验角度验证两项新增模块在当前三源遥感检测系统中的有效性与合理性：

- Window Slot Soft MoE
- Hierarchical Staged Micro Head

其中，Soft MoE 侧重于提升主干特征生成阶段的局部专家化建模能力，Micro Head 侧重于提升浅层特征对小目标的显式感知能力。因此，本章实验设计围绕如下三个核心问题展开：

1. Soft MoE 相比原始 sparse MoE 是否能够在当前三源检测框架中带来更稳定或更优的检测性能。
2. Micro Head 是否能够在不重写原始检测头的前提下，显著提升小目标检测能力。
3. 两类模块的性能变化究竟来源于哪些具体设计因素，包括路由平滑度、专家容量、插入位置、特征层级、小目标定义、热图监督范围和增强强度等。

与单纯给出最终指标不同，本章强调“主结果 + 系统消融 + 现象解释”的完整实验逻辑。考虑到当前仓库中尚未沉淀全部正式训练结果，以下文本采用“正式论文写法 + 结果占位符”的方式组织，可在后续实验完成后直接替换具体数值。

## 2. 实验设置

### 2.1 数据集与多源训练设置

当前工程的数据配置定义在 [configs/_base_/datasets/SOI_Det.py](../../configs/_base_/datasets/SOI_Det.py)。系统采用三源联合训练设置，分别对应：

- SAR 数据：`SARDet_50K`
- RGB 数据：`DOTA_800pix`
- IFR 数据：`DroneVehicle`

训练集分别为：

- SAR：`data/SOI_Det/SARDet_50K/Annotations/after_merge_train.json`
- RGB：`data/SOI_Det/DOTA_800pix/train/annfiles/`
- IFR：`data/SOI_Det/DroneVehicle/dota_train/annfiles/`

验证集分别为：

- SAR：`after_merge_test.json`
- RGB：`DOTA_800pix/val/annfiles/`
- IFR：`DroneVehicle/dota_test/annfiles/`

输入尺度统一设置为 800，三条分支的训练数据通过 `ConcatDataset` 组织，并在 dataloader 中采用多数据源联合采样。基础配置中 `source_ratio=[1, 1, 1]`，表示三种模态在训练阶段以均衡比例参与迭代。

这一设置的目的在于：

- 保持三源任务的同步训练。
- 使 backbone、neck 与 detector 可以在统一框架下共享参数。
- 在保持多模态互补性的同时，分别评价模块对 SAR、RGB 和 IFR 场景的适配能力。

### 2.2 数据预处理与增强

根据 [configs/_base_/datasets/SOI_Det.py](../../configs/_base_/datasets/SOI_Det.py)，三种模态使用了不同但互相兼容的数据增强流程。

对于 SAR 分支，采用：

- Resize 到 800 尺度
- RandomFlip
- Normalize
- Pad

对于 RGB 与 IFR 分支，采用：

- 旋转感知缩放 `RResize`
- 多方向随机翻转 `RRandomFlip`
- 多边形随机旋转 `PolyRandomRotate`
- Normalize
- Pad

这种数据增强设计与当前三源旋转检测任务的目标表示形式保持一致，有助于提升检测头对旋转框与多边形标注的适应性。

### 2.3 评价指标

基础评价配置定义在 [configs/_base_/schedules/schedule_1x.py](../../configs/_base_/schedules/schedule_1x.py)，当前默认评价指标为：

- `mAP`

在正式论文撰写时，建议将主结果表拆分为以下三类指标：

- 全局总体 mAP
- 分模态 mAP：SAR / RGB / IFR
- 面向小目标的细粒度指标，例如按面积区间拆分后的 AP 或 Recall

如果当前评估脚本只直接提供总体 mAP，则论文中可先报告总体 mAP，再补充按预测可视化和误检/漏检现象进行小目标分析。若后续增加尺度划分统计，建议进一步报告：

- AP_small
- AP_medium
- AP_large

这样可以更直接地验证 Micro Head 是否真正作用于小目标。

### 2.4 网络结构与基础配置

当前主要实验的基础结构来自 [configs/SM3Det/SM3Det_convnext_t.py](../../configs/SM3Det/SM3Det_convnext_t.py)，核心设置如下：

- 检测器：`TriSourceDetector`
- 主干：`ConvNeXt_moe_MultiInput`
- 主干规模：ConvNeXt-Tiny
- 颈部：`MultitaskFPN`
- SAR 检测头：`GFLHead`
- RGB/IFR 检测头：`OrientedRPNHead + OrientedStandardRoIHead`

优化器采用 AdamW，基础学习率为 $1 \times 10^{-4}$，并启用 `fp16` 动态 loss scale。训练调度继承 1x 基础计划，总训练轮数为 12 epoch，学习率衰减节点为第 8 与第 11 个 epoch。

与默认 schedule 相比，主配置中显式覆盖了优化器，因此在论文中建议统一写为：

- 优化器：AdamW
- 初始学习率：$1 \times 10^{-4}$
- Weight decay：0.05
- 训练轮数：12 epoch
- 混合精度：开启

### 2.5 对比方法与实验分组

为了清晰区分各模块的贡献来源，本章实验分为三组：

#### 2.5.1 主结果组

- 原始 SM3Det 基线
- Sparse MoE 基线
- Soft MoE 基线
- Micro Head 基线
- 可选扩展：Soft MoE 与 Micro Head 联合模型

其中：

- Soft MoE 的配置入口为 [configs/M3ADet/M3ADet_convnext_t_soft_moe.py](../../configs/M3ADet/M3ADet_convnext_t_soft_moe.py)
- Micro Head 的配置入口为 [configs/M3ADet/M3ADet_convnext_t_micro_head.py](../../configs/M3ADet/M3ADet_convnext_t_micro_head.py)

#### 2.5.2 Soft MoE 消融组

该组重点分析：

- 路由平滑度
- 专家容量
- 插入深度
- 局部窗口大小
- 共享专家与辅助损失设计

#### 2.5.3 Micro Head 消融组

该组重点分析：

- 小目标辅助监督本身是否有效
- 热图引导增强是否有效
- 哪些浅层 level 更重要
- 小目标定义阈值如何影响收益
- 热图半径、增强强度和分支容量如何影响结果

## 3. 主结果实验

### 3.1 对比设置

主结果实验建议采用统一训练设置，仅改变以下关键变量：

- 是否启用 Soft MoE
- 是否启用 Micro Head

推荐主对比表如下。

| 方法 | Backbone 路由 | Micro Head | SAR mAP | RGB mAP | IFR mAP | Overall mAP |
| --- | --- | --- | --- | --- | --- | --- |
| SM3Det Baseline | Sparse / Default | No | [待填写] | [待填写] | [待填写] | [待填写] |
| Soft MoE Base | Soft | No | [待填写] | [待填写] | [待填写] | [待填写] |
| Micro Head Base | Sparse / Default | Yes | [待填写] | [待填写] | [待填写] | [待填写] |
| Soft MoE + Micro Head | Soft | Yes | [待填写] | [待填写] | [待填写] | [待填写] |

若当前阶段尚未构造 Soft MoE 与 Micro Head 的联合配置，可先保留前三行，第四行作为后续扩展实验。

### 3.2 主结果分析写法

在结果填入后，建议从以下三个角度组织分析：

#### 3.2.1 Soft MoE 对整体表征的影响

若 Soft MoE Base 相比原始基线取得提升，可以将其解释为：

- 窗口级连续专家混合改善了 backbone 内的局部语义建模。
- soft 路由相对于 sparse top-k 路由更平滑，降低了离散选择带来的训练波动。
- 在复杂背景和局部细粒度纹理场景中，soft 专家组合更适合当前三源遥感检测任务。

#### 3.2.2 Micro Head 对小目标与浅层特征的影响

若 Micro Head Base 相比原始基线提升明显，可进一步解释为：

- 显式的小目标中心、偏移与尺寸监督提升了浅层特征的目标敏感性。
- 热图引导的特征增强使原始检测头在浅层分辨率上获得更强的小目标显著性。
- 由于未重写原始头，性能增益可以更明确地归因于小目标辅助建模本身。

#### 3.2.3 分模态收益差异

若三种模态提升幅度不同，建议重点分析：

- SAR 分支是否因浅层步长偏大而更依赖动态 stride 对齐。
- RGB/IFR 分支是否因旋转目标更丰富而更受益于小目标中心热图增强。
- Soft MoE 是否在背景纹理更复杂的模态上体现出更明显收益。

## 4. Soft MoE 消融实验

### 4.1 消融目标

Soft MoE 消融的核心目的是回答以下问题：

1. Soft 路由本身是否优于 sparse 基线。
2. 温度系数如何影响软分配的平滑程度与性能。
3. 专家数变化带来的提升究竟来自容量还是路由机制。
4. Soft MoE 应该插入在浅层、中层还是深层 block。
5. 共享专家与辅助损失权重是否有助于稳定训练。

### 4.2 配置组说明

当前已整理的 Soft MoE 消融配置包括：

- sparse baseline
- soft base
- `temp05`
- `temp20`
- `e4`
- `e16`
- `blocks_last1`
- `blocks_last3`
- `blocks_even`
- `blocks_all`
- `ws14`
- `no_shared`
- `aux001`
- `hidden2`
- sparse 对照：`sparse_topk1`、`sparse_topk4`

对应的系统整理见 [ablation_config.md](../../ablation_config.md)。

### 4.3 路由类型对比

推荐首先比较原始 sparse 基线和 soft 基线：

| 方法 | moe_type | num_experts | top_k | temperature | Overall mAP |
| --- | --- | --- | --- | --- | --- |
| Sparse Baseline | sparse | 8 | 3 | - | [待填写] |
| Soft Base | window_slot_soft_moe | 8 | - | 1.0 | [待填写] |

这一对比是整个 Soft MoE 章节的核心，因为它直接回答“soft 路由本身是否值得引入”。

在写作上，建议强调：

- sparse 与 soft 的本质差别不只是超参数变化，而是路由范式差异。
- 因此，这一对比优先级高于所有内部超参数消融。

### 4.4 温度系数消融

温度系数控制 token-to-slot 相似度分布的平滑程度。推荐表格如下：

| 配置 | temperature | Overall mAP | 现象描述 |
| --- | --- | --- | --- |
| temp05 | 0.5 | [待填写] | [待填写] |
| soft_base | 1.0 | [待填写] | [待填写] |
| temp20 | 2.0 | [待填写] | [待填写] |

分析时可从如下角度展开：

- 温度过低时，soft 路由可能趋向于“近似硬选择”。
- 温度过高时，专家组合过于平滑，专家分工被弱化。
- 最优温度反映了“路由选择性”和“专家协同性”之间的平衡。

### 4.5 专家数量消融

专家数影响模型容量与专家细粒度划分能力。推荐表格如下：

| 配置 | num_experts | Overall mAP | 参数量 | 训练稳定性 |
| --- | --- | --- | --- | --- |
| e4 | 4 | [待填写] | [待填写] | [待填写] |
| soft_base | 8 | [待填写] | [待填写] | [待填写] |
| e16 | 16 | [待填写] | [待填写] | [待填写] |

若性能并未随专家数线性增长，可解释为：

- 专家数过少时模型容量不足。
- 专家数过多时，路由学习难度增大，辅助均衡约束变得更重要。
- 当前数据规模与任务复杂度对专家数量存在匹配上限。

### 4.6 插入位置消融

`MoE_Block_inds` 控制 Soft MoE 插入到哪些 ConvNeXt block 中。推荐表格如下：

| 配置 | 插入策略 | Overall mAP | 分析 |
| --- | --- | --- | --- |
| blocks_last1 | 仅最末层 block | [待填写] | [待填写] |
| blocks_last3 | 最后三个 block | [待填写] | [待填写] |
| blocks_even | 各 stage 偶数块 | [待填写] | [待填写] |
| blocks_all | 所有候选块 | [待填写] | [待填写] |

这一实验用于回答：

- Soft MoE 更适合增强高层语义表达，还是更广泛地作用于多层特征。
- 插入过多 block 是否会带来收益饱和甚至训练不稳定。

### 4.7 模块内部设计消融

可进一步比较：

- `ws14`：更大窗口
- `no_shared`：移除共享专家
- `aux001`：减小辅助损失权重
- `hidden2`：减小专家内部扩展比例

推荐表格如下：

| 配置 | 改动项 | Overall mAP | 结论 |
| --- | --- | --- | --- |
| ws14 | window_size=14 | [待填写] | [待填写] |
| no_shared | use_shared_expert=False | [待填写] | [待填写] |
| aux001 | aux_loss_weight=0.001 | [待填写] | [待填写] |
| hidden2 | hidden_ratio=2 | [待填写] | [待填写] |

### 4.8 Sparse 对照实验

为进一步说明 Soft MoE 的收益并非仅来自“改变专家选择强度”，建议保留 sparse 路径的 `top_k` 对照：

| 配置 | moe_type | top_k | Overall mAP |
| --- | --- | --- | --- |
| sparse_topk1 | sparse | 1 | [待填写] |
| sparse_baseline | sparse | 3 | [待填写] |
| sparse_topk4 | sparse | 4 | [待填写] |

若这一组变化较小，而 soft 基线提升更明显，则可以更有力地说明：

- Soft MoE 的主要收益来自连续路由与局部 soft mixing，而非简单改变 sparse 门控稀疏度。

## 5. Micro Head 消融实验

### 5.1 消融目标

Micro Head 消融的目标是识别性能收益的真正来源，具体包括：

1. 增益是否来自小目标辅助监督本身。
2. 增益是否来自热图引导的特征增强。
3. 哪些浅层 level 对小目标更关键。
4. 上下文融合是否有必要。
5. 小目标阈值、热图半径和增强强度如何影响结果。
6. 分支容量是否与收益存在单调关系。

### 5.2 配置组说明

当前已整理的 Micro Head 消融配置包括：

- `baseline`
- `micro_base`
- `stage1_only`
- `no_infer_enhance`
- `levels_p2_only`
- `context_on`
- `area256`
- `area2304`
- `radius0`
- `radius2`
- `enhance025`
- `enhance075`
- `feat64`
- `feat192`

并已配套批量脚本 [tools/misc/run_micro_head_ablation.sh](../../tools/misc/run_micro_head_ablation.sh)。

### 5.3 Baseline 对比

首先比较原始 SM3Det baseline 与 Micro Head baseline：

| 方法 | feat_levels | use_context | enhance | Overall mAP |
| --- | --- | --- | --- | --- |
| baseline | - | - | No | [待填写] |
| micro_base | (0, 1) | False | Yes | [待填写] |

若 micro_base 取得提升，可直接将其归因于：

- 显式小目标监督
- 热图引导浅层增强
- 与原始头协同而非替代的集成方式

### 5.4 阶段性设计消融

该组比较 `stage1_only` 与 `no_infer_enhance`：

| 配置 | 训练增强 | 推理增强 | Overall mAP | 说明 |
| --- | --- | --- | --- | --- |
| stage1_only | No | No | [待填写] | [待填写] |
| no_infer_enhance | Yes | No | [待填写] | [待填写] |
| micro_base | Yes | Yes | [待填写] | [待填写] |

这一实验回答两个问题：

- 仅靠小目标辅助监督是否已经足够。
- 推理阶段保留增强是否仍然有必要。

### 5.5 分层特征与上下文消融

该组比较 `levels_p2_only` 与 `context_on`：

| 配置 | feat_levels | use_context | Overall mAP | 说明 |
| --- | --- | --- | --- | --- |
| levels_p2_only | (0,) | False | [待填写] | [待填写] |
| micro_base | (0, 1) | False | [待填写] | [待填写] |
| context_on | (0, 1) | True | [待填写] | [待填写] |

若只用最浅层效果下降，说明：

- 小目标检测并非只依赖单一浅层 level。
- 多层浅层协同对稳定提升更重要。

若加入 context 后进一步提升，则说明：

- 仅有空间细节仍不足以完全区分小目标。
- 深层语义补充对背景抑制与类别判别仍有帮助。

### 5.6 小目标定义阈值消融

该组比较不同面积阈值：

| 配置 | small_area_thr | 对应尺度解释 | Overall mAP |
| --- | --- | --- | --- |
| area256 | 256 | 约 16x16 | [待填写] |
| micro_base | 1024 | 约 32x32 | [待填写] |
| area2304 | 2304 | 约 48x48 | [待填写] |

这一实验的核心意义在于：

- 判断当前数据集中“真正受益于 Micro Head 的目标尺度区间”。
- 避免把所有尺度目标混在一起讨论。

### 5.7 Heatmap 监督范围消融

推荐比较：

| 配置 | heatmap_radius | Overall mAP | 分析 |
| --- | --- | --- | --- |
| radius0 | 0 | [待填写] | [待填写] |
| micro_base | 1 | [待填写] | [待填写] |
| radius2 | 2 | [待填写] | [待填写] |

若 `radius0` 较差，则说明单点监督过于稀疏；若 `radius2` 反而下降，则说明过宽的邻域会引入模糊监督。由此可以得出适中的热图平滑半径更适合小目标定位。

### 5.8 特征增强强度消融

推荐比较：

| 配置 | enhance_weight | Overall mAP | 分析 |
| --- | --- | --- | --- |
| enhance025 | 0.25 | [待填写] | [待填写] |
| micro_base | 0.50 | [待填写] | [待填写] |
| enhance075 | 0.75 | [待填写] | [待填写] |

该实验用于说明：

- 增强强度过小可能不足以放大小目标响应。
- 增强强度过大则可能放大噪声区域或抑制原始特征平衡。

### 5.9 分支容量消融

推荐比较：

| 配置 | feat_channels | Overall mAP | 参数量 | 分析 |
| --- | --- | --- | --- | --- |
| feat64 | 64 | [待填写] | [待填写] | [待填写] |
| micro_base | 128 | [待填写] | [待填写] | [待填写] |
| feat192 | 192 | [待填写] | [待填写] | [待填写] |

若更大的 `feat_channels` 并未带来明显提升，可解释为：

- 当前任务收益更多来自监督形式与增强机制，而非单纯分支容量增加。

## 6. 复杂度与效率分析

除精度对比外，建议在正式论文中补充一张复杂度表，用于说明新增模块的工程代价。

| 方法 | Params | FLOPs | 训练显存 | 推理时延 | Overall mAP |
| --- | --- | --- | --- | --- | --- |
| Baseline | [待填写] | [待填写] | [待填写] | [待填写] | [待填写] |
| Soft MoE Base | [待填写] | [待填写] | [待填写] | [待填写] | [待填写] |
| Micro Head Base | [待填写] | [待填写] | [待填写] | [待填写] | [待填写] |

分析时应强调：

- Soft MoE 的额外代价主要来自专家分支与局部路由。
- Micro Head 的额外代价主要来自浅层轻量卷积分支和一次热图增强。
- 若性能增益显著而复杂度增长有限，则可以进一步凸显方法的实用性。

## 7. 定性分析与可视化建议

除定量实验外，建议在论文中加入定性可视化，以增强结论的说服力。

### 7.1 Soft MoE 可视化建议

建议展示：

- 不同窗口区域的 expert 响应差异
- 辅助损失训练曲线
- 不同温度下的专家使用分布

这类可视化可帮助说明：

- Soft MoE 不是简单增加参数，而是真正改变了局部专家参与模式。

### 7.2 Micro Head 可视化建议

建议展示：

- 中心热图响应样例
- 增强前后的浅层特征显著性对比
- 小目标区域的检测框改善案例
- 误检/漏检案例对比

若可视化显示：

- 小目标区域热图响应更集中
- 增强后浅层显著性更强
- 漏检小目标数量减少

则可以从现象层面进一步支撑 Micro Head 的有效性。

## 8. 结果讨论模板

当实验结果补齐后，建议使用如下写法组织结论。

### 8.1 总体结论模板

“从主结果表可以看出，所提出的 Window Slot Soft MoE 与 Hierarchical Staged Micro Head 均能够在原始 SM3Det 框架上带来稳定收益。其中，Soft MoE 主要改善 backbone 的局部专家化建模能力，而 Micro Head 主要增强浅层特征对小目标的显式感知能力。二者分别作用于特征生成与检测前增强两个阶段，体现出较好的互补性。”

### 8.2 Soft MoE 结论模板

“Soft MoE 相较于 sparse MoE 的优势主要体现在窗口级连续路由机制上。温度系数与专家数消融表明，Soft MoE 的收益并非简单来自更大容量，而是来自更平滑的专家组合与更稳定的局部语义混合。插入位置实验进一步说明，将 Soft MoE 置于合适的高层或中高层 block 能够更有效地增强主干表达能力。”

### 8.3 Micro Head 结论模板

“Micro Head 的消融结果表明，显式小目标监督与热图引导增强是性能提升的关键来源。仅使用单阶段监督或仅依赖单层浅层特征时，性能均不如完整设计；而引入适度的热图平滑和适中的增强强度后，模型对微小目标的感知能力得到明显增强。这说明该模块能够在不重写原始检测头的前提下，以较低代价提升小目标检测性能。”

## 9. 小结

本章给出了围绕 Soft MoE 与 Micro Head 的完整实验设计逻辑，包括：

- 统一的多源实验设置
- 主结果实验
- Soft MoE 系统消融
- Micro Head 系统消融
- 复杂度分析
- 定性可视化建议

该实验章节既可以作为论文初稿的直接文本基础，也可以作为后续整理实验表格、绘制结果图和撰写结果分析的统一框架。待具体训练结果补齐后，仅需将表格中的占位项替换为真实指标，即可形成较为完整的实验章节。