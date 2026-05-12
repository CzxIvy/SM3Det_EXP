# Soft MoE 与 Hierarchical Staged Micro Head 方法设计

## 1. 方法概述

本文档对当前工作区中新增的两个核心技术模块进行系统总结：

- Window Slot Soft MoE
- Hierarchical Staged Micro Head

二者分别作用于原始 SM3Det 框架中的不同层级。

- Soft MoE 作用于主干网络内部，用于增强特征生成阶段的局部专家建模能力。
- Micro Head 作用于颈部网络与原始检测头之间，用于增强浅层特征对小目标的感知能力。

这两个模块都遵循“最小侵入式改造”的设计原则，即：

- 不重写原始三源检测框架。
- 不改变原始 SAR、RGB、IFR 检测头的基本结构。
- 不改变最终推理输出的数据格式。
- 在尽量保持原始训练、验证、推理流程稳定的前提下，增加新的建模能力。

从系统层面看，原始模型由多输入主干、共享颈部与三条任务分支构成：

- 主干采用多输入版 ConvNeXt-MoE。
- 颈部采用 MultitaskFPN。
- SAR 分支使用单阶段 GFLHead。
- RGB 与 IFR 分支使用两阶段 OrientedRPNHead 与 OrientedStandardRoIHead。

在这一框架中：

- Soft MoE 改造的是“主干内的前馈专家路由机制”。
- Micro Head 改造的是“浅层特征进入原始检测头之前的辅助建模与增强机制”。

因此，两项改动分别对应“特征生成能力增强”和“小目标感知能力增强”两个层次。

## 2. 原始 SM3Det 架构与插入位置

### 2.1 原始三源检测框架

当前工作区的主配置文件为 [configs/SM3Det/SM3Det_convnext_t.py](../../configs/SM3Det/SM3Det_convnext_t.py)。原始模型采用三源输入结构：

- SAR 图像分支
- RGB 图像分支
- IFR 图像分支

检测器主体为 [mmrotate/models/detectors/trisource_H1stage_R2stage_detector.py](../../mmrotate/models/detectors/trisource_H1stage_R2stage_detector.py) 中的 `TriSourceDetector`。

其基本流程如下：

1. 多模态输入首先进入共享主干 `ConvNeXt_moe_MultiInput`。
2. 主干输出多尺度特征后进入 `MultitaskFPN`。
3. 对于 SAR 分支，FPN 使用 `start_level=1`，浅层特征相对 RGB/IFR 更“深”一层。
4. 对于 RGB 与 IFR 分支，FPN 使用标准的 `start_level=0`。
5. 三条分支分别进入各自的检测头完成训练与推理。

这一结构带来一个非常重要的实现约束：

- 三个模态共享总体框架，但浅层特征的实际步长并不完全一致。

这也是后续 Micro Head 需要动态推断步长的根本原因。

### 2.2 Soft MoE 的插入位置

Soft MoE 不作用于 detector 或 neck，而是插入到 ConvNeXt block 内部，替换部分 block 中原本的前馈网络 FFN。

其插入位置由以下配置控制：

- `model.backbone.moe_type`
- `model.backbone.soft_moe_cfg`
- `model.backbone.MoE_Block_inds`

因此，Soft MoE 本质上是“主干内部的局部结构替换”。

### 2.3 Micro Head 的插入位置

Micro Head 不改动 backbone 内部结构，也不重写原始检测头，而是在 neck 输出和原始检测头之间增加一个轻量级辅助分支。

这一分支在训练阶段承担两项功能：

- 为小目标提供显式的中心、偏移和尺寸监督。
- 根据预测得到的小目标中心热图增强浅层特征。

在推理阶段，该分支仅执行特征增强，不改变原始检测结果格式。

## 3. Window Slot Soft MoE

### 3.1 设计动机

原始 sparse MoE 在每个 token 上采用离散 top-k 专家选择，具有较强的稀疏性和表达能力，但同时也存在如下问题：

- 离散路由导致训练不够平滑。
- token 之间对专家资源的竞争更强。
- 当目标较小、背景复杂、局部纹理细碎时，硬路由可能不利于稳定建模。

为解决这些问题，本文引入 Window Slot Soft MoE，将原有的稀疏离散专家路由改造为局部窗口内的连续软分配机制。

该设计的关键思想是：

- 不在全局范围内进行专家竞争，而是在局部窗口范围内完成 token 与 expert slot 的交互。
- 不再强制每个 token 只路由到少数专家，而是允许其以连续权重与多个专家发生联系。
- 通过辅助约束提高专家利用率均衡性和 slot 区分度。

因此，Soft MoE 的核心特征可以概括为：

- 局部化
- 连续化
- 可微化
- 对预训练主干友好

### 3.2 核心模块结构

Soft MoE 的实现位于 [mmrotate/models/backbones/window_slot_soft_moe.py](../../mmrotate/models/backbones/window_slot_soft_moe.py)。

核心模块为 `WindowSlotSoftMoE2D`，其输入输出接口均采用 channels-last 格式，即特征张量形状为：

$$
[B, H, W, C]
$$

模块内部包含如下组件：

- `token_proj`：将 token 投影到路由空间。
- `slot_embed`：可学习的 expert slot 原型。
- `experts`：多个独立的前馈专家。
- `shared_expert`：可选共享专家分支。
- `alpha`：控制 soft 分支残差注入强度的门控参数。

此外，模块内部还包含窗口划分、补边、辅助损失计算和窗口逆恢复等子过程。

### 3.3 窗口级软路由机制

设一个窗口内包含 $P = w \times w$ 个 token，专家数量为 $E$。记窗口内第 $p$ 个 token 为 $x_p$，第 $e$ 个 slot 向量为 $s_e$，则首先计算 token 与 slot 的相似度：

$$
L_{p,e} = \frac{\langle \hat{W x_p}, \hat{s_e} \rangle}{T}
$$

其中：

- $W$ 表示 `token_proj` 对应的线性变换。
- $\hat{\cdot}$ 表示归一化。
- $T$ 为温度系数 `temperature`。

随后对该相似度矩阵执行两种 softmax，分别得到 dispatch 权重与 combine 权重：

$$
D_{p,e} = softmax_{p}(L_{p,e})
$$

$$
C_{p,e} = softmax_{e}(L_{p,e})
$$

二者的作用不同：

- $D_{p,e}$ 用于将窗口内 token 聚合到专家槽位。
- $C_{p,e}$ 用于将专家输出重新组合回 token 空间。

基于 dispatch 权重，专家输入可写为：

$$
z_e = \sum_p D_{p,e} x_p
$$

每个专家独立处理其输入：

$$
u_e = f_e(z_e)
$$

再通过 combine 权重把专家输出映射回每个 token：

$$
y_p = \sum_e C_{p,e} u_e
$$

若启用共享专家分支，则额外有一条共享前馈路径叠加到输出中。

### 3.4 局部窗口设计的意义

与全局专家混合相比，窗口级设计具有以下优势：

- 计算复杂度更可控。
- 避免整张特征图上的 token 发生过强竞争。
- 更符合遥感图像中局部空间结构对目标识别的重要性。
- 对小目标和局部纹理模式更友好。

因此，Soft MoE 不是简单地把 sparse MoE 改成 soft，而是进一步加入了“局部空间约束”。

### 3.5 残差门控与稳定初始化

Soft MoE 输出最终以残差形式注入主干特征。设 Soft MoE 重建得到的窗口输出为 $\tilde{y}$，原始输入为 $x$，则最终输出为：

$$
y = x + \tanh(\alpha) \cdot \tilde{y}
$$

其中 $\alpha$ 为可学习参数，初始化为 0。

这一设计具有重要意义：

- 初始化时 $\tanh(\alpha) = 0$，因此新增分支初始几乎不扰动原始预训练 backbone。
- 训练过程中模型可以逐步学习 Soft MoE 分支对主干的贡献强度。
- 与直接把新分支硬插入网络相比，这种渐进式残差注入更稳定。

这是一项典型的“预训练友好型”工程设计。

### 3.6 辅助损失设计

Soft MoE 的辅助损失由两部分组成：

- 专家使用均衡损失
- slot 去相关损失

整体形式可写为：

$$
L_{aux} = L_{balance} + 0.1 \cdot L_{repulsion}
$$

其中：

- $L_{balance}$ 通过约束平均 combine 权重接近均匀分布，鼓励所有 expert 都被合理使用。
- $L_{repulsion}$ 通过约束不同 slot embedding 之间不要过于相似，抑制专家塌缩。

该辅助损失并不直接承担检测任务，而是用于规范 soft 路由结构本身的训练行为。

### 3.7 与原始 ConvNeXt block 的集成方式

Soft MoE 的接入位置在 [mmrotate/models/backbones/convnext_moe.py](../../mmrotate/models/backbones/convnext_moe.py) 中的 `ConvNeXtBlock`。

原始 ConvNeXt block 主要包含：

- depthwise convolution
- normalization
- pointwise FFN
- residual shortcut

本次改动中：

- depthwise convolution 与残差主干保持不变。
- 仅在指定 block 的 FFN 位置替换为 Soft MoE。
- 未被指定为 MoE block 的层仍保留原始 FFN。

这种局部替换策略有助于：

- 控制新增参数量和计算量。
- 降低对原始 backbone 结构的扰动。
- 通过 `MoE_Block_inds` 灵活控制 Soft MoE 的插入深度与密度。

### 3.8 主干配置与构造逻辑

在 `ConvNeXt_moe` 中，`moe_type` 用于区分两类专家机制：

- `sparse`
- `window_slot_soft_moe`

主干初始化时会根据配置做如下操作：

1. 构造 sparse 路径的统一配置。
2. 构造 soft 路径的统一配置。
3. 根据 `MoE_Block_inds` 决定每个 stage 中哪些 block 被标记为 MoE block。
4. 对于每个被标记的 block，根据 `moe_type` 决定实例化 sparse MoE 或 Soft MoE。

这一统一的构造逻辑具有两点优势：

- 保持了 sparse 与 soft 路径在主干构造上的统一接口。
- 使得消融实验可以在尽量少改动配置的前提下完成。

### 3.9 多输入主干中的 Soft MoE 路径

当前工程使用的是 `ConvNeXt_moe_MultiInput`，其实现位于 [mmrotate/models/backbones/convnext_moe.py](../../mmrotate/models/backbones/convnext_moe.py)。

对于三源输入：

- 多模态图像首先被拼接成一个 batch。
- 通过共享的 stem 和共享主干进行特征提取。
- 在所有 MoE block 中收集辅助损失。
- 最终输出多尺度特征和一个聚合后的 `aux_loss`。

因此，从 detector 视角看，主干返回的是：

- `tuple(outs)`，表示各尺度特征
- `aux_loss`，表示所有 Soft MoE block 的平均辅助损失

### 3.10 训练损失路由与日志标识

在 detector 的 `forward_train` 中，主干返回的专家辅助损失会被进一步写入总损失字典。

为避免与原始任务损失混淆，本实现对两类专家路径采用不同的命名方式：

- sparse 路径使用 `loss_moe_aux`
- Soft MoE 路径使用 `loss_soft_moe_aux`

这一设计具有三方面意义：

- 可以在训练日志中直接区分 sparse 与 soft 路径。
- 避免与主任务损失重名或混淆。
- 防止辅助损失被错误纳入 DSO 的 `reweight_losses` 列表。

因此，Soft MoE 在系统中不仅是结构上的替换，也是训练信号路由上的显式扩展。

### 3.11 预训练权重复用策略

Soft MoE 的另一个关键工程点在于预训练权重兼容。

在主干 `init_weights` 中，原始 ConvNeXt 预训练权重会被重新映射：

- 原始 FFN 的 pointwise 层权重会复制到所有专家中。
- 若启用共享专家，则共享专家也复用相应的 FFN 权重。
- 非 MoE block 仍保留原始 FFN 的映射方式。

这样做的意义在于：

- 避免 Soft MoE 专家从完全随机初始化开始训练。
- 最大程度继承原始 ConvNeXt 的预训练语义先验。
- 降低新结构插入后训练不稳定的风险。

因此，本实现并不是“重新训练一个全新 backbone”，而是在预训练 ConvNeXt 的基础上做结构兼容式扩展。

### 3.12 Soft MoE 的主要可消融因素

在当前实现中，Soft MoE 的关键可控因素包括：

- `temperature`
- `num_experts`
- `window_size`
- `hidden_ratio`
- `use_shared_expert`
- `aux_loss_weight`
- `MoE_Block_inds`

需要特别指出的是：

- `top_k` 是 sparse MoE 的关键控制参数。
- 在当前 Window Slot Soft MoE 实现中，`top_k` 不再作为 active knob 生效。

这一差异说明 Soft MoE 与 sparse MoE 的核心区别不只是“专家输出更平滑”，而是整个路由范式发生了变化。

## 4. Hierarchical Staged Micro Head

### 4.1 设计动机

在遥感与多模态旋转目标检测场景中，小目标通常具有以下特点：

- 尺度小
- 数量多
- 容易淹没于复杂背景
- 对浅层高分辨率特征更敏感

然而，原始检测头通常面向全尺度目标统一建模，可能难以显式强调小目标区域。若直接重写所有检测头，不仅工程代价高，也会增加实验不确定性。

因此，本工作引入 `HierarchicalStagedMicroHead`，以轻量级辅助分支的形式增强原始模型对小目标的建模能力。

该设计遵循两条原则：

- 不替代原始检测头，只作为前置辅助模块。
- 不改变最终输出格式，只增强浅层特征表示。

### 4.2 “Hierarchical” 与 “Staged” 的含义

本模块名称中的两个关键词分别对应两个设计维度。

#### 4.2.1 Hierarchical

Hierarchical 表示该模块并不在所有 FPN 层上等价工作，而是只在指定的浅层特征层上工作。

默认情况下，`feat_levels=(0, 1)`，表示：

- 选取最浅的两个 FPN 层进行小目标建模。
- 这些层保留更高的空间分辨率，更适合感知微小目标。

此外，还可选用 `context_level` 引入更深层语义特征，为浅层提供额外上下文。

#### 4.2.2 Staged

Staged 并不表示检测框级联，而是表示模块内部具有两个连续阶段：

1. 小目标辅助预测阶段
2. 热图引导的浅层特征增强阶段

第一个阶段负责输出中心、偏移和尺寸等显式监督信号；第二个阶段负责把阶段一的中心热图转化为空间注意力，对原始浅层特征进行增强。

### 4.3 模块结构

`HierarchicalStagedMicroHead` 的实现位于 [mmrotate/models/dense_heads/hierarchical_staged_micro_head.py](../../mmrotate/models/dense_heads/hierarchical_staged_micro_head.py)。

对于每一个被选中的特征层，模块内部建立以下子结构：

- 一个两层卷积的特征变换块
- 一个中心热图头 `center_head`
- 一个偏移预测头 `offset_head`
- 一个尺寸预测头 `size_head`
- 一个可选的上下文投影层 `context_proj`

若设输入特征通道数为 `in_channels`，中间特征维度为 `feat_channels`，则每个 level 的流程可概括为：

1. 输入浅层特征。
2. 通过两层卷积提取更适合小目标建模的局部表征。
3. 若开启 context 融合，则引入深层语义特征并上采样后相加。
4. 输出中心热图、偏移和尺寸三个预测结果。

这一结构具有如下优点：

- 参数量小。
- 逻辑清晰。
- 容易单独调试和单独验证。
- 不影响原始检测头的结构定义。

### 4.4 第一阶段：显式小目标监督

模块第一阶段会在每个选定层上输出三类预测：

- 中心热图 `center_logits`
- 偏移 `offset_preds`
- 尺寸 `size_preds`

其形式可记为：

$$
P_l = \{H_l, O_l, S_l\}
$$

其中：

- $H_l \in \mathbb{R}^{1 \times H \times W}$ 表示中心热图 logits。
- $O_l \in \mathbb{R}^{2 \times H \times W}$ 表示中心偏移。
- $S_l \in \mathbb{R}^{2 \times H \times W}$ 表示宽高的对数尺度。

这种建模方式的意义在于：

- 中心热图显式刻画小目标可能出现的位置。
- 偏移提供从离散网格到连续位置的补偿。
- 尺寸监督使浅层特征具备更明确的几何感知能力。

因此，第一阶段的目标不是直接替代检测框回归，而是为后续原始检测头提供更具小目标先验的浅层表征。

### 4.5 第二阶段：热图引导的特征增强

在第二阶段，模块利用第一阶段预测得到的中心热图对浅层特征进行增强。

记原始浅层特征为 $F_l$，中心热图 logits 为 $H_l$，则增强后的特征为：

$$
F'_l = F_l \odot (1 + \beta \cdot \sigma(H_l))
$$

其中：

- $\sigma(\cdot)$ 为 sigmoid。
- $\beta$ 为增强强度参数 `enhance_weight`。
- $\odot$ 表示逐元素乘法。

这一设计意味着：

- 热图响应高的位置会被放大。
- 背景区域保持接近原始特征。
- 增强是连续、可控且空间敏感的。

与直接引入不可解释的注意力模块相比，这种增强方式具有更明确的物理语义，即“利用小目标中心先验来放大小目标相关特征”。

### 4.6 上下文特征融合

若开启 `use_context=True`，则模块会从更深层的 `context_level` 提取特征，并通过以下方式融入当前浅层 level：

1. 使用 $1 \times 1$ 卷积将深层特征投影到 `feat_channels`。
2. 通过双线性插值将其上采样到浅层分辨率。
3. 与当前浅层隐特征逐元素相加。

这一机制的意义在于：

- 浅层特征提供空间细节。
- 深层特征提供语义上下文。
- 二者结合可以在不显著增加复杂度的前提下提高小目标区分能力。

因此，该模块并不仅仅依赖浅层纹理，也允许通过可控方式引入高层语义补充。

### 4.7 多种标注框格式的统一解析

小目标辅助监督依赖于从 GT 框中提取目标中心、尺寸与面积。为兼容当前工程中的不同标注形式，`HierarchicalStagedMicroHead` 统一支持三类框格式：

- 水平框 HBB：4 维
- 旋转框 OBB：5 维
- 四边形 polygon：8 维

对应处理方式如下：

#### 4.7.1 HBB

对于 HBB，使用左上角与右下角坐标恢复：

- 中心点
- 宽高
- 面积

#### 4.7.2 OBB

对于 OBB，直接使用前两维作为中心点，第三、四维作为宽高，面积由宽高相乘得到。

#### 4.7.3 Polygon

对于 polygon，使用四个顶点：

- 通过顶点均值得到中心点。
- 通过顶点的最大最小坐标差得到外接宽高估计。
- 通过鞋带公式计算多边形面积。

这一统一解析逻辑使得 Micro Head 可以直接作用于当前多种检测分支与标注格式，而无需额外改造数据管线。

### 4.8 小目标样本选择策略

Micro Head 的监督并不对所有目标生效，而是只针对“小目标”构建目标图。

当前实现支持两种筛选方式：

- 基于面积阈值 `small_area_thr`
- 基于最大边阈值 `small_size_thr`

具体规则为：

- 若 `small_size_thr > 0`，则以目标最大边是否不超过阈值为准。
- 否则，默认以面积是否不超过 `small_area_thr` 为准。

这一设计的意义在于：

- 小目标定义不被写死，可以根据数据集目标尺度分布灵活调整。
- 通过阈值消融可以分析收益究竟来自何种尺度范围的目标。

### 4.9 热图监督生成方式

对于每个小目标，模块会根据目标中心在特征图上的映射位置构建监督。具体过程为：

1. 将图像坐标下的中心点除以特征层步长，映射到当前 level 的网格坐标。
2. 将中心点的整数部分作为正样本位置。
3. 小数部分作为 offset 监督。
4. 将宽高除以步长并取对数，作为 size 监督。
5. 在中心位置及其邻域绘制高斯热图或单点热图。

当 `heatmap_radius` 大于 0 时，会在中心点附近绘制局部高斯响应；当其等于 0 时，只保留单点正样本。

因此，`heatmap_radius` 实际上控制的是：

- 中心监督的空间平滑范围
- 正样本邻域的扩散程度

### 4.10 步长动态推断机制

这是当前 Micro Head 集成中最关键的工程细节之一。

理论上，Micro Head 配置中保留了 `feat_strides` 字段，但在真实 detector 运行时，并不会简单地完全信任这个静态值，而是由 detector 根据当前模态实际特征图尺寸动态推断各层 stride。

这样设计的原因在于：

- RGB 与 IFR 分支的 neck 输出默认从 `start_level=0` 开始。
- SAR 分支的 neck 输出从 `start_level=1` 开始。
- 因此，三种模态的浅层特征与输入图像之间的步长关系并不一致。

若把 stride 固定写死，会导致：

- 小目标中心映射偏移
- 热图监督位置错误
- offset 与 size 目标对不齐

因此，detector 在训练时会：

1. 获取当前模态输入图像的 `batch_input_shape`。
2. 根据每层特征图分辨率反推出步长。
3. 再把与 `feat_levels` 对应的步长传给 Micro Head 的损失函数。

这一点保证了同一套 Micro Head 可以兼容 SAR、RGB、IFR 三个模态分支。

### 4.11 损失函数设计

Micro Head 的总损失由三部分组成：

$$
L_{micro} = \lambda_c L_{center} + \lambda_o L_{offset} + \lambda_s L_{size}
$$

其中：

- $L_{center}$ 为中心热图损失。
- $L_{offset}$ 为偏移回归损失。
- $L_{size}$ 为尺寸回归损失。

当前实现中：

- `loss_type='focal'` 时，中心热图采用 sigmoid focal loss。
- `loss_type='bce'` 时，中心热图采用 BCE。
- 偏移和尺寸均采用仅在正样本位置生效的 L1 损失。

默认权重通常设置为：

- `center_loss_weight = 0.2`
- `offset_loss_weight = 0.05`
- `size_loss_weight = 0.05`

这一设计说明：

- 中心监督是主导项。
- 偏移与尺寸提供辅助几何约束。

### 4.12 与 detector 的训练流程集成

Micro Head 的 detector 端集成位于 `TriSourceDetector` 中。

训练阶段的核心流程如下：

1. detector 完成 backbone 与 neck 特征提取。
2. 对每个模态分支，若启用了 Micro Head，则先进入 `_forward_micro_train`。
3. `_forward_micro_train` 内部先执行 Micro Head 前向预测。
4. 然后基于中心热图增强浅层特征。
5. 再计算当前模态对应的 Micro Head 损失。
6. 最后把增强后的特征送入原始检测头。

因此，对于每个模态，训练时都会形成如下逻辑：

- 原始检测头负责主检测任务。
- Micro Head 提供附加的小目标监督与浅层特征增强。

需要强调的是：

- 原始检测头结构保持不变。
- 原始损失项仍然完整保留。

### 4.13 多模态损失命名与任务解耦

为了在三源训练中保持可分析性，Micro Head 的损失在 detector 中会带上模态前缀。

例如：

- `sar_loss_micro_center`
- `sar_loss_micro_offset`
- `sar_loss_micro_size`
- `rgb_loss_micro_center`
- `rgb_loss_micro_offset`
- `rgb_loss_micro_size`
- `ifr_loss_micro_center`
- `ifr_loss_micro_offset`
- `ifr_loss_micro_size`

这种命名方式的好处在于：

- 可以分别观察三种模态对 Micro Head 的响应。
- 避免与原始检测头损失重名。
- 方便在实验日志中直接检索和统计。

同时，这些 Micro Head 辅助损失不会被加入 DSO 的 `reweight_losses` 列表，从而保持原始主任务重加权逻辑不变。

### 4.14 推理阶段的集成方式

在推理阶段，Micro Head 的处理逻辑进一步简化。

若 `use_inference_enhance=True`，则：

1. 先基于当前特征运行 Micro Head 前向。
2. 利用中心热图增强浅层特征。
3. 将增强后的特征送入原始检测头。

但需要特别指出：

- 推理阶段并不会额外输出 Micro Head 的独立结果。
- 模型最终输出仍然是原始检测头的标准结果格式。

因此，Micro Head 在推理阶段表现为一种“显式监督驱动的前置特征增强模块”，而不是新的检测分支。

### 4.15 Micro Head 的主要可消融因素

当前实现中，Micro Head 的主要可调参数包括：

- `feat_channels`
- `feat_levels`
- `use_context`
- `context_level`
- `small_area_thr`
- `small_size_thr`
- `heatmap_radius`
- `center_loss_weight`
- `offset_loss_weight`
- `size_loss_weight`
- `enhance_weight`
- `detach_heatmap`
- `use_feat_enhance`
- `use_inference_enhance`
- `loss_type`

这些参数分别对应：

- 分支容量
- 分层特征选择
- 语义上下文融合
- 小目标定义方式
- 热图平滑范围
- 特征增强强度
- 训练与推理阶段的一致性

因此，Micro Head 的实验空间具有较好的结构化特征，适合开展系统消融研究。

## 5. 两个模块在系统中的互补关系

从系统角度看，Soft MoE 与 Micro Head 分别作用于不同层级，二者并不是相互替代的，而是形成互补。

### 5.1 Soft MoE 的作用层级

Soft MoE 工作在 backbone block 内部，其主要目标是：

- 改善主干特征生成阶段的专家化建模能力。
- 提升局部语义混合的连续性与稳定性。
- 在不重写 backbone 主体的前提下增强特征表达能力。

### 5.2 Micro Head 的作用层级

Micro Head 工作在 neck 与原始检测头之间，其主要目标是：

- 增强浅层特征对小目标位置与尺度的敏感性。
- 为原始检测头提供显式的小目标先验。
- 在不重写检测头的前提下提升小目标感知能力。

### 5.3 二者的互补性

因此，二者可分别概括为：

- Soft MoE 解决“特征如何被更好地生成”。
- Micro Head 解决“浅层特征如何更好地关注小目标”。

前者偏向表示学习与专家路由，后者偏向目标感知与检测前增强。它们共同构成了从 backbone 到 head 的分层增强设计。

## 6. 工程实现特点与方法优势

### 6.1 最小侵入式实现

本次设计在工程实现上的一个重要特点，是尽量避免对原始系统做大规模改写。

具体体现在：

- Soft MoE 只替换指定 ConvNeXt block 的 FFN。
- Micro Head 只在 neck 与原始头之间增加辅助分支。
- 原始 SAR、RGB、IFR 检测头定义不变。
- 最终推理输出格式不变。
- 多任务重加权逻辑不被破坏。

这使得新增模块的效果更容易被独立分析，实验结论也更清晰。

### 6.2 预训练兼容性强

Soft MoE 通过以下方式保持与预训练 backbone 的兼容：

- 复用原始 FFN 权重初始化专家。
- 通过零初始化残差门控实现渐进式功能注入。

Micro Head 则通过以下方式降低对原系统的冲击：

- 仅对浅层特征做增强。
- 默认不改变原始检测头接口。
- 推理阶段只做特征增强，不新增预测结果格式。

### 6.3 对多模态异构结构的适配

当前三源结构中，SAR 与 RGB/IFR 的浅层特征步长存在差异。Micro Head 通过 detector 侧动态步长推断解决了这一问题；Soft MoE 则统一在共享 backbone 内部工作，不依赖具体模态头部结构。

因此，这两项设计并不是简单地为单一路径定制，而是考虑了当前三源系统的整体一致性。

## 7. 实现文件对应关系

为了便于后续写作、汇报与代码追踪，下面给出关键实现文件与其职责。

### 7.1 Soft MoE 相关

- [configs/M3ADet/M3ADet_convnext_t_soft_moe.py](../../configs/M3ADet/M3ADet_convnext_t_soft_moe.py)
  - Soft MoE 的主配置入口。
- [mmrotate/models/backbones/window_slot_soft_moe.py](../../mmrotate/models/backbones/window_slot_soft_moe.py)
  - Window Slot Soft MoE 的核心实现。
- [mmrotate/models/backbones/convnext_moe.py](../../mmrotate/models/backbones/convnext_moe.py)
  - ConvNeXt block 中的 MoE 接入、主干构造、辅助损失聚合和预训练权重映射。

### 7.2 Micro Head 相关

- [configs/M3ADet/M3ADet_convnext_t_micro_head.py](../../configs/M3ADet/M3ADet_convnext_t_micro_head.py)
  - Micro Head 的配置入口。
- [mmrotate/models/dense_heads/hierarchical_staged_micro_head.py](../../mmrotate/models/dense_heads/hierarchical_staged_micro_head.py)
  - Micro Head 的核心实现。
- [mmrotate/models/dense_heads/__init__.py](../../mmrotate/models/dense_heads/__init__.py)
  - 注册与导出 Micro Head。
- [mmrotate/models/detectors/trisource_H1stage_R2stage_detector.py](../../mmrotate/models/detectors/trisource_H1stage_R2stage_detector.py)
  - detector 端的训练与推理接入。
- [tools/misc/test_micro_head_smoke.py](../../tools/misc/test_micro_head_smoke.py)
  - Micro Head 独立 smoke test。

### 7.3 消融与实验文档

- [ablation_config.md](../../ablation_config.md)
  - 已整理好的 Soft MoE 与 Micro Head 消融说明、推荐顺序与批量运行方式。

## 8. 小结

综上，本文档总结的两项技术分别针对当前三源遥感检测系统中的两个不同瓶颈。

- Window Slot Soft MoE 通过局部窗口化的连续专家混合机制，增强 backbone 内部的特征生成与路由能力。
- Hierarchical Staged Micro Head 通过显式的小目标监督和热图引导的浅层特征增强，提高原始检测头对微小目标的感知能力。

二者的共同特点在于：

- 结构上最小侵入
- 与原始框架兼容
- 便于独立消融分析
- 适合在工程系统中渐进式集成

从方法论角度看，这一组合体现了“主干表示增强 + 检测前小目标增强”的分层设计思想；从工程角度看，则体现了“尽量不重写原系统、优先扩展关键路径”的实现策略。这两点都为后续撰写论文方法章节、制作技术报告或汇报 PPT 提供了较为完整的叙事基础。