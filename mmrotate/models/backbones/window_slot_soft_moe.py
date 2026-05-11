import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_activation_layer


def _build_act_layer(act_layer):
    if isinstance(act_layer, dict):
        return build_activation_layer(act_layer)
    if isinstance(act_layer, nn.Module):
        return act_layer
    if isinstance(act_layer, type):
        return act_layer()
    return nn.GELU()


def window_partition(x, window_size):
    """Partition a [B, H, W, C] tensor into windows."""
    batch_size, height, width, channels = x.shape
    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, channels)


def window_reverse(windows, window_size, batch_size, height, width, channels):
    """Reverse partitioned windows back to [B, H, W, C]."""
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, height, width, channels)


class SoftMoEExpert(nn.Module):
    def __init__(self, dim, hidden_ratio=4, drop=0.0, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.pointwise_conv1 = nn.Linear(dim, hidden_dim)
        self.act = _build_act_layer(act_layer)
        self.drop1 = nn.Dropout(drop)
        self.pointwise_conv2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.pointwise_conv2(x)
        x = self.drop2(x)
        return x


class WindowSlotSoftMoE2D(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=8,
                 window_size=7,
                 hidden_ratio=4,
                 temperature=1.0,
                 residual_alpha_init=0.0,
                 use_shared_expert=True,
                 drop=0.0,
                 act_layer=nn.GELU,
                 aux_loss_weight=0.01):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.window_size = window_size
        self.temperature = temperature
        self.use_shared_expert = use_shared_expert
        self.aux_loss_weight = aux_loss_weight

        self.token_proj = nn.Linear(dim, dim)
        self.slot_embed = nn.Parameter(torch.randn(num_experts, dim))
        self.experts = nn.ModuleList([
            SoftMoEExpert(
                dim=dim,
                hidden_ratio=hidden_ratio,
                drop=drop,
                act_layer=act_layer) for _ in range(num_experts)
        ])
        self.shared_expert = SoftMoEExpert(
            dim=dim,
            hidden_ratio=hidden_ratio,
            drop=drop,
            act_layer=act_layer) if use_shared_expert else None
        self.alpha = nn.Parameter(torch.tensor(float(residual_alpha_init)))

        nn.init.trunc_normal_(self.slot_embed, std=0.02)
        nn.init.trunc_normal_(self.token_proj.weight, std=0.02)
        nn.init.constant_(self.token_proj.bias, 0.0)

    def _pad_input(self, x):
        batch_size, height, width, channels = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h == 0 and pad_w == 0:
            return x, height, width, height, width

        # NOTE: SOFT MOE WINDOW PAD - Convert to channel-first only for F.pad,
        # then restore [B, H, W, C] so the public module interface stays unchanged.
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.permute(0, 2, 3, 1).contiguous()
        padded_height = height + pad_h
        padded_width = width + pad_w
        return x, height, width, padded_height, padded_width

    def _compute_aux_loss(self, combine):
        usage = combine.mean(dim=(0, 1))
        target = torch.full_like(usage, 1.0 / max(self.num_experts, 1))
        balance_loss = F.mse_loss(usage, target)

        if self.num_experts <= 1:
            repulsion_loss = combine.new_zeros(())
        else:
            slot_embed = F.normalize(self.slot_embed, dim=-1)
            corr = torch.matmul(slot_embed, slot_embed.t())
            diag_mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=corr.device)
            repulsion_loss = corr.masked_select(diag_mask).pow(2).mean()

        return balance_loss + 0.1 * repulsion_loss

    def forward(self, x):
        batch_size, height, width, channels = x.shape
        padded_x, original_height, original_width, padded_height, padded_width = self._pad_input(x)

        # NOTE: SOFT MOE WINDOW SHAPES - Partition features as [BN, P, C],
        # where BN is the total number of windows and P=window_size*window_size.
        windows = window_partition(padded_x, self.window_size)
        token_features = F.normalize(self.token_proj(windows), dim=-1)
        slot_features = F.normalize(self.slot_embed, dim=-1)
        logits = torch.einsum('bpc,sc->bps', token_features, slot_features)
        logits = logits / max(self.temperature, 1e-6)

        dispatch = torch.softmax(logits, dim=1)
        combine = torch.softmax(logits, dim=2)

        slots = torch.einsum('bps,bpc->bsc', dispatch, windows)
        expert_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            expert_outputs.append(expert(slots[:, expert_idx, :]))
        expert_outputs = torch.stack(expert_outputs, dim=1)

        out_windows = torch.einsum('bps,bsc->bpc', combine, expert_outputs)
        if self.shared_expert is not None:
            out_windows = out_windows + self.shared_expert(windows)

        out = window_reverse(
            out_windows,
            self.window_size,
            batch_size,
            padded_height,
            padded_width,
            channels)
        out = out[:, :original_height, :original_width, :].contiguous()

        # NOTE: SOFT MOE RESIDUAL GATE - Start from tanh(alpha)=0 so the new
        # branch does not disturb the pretrained ConvNeXt backbone at init time.
        out = x + torch.tanh(self.alpha) * out
        aux_loss = self._compute_aux_loss(combine)
        return out, aux_loss