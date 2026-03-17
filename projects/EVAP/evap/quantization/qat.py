from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from detectron2.layers import Conv2d as D2Conv2d
except Exception:  # pragma: no cover - detectron2 may be unavailable in lightweight checks
    D2Conv2d = None


@dataclass(frozen=True)
class QuantConfig:
    enabled: bool = False
    w_bits: int = 4
    a_bits: int = 4
    attn_bits: int = 8
    skip_text_encoder: bool = True
    skip_first_last: bool = True
    w_granularity: str = "per_channel"
    a_granularity: str = "per_tensor"
    attn_granularity: str = "per_head"
    learnable_scale: bool = True
    debug_print: bool = False


def build_quant_config(cfg) -> QuantConfig:
    quant_cfg = getattr(getattr(cfg, "MODEL", None), "QUANT", None)
    if quant_cfg is None:
        return QuantConfig()
    return QuantConfig(
        enabled=bool(quant_cfg.ENABLED),
        w_bits=int(quant_cfg.W_BITS),
        a_bits=int(quant_cfg.A_BITS),
        attn_bits=int(quant_cfg.ATTN_BITS),
        skip_text_encoder=bool(quant_cfg.SKIP_TEXT_ENCODER),
        skip_first_last=bool(quant_cfg.SKIP_FIRST_LAST),
        w_granularity=str(quant_cfg.W_GRANULARITY),
        a_granularity=str(quant_cfg.A_GRANULARITY),
        attn_granularity=str(quant_cfg.ATTN_GRANULARITY),
        learnable_scale=bool(quant_cfg.LEARNABLE_SCALE),
        debug_print=bool(getattr(quant_cfg, "DEBUG_PRINT", False)),
    )


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def _inv_softplus(value: float) -> float:
    value_tensor = torch.tensor(float(value))
    return torch.log(torch.expm1(value_tensor)).item()


class FakeQuantizer(nn.Module):
    """
    Generic fake quantizer with STE. `per_head` is treated as a per-channel
    quantizer whose channel axis should be the explicit head dimension.
    """

    def __init__(
        self,
        bits: int,
        *,
        symmetric: bool,
        granularity: str,
        channel_axis: int = 0,
        enabled: bool = True,
        learnable_scale: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.granularity = granularity
        self.channel_axis = channel_axis
        self.enabled = enabled and bits is not None and bits < 32
        self.eps = eps
        if self.enabled and learnable_scale:
            self.scale_factor = nn.Parameter(torch.full((1,), _inv_softplus(1.0)))
        else:
            self.register_parameter("scale_factor", None)
        self.collect_stats = False
        self.use_calibrated_stats = False
        self.register_buffer("calib_scale", torch.ones(1))
        self.register_buffer("calib_zero_point", torch.zeros(1))
        self.register_buffer("calib_batches", torch.zeros((), dtype=torch.long))
        self.register_buffer("calib_ready", torch.zeros((), dtype=torch.bool))

    def _positive_scale_factor(self, dtype: torch.dtype, device: torch.device):
        if self.scale_factor is None:
            return None
        return (F.softplus(self.scale_factor) + 1e-6).to(dtype=dtype, device=device)

    def _reduce_dims(self, x: torch.Tensor) -> Tuple[int, ...]:
        if x.ndim == 0:
            return ()
        if self.granularity == "per_tensor":
            return tuple(range(x.ndim))
        if self.granularity in {"per_channel", "per_head"}:
            axis = self.channel_axis if self.channel_axis >= 0 else x.ndim + self.channel_axis
            axis = min(max(axis, 0), x.ndim - 1)
            return tuple(dim for dim in range(x.ndim) if dim != axis)
        raise ValueError(f"Unsupported quantization granularity: {self.granularity}")

    def _apply_learnable_scale(self, scale: torch.Tensor) -> torch.Tensor:
        scale_factor = self._positive_scale_factor(scale.dtype, scale.device)
        if scale_factor is None:
            return scale
        return scale * scale_factor

    def begin_calibration(self, reset: bool = True) -> None:
        self.collect_stats = True
        self.use_calibrated_stats = False
        if reset:
            self.calib_batches.zero_()
            self.calib_ready.zero_()

    def end_calibration(self, use_calibrated_stats: bool = True) -> None:
        self.collect_stats = False
        self.use_calibrated_stats = use_calibrated_stats and bool(self.calib_ready.item())

    def _update_calibration(self, scale: torch.Tensor, zero_point: Optional[torch.Tensor]) -> None:
        if not self.collect_stats:
            return

        scale_detached = scale.detach()
        zero_point_detached = None if zero_point is None else zero_point.detach()
        if not bool(self.calib_ready.item()) or self.calib_scale.shape != scale_detached.shape:
            self.calib_scale = scale_detached.clone()
            if zero_point_detached is not None:
                self.calib_zero_point = zero_point_detached.clone()
            self.calib_batches.fill_(1)
            self.calib_ready.fill_(True)
            return

        next_count = int(self.calib_batches.item()) + 1
        mix = 1.0 / next_count
        self.calib_scale = self.calib_scale.to(scale_detached) * (1.0 - mix) + scale_detached * mix
        if zero_point_detached is not None:
            self.calib_zero_point = self.calib_zero_point.to(zero_point_detached) * (1.0 - mix) + zero_point_detached * mix
        self.calib_batches.fill_(next_count)

    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None or not self.enabled or not torch.is_floating_point(x):
            return x

        orig_dtype = x.dtype
        work_x = x.float()
        reduce_dims = self._reduce_dims(work_x)

        if self.symmetric:
            qmax = (1 << (self.bits - 1)) - 1
            qmin = -(1 << (self.bits - 1))
            if reduce_dims:
                max_val = work_x.abs().amax(dim=reduce_dims, keepdim=True)
            else:
                max_val = work_x.abs()
            scale = torch.clamp(max_val / max(qmax, 1), min=self.eps)
            self._update_calibration(scale, None)
            if self.use_calibrated_stats and bool(self.calib_ready.item()):
                scale = self.calib_scale.to(dtype=work_x.dtype, device=work_x.device)
            scale = self._apply_learnable_scale(scale)
            q = _round_ste(work_x / scale).clamp(qmin, qmax)
            dq = q * scale
        else:
            qmin = 0
            qmax = (1 << self.bits) - 1
            if reduce_dims:
                x_min = work_x.amin(dim=reduce_dims, keepdim=True)
                x_max = work_x.amax(dim=reduce_dims, keepdim=True)
            else:
                x_min = work_x
                x_max = work_x
            scale = torch.clamp((x_max - x_min) / max(qmax - qmin, 1), min=self.eps)
            zero_point = _round_ste(qmin - x_min / scale).clamp(qmin, qmax)
            self._update_calibration(scale, zero_point)
            if self.use_calibrated_stats and bool(self.calib_ready.item()):
                scale = self.calib_scale.to(dtype=work_x.dtype, device=work_x.device)
                zero_point = self.calib_zero_point.to(dtype=work_x.dtype, device=work_x.device)
            scale = self._apply_learnable_scale(scale)
            q = _round_ste(work_x / scale + zero_point).clamp(qmin, qmax)
            dq = (q - zero_point) * scale

        dq = dq.to(orig_dtype)
        return x + (dq - x).detach()


def build_weight_quantizer(quant_cfg: QuantConfig, channel_axis: int = 0) -> FakeQuantizer:
    return FakeQuantizer(
        quant_cfg.w_bits,
        symmetric=True,
        granularity=quant_cfg.w_granularity,
        channel_axis=channel_axis,
        enabled=quant_cfg.enabled,
        learnable_scale=quant_cfg.learnable_scale,
    )


def build_activation_quantizer(quant_cfg: Optional[QuantConfig], channel_axis: int = 1) -> Optional[FakeQuantizer]:
    if quant_cfg is None:
        return None
    return FakeQuantizer(
        quant_cfg.a_bits,
        symmetric=False,
        granularity=quant_cfg.a_granularity,
        channel_axis=channel_axis,
        enabled=quant_cfg.enabled,
        learnable_scale=quant_cfg.learnable_scale,
    )


def build_attention_quantizer(quant_cfg: Optional[QuantConfig], channel_axis: int = 1) -> Optional[FakeQuantizer]:
    if quant_cfg is None:
        return None
    granularity = quant_cfg.attn_granularity
    return FakeQuantizer(
        quant_cfg.attn_bits,
        symmetric=False,
        granularity=granularity,
        channel_axis=channel_axis,
        enabled=quant_cfg.enabled,
        learnable_scale=quant_cfg.learnable_scale,
    )


class QuantLinear(nn.Module):
    def __init__(self, module: nn.Linear, quant_cfg: QuantConfig) -> None:
        super().__init__()
        self.module = module
        self.input_quant = build_activation_quantizer(quant_cfg, channel_axis=-1)
        self.weight_quant = build_weight_quantizer(quant_cfg, channel_axis=0)

    @property
    def weight(self):
        return self.module.weight

    @property
    def bias(self):
        return self.module.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_quant is not None:
            x = self.input_quant(x)
        return F.linear(x, self.weight_quant(self.module.weight), self.module.bias)


class QuantConv2d(nn.Module):
    def __init__(self, module: nn.Conv2d, quant_cfg: QuantConfig) -> None:
        super().__init__()
        self.module = module
        self.input_quant = build_activation_quantizer(quant_cfg, channel_axis=1)
        self.weight_quant = build_weight_quantizer(quant_cfg, channel_axis=0)

    @property
    def weight(self):
        return self.module.weight

    @property
    def bias(self):
        return self.module.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_quant is not None:
            x = self.input_quant(x)
        if D2Conv2d is not None and isinstance(self.module, D2Conv2d):
            if not torch.jit.is_scripting() and x.numel() == 0 and self.module.training:
                assert not isinstance(
                    getattr(self.module, "norm", None), nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs in Detectron2 Conv2d."
        y = self.module._conv_forward(x, self.weight_quant(self.module.weight), self.module.bias)
        norm = getattr(self.module, "norm", None)
        if norm is not None:
            y = norm(y)
        activation = getattr(self.module, "activation", None)
        if activation is not None:
            y = activation(y)
        return y


class QuantMultiheadAttention(nn.Module):
    """
    Minimal self-attention fake-quant wrapper for the decoder MHA path.
    """

    def __init__(self, module: nn.MultiheadAttention, quant_cfg: QuantConfig) -> None:
        super().__init__()
        if not module._qkv_same_embed_dim:
            raise NotImplementedError("Only same-dimension qkv attention is supported for QAT in this pass.")
        self.module = module
        self.input_quant = build_activation_quantizer(quant_cfg, channel_axis=-1)
        self.weight_quant = build_weight_quantizer(quant_cfg, channel_axis=0)
        self.attn_quant = build_attention_quantizer(quant_cfg, channel_axis=1)

    def _split_qkv(self):
        q_w, k_w, v_w = self.module.in_proj_weight.chunk(3, dim=0)
        if self.module.in_proj_bias is None:
            return (q_w, k_w, v_w), (None, None, None)
        q_b, k_b, v_b = self.module.in_proj_bias.chunk(3, dim=0)
        return (q_w, k_w, v_w), (q_b, k_b, v_b)

    def _apply_attn_mask(self, scores: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attn_mask is None:
            return scores
        if attn_mask.dtype == torch.bool:
            fill_value = float("-inf")
        else:
            fill_value = None

        if attn_mask.dim() == 2:
            if fill_value is not None:
                return scores.masked_fill(attn_mask[None, None], fill_value)
            return scores + attn_mask[None, None]

        if attn_mask.dim() != 3:
            raise ValueError(f"Unsupported attention mask shape: {attn_mask.shape}")

        batch_size = scores.shape[0]
        num_heads = scores.shape[1]
        if attn_mask.shape[0] == batch_size * num_heads:
            attn_mask = attn_mask.view(batch_size, num_heads, attn_mask.shape[-2], attn_mask.shape[-1])
        elif attn_mask.shape[0] == batch_size:
            attn_mask = attn_mask[:, None].expand(-1, num_heads, -1, -1)
        elif attn_mask.shape[0] == 1:
            attn_mask = attn_mask.view(1, 1, attn_mask.shape[-2], attn_mask.shape[-1]).expand(
                batch_size, num_heads, -1, -1
            )
        else:
            raise ValueError(f"Unsupported batched attention mask shape: {attn_mask.shape}")

        if fill_value is not None:
            return scores.masked_fill(attn_mask, fill_value)
        return scores + attn_mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if is_causal:
            raise NotImplementedError("Causal attention is not used in this TROY-VIS path.")

        if self.module.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        if self.input_quant is not None:
            query = self.input_quant(query)
            key = self.input_quant(key)
            value = self.input_quant(value)

        (q_w, k_w, v_w), (q_b, k_b, v_b) = self._split_qkv()
        q = F.linear(query, self.weight_quant(q_w), q_b)
        k = F.linear(key, self.weight_quant(k_w), k_b)
        v = F.linear(value, self.weight_quant(v_w), v_b)

        tgt_len, batch_size, embed_dim = q.shape
        src_len = k.shape[0]
        num_heads = self.module.num_heads
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q = q.transpose(0, 1).reshape(batch_size, tgt_len, num_heads, head_dim).transpose(1, 2)
        k = k.transpose(0, 1).reshape(batch_size, src_len, num_heads, head_dim).transpose(1, 2)
        v = v.transpose(0, 1).reshape(batch_size, src_len, num_heads, head_dim).transpose(1, 2)

        if self.attn_quant is not None:
            q = self.attn_quant(q)
            k = self.attn_quant(k)
            v = self.attn_quant(v)

        scores = torch.matmul(q * scale, k.transpose(-2, -1))
        if self.attn_quant is not None:
            scores = self.attn_quant(scores)
        scores = self._apply_attn_mask(scores, attn_mask)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None].bool(), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.attn_quant is not None:
            attn = self.attn_quant(attn)
        attn = F.dropout(attn, p=self.module.dropout, training=self.training)

        out = torch.matmul(attn, v)
        if self.attn_quant is not None:
            out = self.attn_quant(out)
        out = out.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim).transpose(0, 1)
        out = F.linear(out, self.weight_quant(self.module.out_proj.weight), self.module.out_proj.bias)

        if self.module.batch_first:
            out = out.transpose(0, 1)

        if not need_weights:
            return out, None

        weights = attn.mean(dim=1) if average_attn_weights else attn
        return out, weights


QUANT_WRAPPER_TYPES = (QuantLinear, QuantConv2d, QuantMultiheadAttention)


def has_quant_wrappers(module: Optional[nn.Module]) -> bool:
    if module is None:
        return False
    return any(isinstance(child, QUANT_WRAPPER_TYPES) for child in module.modules())


def build_predictor_skip_names(module: Optional[nn.Module]) -> Set[str]:
    if module is None:
        return set()

    skip_names: Set[str] = set()
    for attr_name in ("confidence_score", "mask_embed", "_bbox_embed"):
        head = getattr(module, attr_name, None)
        layers = getattr(head, "layers", None)
        if layers is not None and len(layers) > 0:
            skip_names.add(f"{attr_name}.layers.{len(layers) - 1}")

    match_head = getattr(module, "match_head", None)
    if match_head is not None and hasattr(match_head, "dot_product_projection_text"):
        skip_names.add("match_head.dot_product_projection_text")

    return skip_names


def build_backbone_skip_names(module: Optional[nn.Module]) -> Set[str]:
    if module is None:
        return set()

    quantizable_names = list(_collect_quantizable_names(module))
    if not quantizable_names:
        return set()

    preferred_prefixes = (
        "input_stem.op_list.0.conv",
        "input_stem.0.conv",
        "stem.conv1",
        "stem.0",
        "conv1",
        "patch_embed.proj",
        "patch_embed",
    )
    for prefix in preferred_prefixes:
        for name in quantizable_names:
            if name == prefix or name.startswith(prefix + "."):
                return {name}
    return {quantizable_names[0]}


def _is_quantizable_module(module: nn.Module) -> bool:
    if D2Conv2d is not None and isinstance(module, D2Conv2d):
        return True
    return isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention))


def _module_type_label(module: nn.Module) -> str:
    if D2Conv2d is not None and isinstance(module, D2Conv2d):
        return "detectron2.Conv2d"
    if isinstance(module, nn.Conv2d):
        return "nn.Conv2d"
    if isinstance(module, nn.Linear):
        return "nn.Linear"
    if isinstance(module, nn.MultiheadAttention):
        return "nn.MultiheadAttention"
    return type(module).__name__


def _collect_quantizable_names(module: nn.Module) -> Sequence[str]:
    names = []
    for name, child in module.named_modules():
        if not name:
            continue
        if _is_quantizable_module(child):
            names.append(name)
    return names


def _wrap_module(module: nn.Module, quant_cfg: QuantConfig) -> nn.Module:
    if isinstance(module, nn.MultiheadAttention):
        return QuantMultiheadAttention(module, quant_cfg)
    if isinstance(module, nn.Linear):
        return QuantLinear(module, quant_cfg)
    if isinstance(module, nn.Conv2d):
        return QuantConv2d(module, quant_cfg)
    return module


def prepare_module_for_qat(
    module: nn.Module,
    quant_cfg: QuantConfig,
    module_name: str,
    explicit_skip: Optional[Iterable[str]] = None,
    skip_prefixes: Optional[Iterable[str]] = None,
) -> Dict[str, Sequence[str]]:
    if module is None:
        return {"scope": module_name, "wrapped": [], "skipped": [], "skip_prefixes": [], "wrapped_type_counts": {}}
    if not quant_cfg.enabled:
        return {"scope": module_name, "wrapped": [], "skipped": [], "skip_prefixes": [], "wrapped_type_counts": {}}

    quantizable_names = list(_collect_quantizable_names(module))
    skip_names: Set[str] = set(explicit_skip or [])
    prefix_skips: Set[str] = set(skip_prefixes or [])
    if quant_cfg.skip_first_last and quantizable_names:
        if module_name == "backbone":
            skip_names.update(build_backbone_skip_names(module))
        elif module_name == "predictor" and not explicit_skip:
            skip_names.add(quantizable_names[-1])

    wrapped = []
    type_counts: Counter = Counter()
    seen: Dict[int, nn.Module] = {}

    def recurse(parent: nn.Module, prefix: str = "") -> None:
        if isinstance(parent, QUANT_WRAPPER_TYPES):
            return
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if any(full_name == item or full_name.startswith(f"{item}.") for item in prefix_skips):
                seen[id(child)] = child
                continue
            if id(child) in seen:
                setattr(parent, child_name, seen[id(child)])
                continue

            if _is_quantizable_module(child) and full_name not in skip_names:
                new_child = _wrap_module(child, quant_cfg)
                setattr(parent, child_name, new_child)
                seen[id(child)] = new_child
                child = new_child
                wrapped.append(full_name)
                type_counts[_module_type_label(new_child.module)] += 1
            else:
                seen[id(child)] = child

            recurse(child, full_name)

    recurse(module)

    summary = {
        "scope": module_name,
        "wrapped": wrapped,
        "skipped": sorted(skip_names),
        "skip_prefixes": sorted(prefix_skips),
        "wrapped_type_counts": dict(type_counts),
    }
    if quant_cfg.debug_print:
        print(
            f"[QAT] {module_name}: wrapped={len(summary['wrapped'])}, "
            f"skipped={len(summary['skipped'])}, type_counts={summary['wrapped_type_counts']}"
        )
        if summary["skipped"]:
            print(f"[QAT] {module_name} skipped modules: {summary['skipped']}")
        if summary["skip_prefixes"]:
            print(f"[QAT] {module_name} skipped subtrees: {summary['skip_prefixes']}")
    return summary


def iter_fake_quantizers(module: Optional[nn.Module]):
    if module is None:
        return
    for child in module.modules():
        if isinstance(child, FakeQuantizer):
            yield child


def summarize_quantization_summaries(summaries: Dict[str, Dict[str, Sequence[str]]]) -> Dict[str, Dict[str, int]]:
    type_counts: Counter = Counter()
    for summary in summaries.values():
        type_counts.update(summary.get("wrapped_type_counts", {}))
    return {"wrapped_type_counts": dict(type_counts)}


@torch.no_grad()
def initialize_fake_quantizers(
    model: nn.Module,
    representative_batches,
    forward_step,
    *,
    num_batches: int = 1,
    use_calibrated_stats: bool = True,
) -> Dict[str, int]:
    quantizers = list(iter_fake_quantizers(model))
    if not quantizers:
        return {"num_quantizers": 0, "num_batches": 0}
    if num_batches <= 0:
        return {"num_quantizers": len(quantizers), "num_batches": 0}

    was_training = model.training
    model.eval()
    try:
        for quantizer in quantizers:
            quantizer.begin_calibration(reset=True)

        num_seen = 0
        for batch in representative_batches:
            forward_step(model, batch)
            num_seen += 1
            if num_seen >= num_batches:
                break
    finally:
        for quantizer in quantizers:
            quantizer.end_calibration(use_calibrated_stats=use_calibrated_stats)
        if was_training:
            model.train()

    return {"num_quantizers": len(quantizers), "num_batches": num_seen}
