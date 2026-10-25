"""Causal LM scaffold at the frozen Core A1 shape.

    d_model=256, n_layer=6, n_head=4, d_ff=1024, vocab=8000.

Default variant is SDA-Recency. The LM head is a separate projection (not
tied) so this shape stays near 1e7 parameters. No learned position table:
order is the causal scan. ``max_seq_len`` is a runtime cap, not a width.

Run this file to print the parameter count. The count is not a quality score,
and a smoke loss is not evidence that a trained layer works.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

try:
    from .baselines import NoDecayDelta, SoftmaxAttention
    from .sda_block import SDABlock, VARIANTS
except ImportError:
    from baselines import NoDecayDelta, SoftmaxAttention
    from sda_block import SDABlock, VARIANTS


MIXERS = ("sda", "softmax", "delta")


@dataclass(frozen=True)
class SDALMConfig:
    """Frozen stack shape, plus which of the two memory layouts is active.

    ``variant="recency"`` is the default. ``variant="expanded"`` turns the
    archive off and widens the delta state. ``expand_factor`` is only legal
    for ``expanded``. Archive ρ is not a field: a later ablation may vary the
    ring, and this config does not invent small/large caps.

    ``mixer`` selects the block. ``sda`` is the default. ``softmax`` and
    ``delta`` are the A3 baselines at this same shape. They are not Expanded.
    """

    vocab_size: int = 8000
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 4
    d_ff: int = 1024
    archive_cap: int = 32
    mixer: str = "sda"
    variant: str = "recency"
    expand_factor: int | None = None
    max_seq_len: int = 128
    beta_init: float = 0.05
    tau_init: float = 0.75
    gate_temp_init: float = 0.25
    tie_embeddings: bool = False
    hard_gate_eval: bool = False

    def __post_init__(self) -> None:
        if self.mixer not in MIXERS:
            raise ValueError(f"mixer must be one of {MIXERS}, got {self.mixer!r}")
        if self.mixer != "sda":
            if self.variant != "recency":
                raise ValueError("softmax and no-decay delta are not the expanded variant")
            if self.expand_factor is not None:
                raise ValueError("expand_factor is only valid when variant='expanded'")
            if self.hard_gate_eval:
                raise ValueError("hard_gate_eval applies only to the SDA mixer")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.d_model < 1 or self.n_layer < 1 or self.n_head < 1 or self.d_ff < 1:
            raise ValueError("d_model, n_layer, n_head and d_ff must be positive")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.archive_cap < 1:
            raise ValueError("archive_cap must be >= 1")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if self.variant == "recency" and self.expand_factor is not None:
            raise ValueError("expand_factor is only valid when variant='expanded'")
        if self.expand_factor is not None and self.expand_factor < 1:
            raise ValueError("expand_factor must be >= 1")
        if (self.vocab_size, self.d_model, self.n_layer, self.n_head, self.d_ff) != (
            8000, 256, 6, 4, 1024,
        ):
            raise ValueError(
                "SDALM-10M shape is frozen: vocab=8000, d_model=256, "
                "n_layer=6, n_head=4, d_ff=1024"
            )


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * scale


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, config: SDALMConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        if config.mixer == "sda":
            self.sda = SDABlock(
                config.d_model, config.n_head, config.archive_cap, config.variant,
                config.beta_init, config.tau_init, config.gate_temp_init,
                config.expand_factor, config.hard_gate_eval,
            )
        elif config.mixer == "softmax":
            self.attn = SoftmaxAttention(config.d_model, config.n_head)
        else:
            self.delta = NoDecayDelta(config.d_model, config.n_head, beta=config.beta_init)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.d_ff)

    @property
    def mixer(self) -> nn.Module:
        if hasattr(self, "sda"):
            return self.sda
        if hasattr(self, "attn"):
            return self.attn
        return self.delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SDALM(nn.Module):
    """Pre-norm causal stack. The default mixer is an SDA block.

    ``mixer="softmax"`` and ``mixer="delta"`` are same-shape baselines.
    They are not Expanded and not a third hybrid.
    """

    def __init__(self, config: SDALMConfig | None = None) -> None:
        super().__init__()
        self.config = config or SDALMConfig()
        cfg = self.config
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for layer in self.layers:
            nn.init.normal_(layer.mixer.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.ffn.fc2.weight, mean=0.0, std=residual_std)

    def rule_state_nbytes(self, batch: int) -> int:
        """Sum of per-layer per-step SDA state bytes at the parameter dtype."""
        if self.config.mixer != "sda":
            raise ValueError("rule_state_nbytes is the SDA per-step state")
        dtype = self.embed.weight.dtype
        return sum(layer.sda.rule_state_nbytes(batch, dtype) for layer in self.layers)

    def logged_state_nbytes(self, batch: int, time: int) -> int:
        """Bytes of this mixer's own state at the parameter dtype.

        Recency includes the ring. No-decay delta does not. Softmax counts K
        and V for ``time``. The three figures are not forced to match.
        """
        dtype = self.embed.weight.dtype
        return sum(layer.mixer.state_nbytes(batch, time, dtype) for layer in self.layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"expected token ids (batch, time), got {tuple(tokens.shape)}")
        time = tokens.shape[1]
        if time > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {time} exceeds max_seq_len {self.config.max_seq_len}"
            )
        hidden = self.embed(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.norm(hidden))


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters, counting tied weights once."""
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        identity = id(param)
        if identity in seen:
            continue
        seen.add(identity)
        total += param.numel()
    return total


def parameter_groups(model: nn.Module) -> dict[str, int]:
    seen: set[int] = set()
    groups: dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        key = name.split(".", 1)[0]
        groups[key] = groups.get(key, 0) + param.numel()
    return groups


def main() -> None:
    cfg = SDALMConfig()
    print(
        f"d_model={cfg.d_model} n_layer={cfg.n_layer} n_head={cfg.n_head} "
        f"vocab={cfg.vocab_size} d_ff={cfg.d_ff} archive_cap={cfg.archive_cap} "
        f"max_seq_len={cfg.max_seq_len} variant_default={cfg.variant}"
    )
    for variant in VARIANTS:
        model = SDALM(SDALMConfig(variant=variant))
        count = count_parameters(model)
        extra = ""
        if variant == "expanded":
            extra = f" expand_factor={model.layers[0].sda.expand_factor}"
        print(
            f"variant={variant} param_count={count} "
            f"rule_state_nbytes_batch1={model.rule_state_nbytes(1)}{extra}"
        )
        for name, numel in parameter_groups(model).items():
            print(f"  {name}={numel}")


if __name__ == "__main__":
    main()
