"""A few CPU AdamW steps on one synthetic batch.

Prints the parameter count and the smoke loss trajectory. Those losses are
an optimization trace on a fixed batch. They are not an evaluation, not
evidence that a trained layer works, and not a comparison of Recency
against Expanded.

Exit status is 0 when the frozen shape's parameter count is inside 1e7 ± 20%,
the delta scan matches a sequential reference, gradients are finite, and
acceptance is reported only for the recency block.
"""
from __future__ import annotations

import argparse

import torch

try:
    from .data_mqar import make_mqar_batch
    from .model import SDALM, SDALMConfig, count_parameters
    from .sda_block import (
        causal_band,
        expanded_state_entries,
        nbytes_for_entries,
        recency_state_entries,
        resolve_expand_factor,
        scan_matches_reference,
    )
except ImportError:
    from data_mqar import make_mqar_batch
    from model import SDALM, SDALMConfig, count_parameters
    from sda_block import (
        causal_band,
        expanded_state_entries,
        nbytes_for_entries,
        recency_state_entries,
        resolve_expand_factor,
        scan_matches_reference,
    )


# Frozen-shape totals, untied LM head, no position table. Change these only
# when that choice changes on purpose. They are parameter counts, not metrics.
DEFAULT_RECENCY_PARAMS = 8_817_992
DEFAULT_EXPANDED_PARAMS = 8_817_944
PARAM_LO = 8_000_000
PARAM_HI = 12_000_000


def _assert_finite_grads(model: torch.nn.Module, *, require_nonzero: tuple[str, ...] = ()) -> None:
    missing = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
            continue
        if not torch.isfinite(param.grad).all():
            raise AssertionError(f"non-finite grad at {name}")
        if any(name.endswith(suffix) for suffix in require_nonzero):
            if param.grad.abs().sum().item() == 0.0:
                raise AssertionError(f"zero grad at {name}")
    if missing:
        raise AssertionError("missing grads: " + ", ".join(missing))


def _build(variant: str, seed: int) -> SDALM:
    torch.manual_seed(seed)
    model = SDALM(SDALMConfig(variant=variant))
    model.to("cpu")
    model.train()
    return model


def _one_backward(model: SDALM, inputs: torch.Tensor, targets: torch.Tensor,
                  nonzero: tuple[str, ...]) -> None:
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
    )
    if not torch.isfinite(loss):
        raise AssertionError("non-finite loss")
    loss.backward()
    _assert_finite_grads(model, require_nonzero=nonzero)


def _check_budgets_and_bytes(model: SDALM) -> None:
    cfg = model.config
    if (cfg.d_model, cfg.n_layer, cfg.n_head, cfg.d_ff, cfg.vocab_size) != (256, 6, 4, 1024, 8000):
        raise AssertionError("frozen shape was changed")
    if cfg.variant != "recency":
        raise AssertionError("default config must be recency")
    d_head = cfg.d_model // cfg.n_head
    factor = resolve_expand_factor(d_head, cfg.archive_cap, None)
    recency = recency_state_entries(d_head, cfg.archive_cap)
    expanded = expanded_state_entries(d_head, factor)
    if recency != expanded:
        raise AssertionError(f"default budgets differ: recency {recency} vs expanded {expanded}")
    item = torch.empty((), dtype=model.embed.weight.dtype).element_size()
    entries = recency * cfg.n_head * cfg.n_layer
    if model.rule_state_nbytes(1) != entries * item:
        raise AssertionError("rule_state_nbytes is not entry count times item size")
    if nbytes_for_entries(5, torch.float32) == nbytes_for_entries(5, torch.float64):
        raise AssertionError("byte count ignored dtype")
    band = causal_band(5, 3, torch.device("cpu"))
    if band[0].tolist() != [True, False, False, False, False]:
        raise AssertionError("ring does not keep only the current token at t=0")
    if band[4].tolist() != [False, False, True, True, True]:
        raise AssertionError("ring width is not archive_cap")


def _check_gate_scope(recency: SDALM, expanded: SDALM) -> None:
    probe = torch.randn(2, 16, recency.config.d_model)
    rec_fields = recency.layers[0].sda.gate_fields(probe)
    exp_fields = expanded.layers[0].sda.gate_fields(probe)
    if "accepted_rate" not in rec_fields or "hit_rate" in rec_fields:
        raise AssertionError("recency gate fields must carry accepted_rate only")
    rate = rec_fields["accepted_rate"]
    if not isinstance(rate, float) or not 0.0 <= rate <= 1.0:
        raise AssertionError("recency accepted_rate out of range")
    if "accepted_rate" in exp_fields or "hit_rate" in exp_fields:
        raise AssertionError("expanded must not inherit a recency acceptance or hit rate")
    if exp_fields.get("variant") != "expanded":
        raise AssertionError("expanded report lost its variant")


def _hard_gate_is_finite(model: SDALM, inputs: torch.Tensor) -> None:
    flags = [layer.sda.hard_gate_eval for layer in model.layers]
    was_training = model.training
    try:
        for layer in model.layers:
            layer.sda.hard_gate_eval = True
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise AssertionError("hard-gate eval forward was non-finite")
    finally:
        for layer, flag in zip(model.layers, flags):
            layer.sda.hard_gate_eval = flag
        if was_training:
            model.train()


def run(variant: str, steps: int, seed: int) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if variant not in ("recency", "expanded"):
        raise ValueError("variant must be 'recency' or 'expanded'")
    if not scan_matches_reference():
        raise AssertionError("vectorized delta scan disagreed with the sequential reference")
    print("SDALM-10M CPU smoke — finite grads and a loss trajectory only; not an LM evaluation")
    print(f"device=cpu seed={seed} primary_variant={variant} steps={steps}")
    _check_budgets_and_bytes(SDALM(SDALMConfig()))
    cfg = SDALMConfig()
    print(
        f"d_model={cfg.d_model} n_layer={cfg.n_layer} n_head={cfg.n_head} "
        f"vocab={cfg.vocab_size} d_ff={cfg.d_ff} archive_cap={cfg.archive_cap}"
    )
    primary = _build(variant, seed)
    count = count_parameters(primary)
    print(f"param_count={count}")
    print(f"rule_state_nbytes_batch1={primary.rule_state_nbytes(1)}")
    if not PARAM_LO <= count <= PARAM_HI:
        raise AssertionError(f"param_count {count} outside [{PARAM_LO}, {PARAM_HI}]")
    if variant == "recency" and count != DEFAULT_RECENCY_PARAMS:
        raise AssertionError(f"recency param_count {count} != {DEFAULT_RECENCY_PARAMS}")
    if variant == "expanded" and count != DEFAULT_EXPANDED_PARAMS:
        raise AssertionError(f"expanded param_count {count} != {DEFAULT_EXPANDED_PARAMS}")

    other = "expanded" if variant == "recency" else "recency"
    other_model = _build(other, seed)
    other_count = count_parameters(other_model)
    expected_other = DEFAULT_EXPANDED_PARAMS if other == "expanded" else DEFAULT_RECENCY_PARAMS
    if other_count != expected_other:
        raise AssertionError(f"{other} param_count {other_count} != {expected_other}")
    recency_model = primary if variant == "recency" else other_model
    expanded_model = other_model if variant == "recency" else primary
    _check_gate_scope(recency_model, expanded_model)
    print("accepted_rate_scope=recency_only")
    print("scan_matches_reference=true")

    names = [name for name, _ in primary.named_parameters()]
    if variant == "recency":
        if not any(name.endswith("tau_logit") for name in names):
            raise AssertionError("recency config is missing tau")
        beta = torch.sigmoid(primary.layers[0].sda.beta_logit.detach()).mean()
        tau = torch.tanh(primary.layers[0].sda.tau_logit.detach()).mean()
        print(f"init_beta={beta.item():.6f} init_tau={tau.item():.6f}")
        nonzero = ("beta_logit", "tau_logit", "gate_log_temp")
    else:
        if any(name.endswith("tau_logit") for name in names):
            raise AssertionError("expanded config must not carry tau")
        factor = primary.layers[0].sda.expand_factor
        beta = torch.sigmoid(primary.layers[0].sda.beta_logit.detach()).mean()
        print(f"init_beta={beta.item():.6f} expand_factor={factor}")
        nonzero = ("beta_logit",)

    generator = torch.Generator()
    generator.manual_seed(seed)
    inputs, targets = make_mqar_batch(
        batch_size=2, n_pairs=20, n_queries=4, vocab_size=cfg.vocab_size, generator=generator,
    )
    if inputs.shape[1] <= cfg.archive_cap:
        raise AssertionError("smoke sequence should be longer than archive_cap so the ring evicts")

    optimizer = torch.optim.AdamW(primary.parameters(), lr=3e-4, weight_decay=0.0)
    losses: list[float] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = primary(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        )
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite loss at step {step}")
        loss.backward()
        _assert_finite_grads(primary, require_nonzero=nonzero)
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        print(f"step={step} loss={value:.6f}")
    print("loss_trajectory=" + ",".join(f"{value:.6f}" for value in losses))
    print("grads_finite=true")
    _hard_gate_is_finite(primary if variant == "recency" else recency_model, inputs)
    print("hard_gate_eval_finite=true")

    other_nonzero = ("beta_logit",) if other == "expanded" else ("beta_logit", "tau_logit", "gate_log_temp")
    _one_backward(other_model, inputs, targets, other_nonzero)
    print(f"other_variant={other} param_count={other_count} grads_finite=true")
    print("smoke_ok=true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("recency", "expanded"), default="recency")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.variant, args.steps, args.seed)


if __name__ == "__main__":
    main()
