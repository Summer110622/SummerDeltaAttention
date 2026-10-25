"""A3 CPU smoke for frozen SDA-Recency against two same-shape baselines.

Recency trains with the soft gate. Eval uses the hard gate. AdamW steps only
``beta_logit``, ``tau_logit``, and ``gate_log_temp``. The baselines are causal
softmax attention and no-decay delta (no archive, no gate, not Expanded).

Seeds are the LM/MQAR band only (20300001 train, 20300002 eval). CALIB, TEST,
and HOLDOUT are not read. The step count is fixed. Nothing early-stops.

Next-token cross-entropy is written to the internal log and is not a result
for the README, the papers, or RESEARCH_STATUS.md. Exit 0 means the smoke
checks passed. It does not mean a trained layer works.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

try:
    from .data_mqar import make_mqar_batch
    from .model import SDALM, SDALMConfig, count_parameters
except ImportError:
    from data_mqar import make_mqar_batch
    from model import SDALM, SDALMConfig, count_parameters


# Locked by EVAL_PROTOCOL_JA.md. Not a CLI.
TRAIN_SEED = 20300001
EVAL_SEED = 20300002
LM_BAND_START = 20300001
STEPS = 2
BATCH_SIZE = 2
N_PAIRS = 20
N_QUERIES = 4
LR = 3e-4
SDA_KNOB_NAMES = ("beta_logit", "tau_logit", "gate_log_temp")
RECENCY_PARAMS = 8_817_992
# Recency knobs are 3 vectors of n_head, times n_layer: 3 * 4 * 6.
BASELINE_PARAMS = RECENCY_PARAMS - (3 * 4 * 6)
LOG_PATH = Path(__file__).resolve().parent / "logs" / "a3_cpu_smoke.jsonl"


def _assert_lm_seed(seed: int) -> None:
    if seed < LM_BAND_START:
        raise AssertionError(
            f"seed {seed} is outside the LM/MQAR band {LM_BAND_START}-; "
            "CALIB, TEST, and HOLDOUT are not inputs to this smoke"
        )


def _is_knob(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in SDA_KNOB_NAMES


def _batch(seed: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    _assert_lm_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return make_mqar_batch(
        batch_size=BATCH_SIZE,
        n_pairs=N_PAIRS,
        n_queries=N_QUERIES,
        vocab_size=vocab_size,
        generator=generator,
    )


def _build(mixer: str) -> SDALM:
    torch.manual_seed(TRAIN_SEED)
    if mixer == "sda":
        config = SDALMConfig(mixer="sda", variant="recency", hard_gate_eval=True)
    elif mixer == "softmax":
        config = SDALMConfig(mixer="softmax", variant="recency")
    elif mixer == "delta":
        config = SDALMConfig(mixer="delta", variant="recency")
    else:
        raise AssertionError(f"unknown mixer {mixer}")
    model = SDALM(config).to(device="cpu")
    model.float()
    weight = model.embed.weight
    if weight.device.type != "cpu" or weight.dtype != torch.float32:
        raise AssertionError("A3 smoke is CPU float32 only")
    return model


def _cross_entropy(model: SDALM, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    logits = model(inputs)
    if not torch.isfinite(logits).all():
        raise AssertionError(f"{model.config.mixer} produced non-finite logits")
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
    )
    if not torch.isfinite(loss):
        raise AssertionError(f"{model.config.mixer} produced a non-finite loss")
    return float(loss.detach())


def _recency_accepted_rates(model: SDALM, inputs: torch.Tensor) -> list[float]:
    """Hard-threshold rates from each recency block's own scores."""
    hidden = model.embed(inputs)
    rates: list[float] = []
    for layer in model.layers:
        block = layer.sda
        rate = block.acceptance_rate(layer.norm1(hidden))
        if not isinstance(rate, float) or not 0.0 <= rate <= 1.0:
            raise AssertionError("recency accepted_rate missing or out of range")
        rates.append(rate)
        hidden = layer(hidden)
    return rates


def _snapshot(model: SDALM) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def _assert_unchanged(model: SDALM, before: dict[str, torch.Tensor], *, knobs_ok: bool) -> None:
    for name, param in model.named_parameters():
        if knobs_ok and _is_knob(name):
            continue
        if not torch.equal(param, before[name]):
            raise AssertionError(f"parameter changed: {name}")


def run() -> dict:
    _assert_lm_seed(TRAIN_SEED)
    _assert_lm_seed(EVAL_SEED)
    if TRAIN_SEED == EVAL_SEED:
        raise AssertionError("train and eval seeds must differ")
    if STEPS < 1:
        raise AssertionError("steps must be fixed and positive")

    recency = _build("sda")
    softmax = _build("softmax")
    delta = _build("delta")
    models = {"sda_recency": recency, "softmax": softmax, "no_decay_delta": delta}

    if count_parameters(recency) != RECENCY_PARAMS:
        raise AssertionError("recency param count left the frozen total")
    for name in ("softmax", "no_decay_delta"):
        count = count_parameters(models[name])
        if count != BASELINE_PARAMS:
            raise AssertionError(f"{name} param count {count} != {BASELINE_PARAMS}")
    if not torch.equal(recency.embed.weight, softmax.embed.weight):
        raise AssertionError("softmax init does not share the recency embedding")
    if not torch.equal(recency.embed.weight, delta.embed.weight):
        raise AssertionError("delta init does not share the recency embedding")
    if not torch.equal(recency.layers[0].mixer.qkv.weight, softmax.layers[0].mixer.qkv.weight):
        raise AssertionError("softmax QKV init diverged from recency")
    if not torch.equal(recency.layers[0].mixer.qkv.weight, delta.layers[0].mixer.qkv.weight):
        raise AssertionError("delta QKV init diverged from recency")

    for layer in delta.layers:
        beta = layer.delta.beta
        if beta.requires_grad or not torch.allclose(beta, torch.full_like(beta, 0.05)):
            raise AssertionError("no-decay delta β must be a fixed 0.05 buffer")
        if hasattr(layer, "sda"):
            raise AssertionError("no-decay delta built an SDA block")
    for model in (softmax, delta):
        for name, _param in model.named_parameters():
            if _is_knob(name):
                raise AssertionError(f"baseline has an SDA knob: {name}")
        if hasattr(model.layers[0], "sda"):
            raise AssertionError("baseline layer has an SDA block")

    cfg = recency.config
    if (cfg.d_model, cfg.n_layer, cfg.n_head, cfg.d_ff, cfg.vocab_size) != (256, 6, 4, 1024, 8000):
        raise AssertionError("frozen shape was changed")
    if cfg.variant != "recency" or cfg.mixer != "sda":
        raise AssertionError("A3 primary must be SDA-Recency")

    train_inputs, train_targets = _batch(TRAIN_SEED, cfg.vocab_size)
    eval_inputs, eval_targets = _batch(EVAL_SEED, cfg.vocab_size)
    if train_inputs.shape != eval_inputs.shape:
        raise AssertionError("train and eval batches differ in shape")
    time = int(train_inputs.shape[1])
    if time <= cfg.archive_cap:
        raise AssertionError("smoke sequence should be longer than archive_cap")
    if time > cfg.max_seq_len:
        raise AssertionError("smoke sequence exceeds max_seq_len")

    item = torch.empty((), dtype=torch.float32).element_size()
    d_head = cfg.d_model // cfg.n_head
    recency_bytes = recency.logged_state_nbytes(BATCH_SIZE, time)
    delta_bytes = delta.logged_state_nbytes(BATCH_SIZE, time)
    softmax_bytes = softmax.logged_state_nbytes(BATCH_SIZE, time)
    ring_entries = BATCH_SIZE * cfg.n_head * (2 * cfg.archive_cap * d_head) * cfg.n_layer
    fast_entries = BATCH_SIZE * cfg.n_head * d_head * d_head * cfg.n_layer
    kv_entries = BATCH_SIZE * cfg.n_head * time * d_head * 2 * cfg.n_layer
    if recency_bytes != (fast_entries + ring_entries) * item:
        raise AssertionError("recency byte count is not fast state plus ring, times item size")
    if delta_bytes != fast_entries * item:
        raise AssertionError("no-decay delta byte count includes something other than the fast state")
    if softmax_bytes != kv_entries * item:
        raise AssertionError("softmax byte count is not this sequence's K and V")
    if not (delta_bytes < recency_bytes and softmax_bytes != recency_bytes):
        raise AssertionError("baseline state bytes were copied from recency")
    if recency.rule_state_nbytes(BATCH_SIZE) != recency_bytes:
        raise AssertionError("logged recency bytes disagree with rule_state_nbytes")

    before = {name: _snapshot(model) for name, model in models.items()}
    knobs = [param for name, param in recency.named_parameters() if _is_knob(name)]
    if len(knobs) != cfg.n_layer * 3:
        raise AssertionError("expected one β, τ, and temperature parameter per layer")
    optimizer = torch.optim.AdamW(knobs, lr=LR, weight_decay=0.0)
    grouped = {id(param) for group in optimizer.param_groups for param in group["params"]}
    for name, param in recency.named_parameters():
        in_group = id(param) in grouped
        if _is_knob(name) != in_group:
            raise AssertionError(f"AdamW group mismatch at {name}")

    recency.train()
    train_ce: list[float] = []
    for step in range(STEPS):
        if not recency.training:
            raise AssertionError("training step left train mode; the hard gate would apply")
        if not all(layer.sda.hard_gate_eval for layer in recency.layers):
            raise AssertionError("hard_gate_eval must stay armed and still be ignored while training")
        recency.zero_grad(set_to_none=True)
        logits = recency(train_inputs)
        if not torch.isfinite(logits).all():
            raise AssertionError("recency training logits were non-finite")
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), train_targets.reshape(-1),
        )
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite training loss at step {step}")
        loss.backward()
        for name, param in recency.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise AssertionError(f"missing or non-finite grad at {name}")
            if _is_knob(name) and float(param.grad.abs().sum()) == 0.0:
                raise AssertionError(f"zero grad at {name}; soft gate did not train")
        optimizer.step()
        train_ce.append(float(loss.detach()))
        if len(train_ce) != step + 1:
            raise AssertionError("step loop is not the fixed schedule")
    if len(train_ce) != STEPS:
        raise AssertionError("early stop or extra step changed the schedule")
    _assert_unchanged(recency, before["sda_recency"], knobs_ok=True)
    if not any(
        not torch.equal(param, before["sda_recency"][name])
        for name, param in recency.named_parameters()
        if _is_knob(name)
    ):
        raise AssertionError("AdamW did not change any SDA knob")

    for name, model in models.items():
        model.eval()
        if model.training:
            raise AssertionError(f"{name} stayed in train mode")
    if not all(layer.sda.hard_gate_eval for layer in recency.layers):
        raise AssertionError("eval hard gate is off")

    with torch.no_grad():
        eval_ce = {
            "sda_recency": _cross_entropy(recency, eval_inputs, eval_targets),
            "softmax": _cross_entropy(softmax, eval_inputs, eval_targets),
            "no_decay_delta": _cross_entropy(delta, eval_inputs, eval_targets),
        }
        accepted = _recency_accepted_rates(recency, eval_inputs)
    _assert_unchanged(softmax, before["softmax"], knobs_ok=False)
    _assert_unchanged(delta, before["no_decay_delta"], knobs_ok=False)
    _assert_unchanged(recency, before["sda_recency"], knobs_ok=True)

    record = {
        "result": "smoke_ok",
        "exit": 0,
        "process_exit_code": 0,
        "device": "cpu",
        "dtype": "float32",
        "shape": {"vocab": 8000, "d_model": 256, "n_layer": 6, "n_head": 4, "d_ff": 1024},
        "train_seed": TRAIN_SEED,
        "eval_seed": EVAL_SEED,
        "steps": STEPS,
        "early_stop": False,
        "train_gate": "soft",
        "eval_gate": "hard",
        "adamw": "sda_knobs_only",
        "batch_size": BATCH_SIZE,
        "seq_len": time,
        "claim": "cpu_smoke_only",
        "methods": [
            {
                "name": "sda_recency",
                "param_count": count_parameters(recency),
                "state_kind": recency.layers[0].mixer.state_kind,
                "state_nbytes": recency_bytes,
                "train_ce": train_ce,
                "eval_ce": eval_ce["sda_recency"],
                "accepted_rate_per_layer": accepted,
            },
            {
                "name": "softmax",
                "param_count": count_parameters(softmax),
                "state_kind": softmax.layers[0].mixer.state_kind,
                "state_nbytes": softmax_bytes,
                "train_steps": 0,
                "eval_ce": eval_ce["softmax"],
                "accepted_rate_per_layer": None,
            },
            {
                "name": "no_decay_delta",
                "param_count": count_parameters(delta),
                "state_kind": delta.layers[0].mixer.state_kind,
                "state_nbytes": delta_bytes,
                "train_steps": 0,
                "eval_ce": eval_ce["no_decay_delta"],
                "accepted_rate_per_layer": None,
            },
        ],
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"))
    LOG_PATH.write_text(line + "\n", encoding="utf-8")
    parsed = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    if parsed["result"] != "smoke_ok" or parsed["exit"] != 0:
        raise AssertionError("internal log was not a passing record")
    if parsed["train_seed"] != TRAIN_SEED or parsed["eval_seed"] != EVAL_SEED:
        raise AssertionError("internal log lost the LM-band seeds")
    rates = [method["accepted_rate_per_layer"] for method in parsed["methods"]]
    if rates[1] is not None or rates[2] is not None:
        raise AssertionError("accepted_rate was copied onto a baseline")
    print(
        "a3_cpu_smoke device=cpu dtype=float32 "
        f"steps={STEPS} early_stop=false train_seed={TRAIN_SEED} eval_seed={EVAL_SEED}"
    )
    print("adamw_group=sda_knobs_only mixers=sda_recency,softmax,no_decay_delta")
    print("log=research/sdalm_10m/logs/a3_cpu_smoke.jsonl")
    print("smoke_ok=true")
    return record


def main() -> None:
    run()


if __name__ == "__main__":
    main()
