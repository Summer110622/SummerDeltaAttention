"""CALIB-only ρ OFAT for frozen SDA-Recency on the A3 MQAR/hard schedule.

One factor moves: ``archive_cap``. Layout, β, τ, the soft-train / hard-eval
gate, and knobs-only AdamW stay at the A3 settings in ``a3_cpu_smoke.py`` and
``EVAL_PROTOCOL_JA.md``. The step count is fixed. Nothing early-stops.
τ stays 0.75. This script does not re-lock it.

Train batches are the ordinary MQAR schedule (uniform pair index). CALIB
eval only uses the exact-key mix: ``pi_exact = 1``, ``sigma = 0``, and the
source is inside the shared small cap when that ring still holds a pair key.
The three cells share that eval sequence. TEST, HOLDOUT, and the LM band
stay closed.

The critic reads ``accepted_rate`` before next-token CE. A cell at 0 makes
ρ unidentifiable. That is not a claim that the cap has no main effect.

``DESIGN.md`` and the protocol name a horizon ``C`` and do not assign it an
integer. This round sets ``C = 128`` for the large cell only:
``round(0.75 * C) = 96``. Base stays the Core point ``archive_cap = 32``
and is not ``round(0.5 * C)``. Small stays the explicit low cell 8.
``round(0.25 * 128) = 32`` would collide with that Core point, so small is
not ``0.25 * C`` (that product is 8 only when ``C = 32``, which is not the
horizon used here). The caps are monotonic: 8 < 32 < 96. Large is not
clipped down to the Core cap.

Seeds are the CALIB band only, split into four disjoint train/eval pairs.
TEST, HOLDOUT, and the LM/MQAR band ``20300001–`` are not read.

The log stores next-token cross-entropy plus the internal MSE, hit,
accepted rate, and KiB. The critic may leave CE unread. Exit 0 means the
grid finished and the float16 gate still PASSed. It does not mean a trained
layer works, it is not a language-model result, and a closed gate is not a
null effect of ρ.

Run from the repository root::

    python research/sdalm_10m/ablation_rho.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path

import torch

try:
    from .data_mqar import make_mqar_batch, make_mqar_exact_key_batch
    from .gate_f16_dualwrite import CALIB_SEEDS as GATE_CALIB_SEEDS
    from .gate_f16_dualwrite import LM_BAND_START, main as gate_main
    from .model import SDALM, SDALMConfig, count_parameters
    from .sda_block import _unit, causal_band, recency_state_entries
except ImportError:
    from data_mqar import make_mqar_batch, make_mqar_exact_key_batch
    from gate_f16_dualwrite import CALIB_SEEDS as GATE_CALIB_SEEDS
    from gate_f16_dualwrite import LM_BAND_START, main as gate_main
    from model import SDALM, SDALMConfig, count_parameters
    from sda_block import _unit, causal_band, recency_state_entries


# Locked with the A3 smoke. Not a CLI.
CALIB_SEEDS = tuple(range(20250001, 20250009))
TRAIN_SEEDS = CALIB_SEEDS[:4]
EVAL_SEEDS = CALIB_SEEDS[4:]
TEST_SEEDS = tuple(range(20261023, 20261039))
HOLDOUT_SEEDS = tuple(range(20270001, 20270009))
STEPS = 2
BATCH_SIZE = 2
N_PAIRS = 20
N_QUERIES = 4
LR = 3e-4
SDA_KNOB_NAMES = ("beta_logit", "tau_logit", "gate_log_temp")
RECENCY_PARAMS = 8_817_992
# Protocol names C and does not number it. 128 is the large-cell horizon only:
# round(0.75 * 128) = 96. Small is the explicit low cell 8, not round(0.25 * C)
# (that product is 32 and collides with the Core cap). Base stays 32.
C_HORIZON = 128
CORE_ARCHIVE_CAP = 32
SMALL_ARCHIVE_CAP = 8
BETA_INIT = 0.05
TAU_INIT = 0.75
# CALIB eval only. Not a CLI, and not applied to the train batch.
EVAL_PI_EXACT = 1.0
EVAL_SIGMA = 0.0
# Shared across cells. A source inside this cap is inside every larger ring.
# Using the cell cap here would change the eval tokens with ρ.
EVAL_SOURCE_CAP = SMALL_ARCHIVE_CAP
PREFER_ACCEPT_MEAN = 0.05
LOG_DIR = Path(__file__).resolve().parent / "logs"
JSONL_PATH = LOG_DIR / "ablation_rho_calib.jsonl"
JSON_PATH = LOG_DIR / "ablation_rho_calib.json"

MSE_DEFINITION = (
    "mean over MQAR answer positions of sum_v (softmax(logit)_v - 1_{v=target})^2; "
    "internal Brier score on this cell's hard-eval logits, not the synthetic-face value MSE"
)
HIT_DEFINITION = (
    "fraction of MQAR query positions where this recency block's argmax over strictly "
    "earlier ring positions has the same token id as the query; mean over heads, queries, "
    "batch, and layers. The current position is excluded: it is the query token, not a retrieved write"
)
ACCEPTED_DEFINITION = (
    "mean over layers of SDABlock.acceptance_rate on the eval batch: fraction of positions "
    "whose best in-ring cosine, including the current position, clears that block's own tau"
)
KIB_DEFINITION = (
    "one-sequence six-layer recency rule state: entries * float32 itemsize / 1024. "
    "Not a hardcoded * 8, not the scan workspace, batch 1"
)
CE_DEFINITION = (
    "full-sequence next-token cross-entropy on the hard-eval forward, same reduction as a3_cpu_smoke"
)


def _caps() -> dict[str, dict[str, int | float | bool | str | None]]:
    """Explicit caps 8 / 32 / 96. ``C`` sizes only the large cell.

    ``round(0.25 * C)`` at this horizon equals the Core cap, so the small
    cell does not use that formula. Large is ``round(0.75 * C)`` and is not
    clipped to the Core cap: a clip would put large at or below base.
    """
    quarter = round(0.25 * C_HORIZON)
    large = round(0.75 * C_HORIZON)
    small = SMALL_ARCHIVE_CAP
    if C_HORIZON != 128 or quarter != CORE_ARCHIVE_CAP:
        raise AssertionError(
            "C=128 is what makes round(0.25*C) collide with the core cap; "
            "small is the explicit cell 8 for that reason"
        )
    if small == quarter:
        raise AssertionError("small used round(0.25*C) and collided with the core cap")
    if small != 8 or large != 96 or CORE_ARCHIVE_CAP != 32:
        raise AssertionError(f"caps left 8 / 32 / 96, got {small}, {CORE_ARCHIVE_CAP}, {large}")
    if not (small < CORE_ARCHIVE_CAP < large):
        raise AssertionError("archive_cap is not strictly increasing small < base < large")
    if large < 64:
        raise AssertionError("large must be at least 64 and must exceed the core cap")
    cells = {
        "small": {
            "cell": "small",
            "rho": None,
            "archive_cap": small,
            "clipped_to_core": False,
            "fraction_of_C": False,
            "role": (
                "explicit low cell 8; not round(0.25 * C). "
                "round(0.25 * 128) = 32 collides with the core cap. "
                "round(0.25 * C) = 8 only when C = 32, which is not this horizon"
            ),
        },
        "base": {
            "cell": "base",
            "rho": None,
            "archive_cap": CORE_ARCHIVE_CAP,
            "clipped_to_core": False,
            "fraction_of_C": False,
            "role": "core budget point 32, not round(0.5 * C)",
        },
        "large": {
            "cell": "large",
            "rho": large / C_HORIZON,
            "archive_cap": large,
            "clipped_to_core": False,
            "fraction_of_C": True,
            "role": "round(0.75 * C) with C = 128; exceeds the core cap; not clipped to it",
        },
    }
    budgets = [int(cells[name]["archive_cap"]) for name in ("small", "base", "large")]
    if budgets != [8, 32, 96]:
        raise AssertionError(f"rho grid left {{8, 32, 96}}: {budgets}")
    if any(bool(cells[name]["clipped_to_core"]) for name in ("small", "base", "large")):
        raise AssertionError("a cell was clipped to the core cap")
    if cells["large"]["rho"] != 0.75:
        raise AssertionError("large rho left 0.75")
    if cells["base"]["archive_cap"] != 32:
        raise AssertionError("base cell left the core archive_cap of 32")
    return cells


def _assert_seed_bands() -> None:
    if CALIB_SEEDS != tuple(GATE_CALIB_SEEDS):
        raise AssertionError("ablation CALIB seeds drifted from the f16 gate")
    if len(CALIB_SEEDS) != 8 or len(TRAIN_SEEDS) != 4 or len(EVAL_SEEDS) != 4:
        raise AssertionError("CALIB split must cover 8 seeds as 4 train + 4 eval")
    if set(TRAIN_SEEDS) & set(EVAL_SEEDS):
        raise AssertionError("a CALIB seed is both a train batch and an eval batch")
    if set(TRAIN_SEEDS) | set(EVAL_SEEDS) != set(CALIB_SEEDS):
        raise AssertionError("train/eval split does not cover CALIB")
    banned = set(TEST_SEEDS) | set(HOLDOUT_SEEDS)
    if set(CALIB_SEEDS) & banned:
        raise AssertionError("CALIB overlaps TEST or HOLDOUT")
    if any(seed >= LM_BAND_START for seed in CALIB_SEEDS):
        raise AssertionError("CALIB overlaps the LM/MQAR band")
    if STEPS != 2 or BATCH_SIZE != 2 or N_PAIRS != 20 or N_QUERIES != 4:
        raise AssertionError("MQAR schedule drifted from a3_cpu_smoke")
    if TAU_INIT != 0.75 or BETA_INIT != 0.05:
        raise AssertionError("tau or beta init was re-locked")
    if EVAL_PI_EXACT != 1.0 or EVAL_SIGMA != 0.0:
        raise AssertionError("CALIB eval mix left pi_exact=1 and sigma=0")
    if EVAL_SOURCE_CAP != SMALL_ARCHIVE_CAP or EVAL_SOURCE_CAP != 8:
        raise AssertionError("eval source cap must stay the shared small cap 8")


def _require_gate() -> dict:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = gate_main([])
    lines = [line for line in buffer.getvalue().splitlines() if line]
    if code != 0 or not lines or lines[0] != "PASS":
        raise AssertionError("f16 dual-write gate did not PASS; rho grid is not allowed to start")
    summary = json.loads(lines[-1])
    if summary.get("result") != "PASS" or summary.get("exit") != 0:
        raise AssertionError("f16 gate JSON was not a PASS")
    if list(summary.get("seeds", [])) != list(CALIB_SEEDS):
        raise AssertionError("f16 gate judged a different seed list")
    return {"result": "PASS", "exit": 0, "seeds": list(CALIB_SEEDS)}


def _is_knob(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in SDA_KNOB_NAMES


def _reject_seed(seed: int) -> None:
    if seed not in CALIB_SEEDS:
        raise AssertionError(f"seed {seed} is outside CALIB; TEST, HOLDOUT, and the LM band stay closed")
    if seed >= LM_BAND_START or seed in TEST_SEEDS or seed in HOLDOUT_SEEDS:
        raise AssertionError(f"seed {seed} is not a CALIB input")


def _load_train_batch(seed: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ordinary MQAR batch plus ``which``. Uniform pair index. Not the eval mix."""
    _reject_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs, targets = make_mqar_batch(
        batch_size=BATCH_SIZE,
        n_pairs=N_PAIRS,
        n_queries=N_QUERIES,
        vocab_size=vocab_size,
        generator=generator,
    )
    replay = torch.Generator(device="cpu")
    replay.manual_seed(seed)
    keys = torch.randint(0, vocab_size, (BATCH_SIZE, N_PAIRS), generator=replay)
    values = torch.randint(0, vocab_size, (BATCH_SIZE, N_PAIRS), generator=replay)
    which = torch.randint(0, N_PAIRS, (BATCH_SIZE, N_QUERIES), generator=replay)
    pairs = torch.stack((keys, values), dim=-1).reshape(BATCH_SIZE, 2 * N_PAIRS)
    query = torch.gather(keys, 1, which)
    answer = torch.gather(values, 1, which)
    queries = torch.stack((query, answer), dim=-1).reshape(BATCH_SIZE, 2 * N_QUERIES)
    sequence = torch.cat((pairs, queries), dim=1)
    replay_in, replay_tg = sequence[:, :-1].contiguous(), sequence[:, 1:].contiguous()
    if not torch.equal(inputs, replay_in) or not torch.equal(targets, replay_tg):
        raise AssertionError("MQAR replay diverged from make_mqar_batch")
    expected_time = 2 * N_PAIRS + 2 * N_QUERIES - 1
    if inputs.shape != (BATCH_SIZE, expected_time):
        raise AssertionError(f"MQAR length {tuple(inputs.shape)} != {(BATCH_SIZE, expected_time)}")
    qpos = _query_positions()
    if not torch.equal(inputs[:, qpos], query):
        raise AssertionError("query positions do not hold the sampled keys")
    if not torch.equal(targets[:, qpos], answer):
        raise AssertionError("answer positions do not hold the paired values")
    return inputs, targets, which


def _load_eval_batch(seed: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact-key CALIB eval batch. Same sequence for every archive cap."""
    _reject_seed(seed)
    if EVAL_PI_EXACT != 1.0 or EVAL_SIGMA != 0.0:
        raise AssertionError("refusing to build a CALIB eval batch that is not pi_exact=1, sigma=0")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs, targets, which = make_mqar_exact_key_batch(
        batch_size=BATCH_SIZE,
        n_pairs=N_PAIRS,
        n_queries=N_QUERIES,
        vocab_size=vocab_size,
        source_cap=EVAL_SOURCE_CAP,
        pi_exact=EVAL_PI_EXACT,
        sigma=EVAL_SIGMA,
        generator=generator,
    )
    expected_time = 2 * N_PAIRS + 2 * N_QUERIES - 1
    if inputs.shape != (BATCH_SIZE, expected_time) or which.shape != (BATCH_SIZE, N_QUERIES):
        raise AssertionError("exact-key eval batch has the wrong shape")
    qpos = _query_positions()
    keys = inputs[:, 0:2 * N_PAIRS:2]
    query = inputs[:, qpos]
    source = torch.gather(keys, 1, which)
    if not torch.equal(query, source):
        raise AssertionError("eval query token is not the source key")
    if targets[:, qpos].shape != query.shape:
        raise AssertionError("eval answer positions do not match the queries")
    distance = qpos.unsqueeze(0) - 2 * which
    if not bool((distance > 0).all()):
        raise AssertionError("eval query is not strictly after its source key")
    # Queries that have a source inside the shared cap must use one. The last
    # query on this length sits 8 steps after the final pair key, so cap 8
    # has no in-ring source there; that row copies the nearest key instead.
    for query_index in range(N_QUERIES):
        q_distance = distance[:, query_index]
        if query_index < N_QUERIES - 1:
            if not bool((q_distance < EVAL_SOURCE_CAP).all()):
                raise AssertionError("an earlier eval query left the shared small ring")
        else:
            if not bool((q_distance == 8).all()):
                raise AssertionError("the last eval query is not the nearest key at distance 8")
    return inputs, targets, which


def _query_positions() -> torch.Tensor:
    # Full sequence is k v (n_pairs) then q a (n_queries). Inputs drop the last answer,
    # and every query index still lands inside that prefix.
    positions = [2 * N_PAIRS + 2 * index for index in range(N_QUERIES)]
    if positions[-1] >= 2 * N_PAIRS + 2 * N_QUERIES - 1:
        raise AssertionError("a query was dropped with the final target token")
    return torch.tensor(positions, dtype=torch.long)


def _build(archive_cap: int, seed: int) -> SDALM:
    _reject_seed(seed)
    torch.manual_seed(seed)
    model = SDALM(SDALMConfig(
        mixer="sda",
        variant="recency",
        archive_cap=archive_cap,
        hard_gate_eval=True,
        beta_init=BETA_INIT,
        tau_init=TAU_INIT,
    )).to(device="cpu")
    model.float()
    weight = model.embed.weight
    if weight.device.type != "cpu" or weight.dtype != torch.float32:
        raise AssertionError("rho OFAT is CPU float32 only")
    if count_parameters(model) != RECENCY_PARAMS:
        raise AssertionError("recency param count left the frozen total")
    cfg = model.config
    if (cfg.d_model, cfg.n_layer, cfg.n_head, cfg.d_ff, cfg.vocab_size) != (256, 6, 4, 1024, 8000):
        raise AssertionError("frozen shape was changed")
    if cfg.variant != "recency" or cfg.mixer != "sda":
        raise AssertionError("rho OFAT is SDA-Recency only")
    if cfg.archive_cap != archive_cap:
        raise AssertionError("archive_cap was not applied")
    return model


def _snapshot(model: SDALM) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def _assert_same_init(models: dict[str, SDALM]) -> None:
    reference = next(iter(models.values()))
    ref = _snapshot(reference)
    for name, model in models.items():
        got = _snapshot(model)
        if got.keys() != ref.keys():
            raise AssertionError(f"{name} init parameters differ in name")
        for key, value in got.items():
            if not torch.equal(value, ref[key]):
                raise AssertionError(f"init diverged across archive caps at {key}")


def _train(model: SDALM, inputs: torch.Tensor, targets: torch.Tensor,
           before: dict[str, torch.Tensor]) -> list[float]:
    knobs = [param for name, param in model.named_parameters() if _is_knob(name)]
    if len(knobs) != model.config.n_layer * 3:
        raise AssertionError("expected one β, τ, and temperature parameter per layer")
    optimizer = torch.optim.AdamW(knobs, lr=LR, weight_decay=0.0)
    grouped = {id(param) for group in optimizer.param_groups for param in group["params"]}
    for name, param in model.named_parameters():
        if _is_knob(name) != (id(param) in grouped):
            raise AssertionError(f"AdamW group mismatch at {name}")
    model.train()
    train_ce: list[float] = []
    for step in range(STEPS):
        if not model.training:
            raise AssertionError("training step left train mode; the hard gate would apply")
        if not all(layer.sda.hard_gate_eval for layer in model.layers):
            raise AssertionError("hard_gate_eval must stay armed and still be ignored while training")
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise AssertionError("training logits were non-finite")
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        )
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite training loss at step {step}")
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise AssertionError(f"missing or non-finite grad at {name}")
            if _is_knob(name) and float(param.grad.abs().sum()) == 0.0:
                raise AssertionError(f"zero grad at {name}; soft gate did not train")
        optimizer.step()
        train_ce.append(float(loss.detach()))
    if len(train_ce) != STEPS:
        raise AssertionError("early stop or extra step changed the schedule")
    for name, param in model.named_parameters():
        if _is_knob(name):
            continue
        if not torch.equal(param, before[name]):
            raise AssertionError(f"non-knob parameter changed: {name}")
    if not any(
        not torch.equal(param, before[name])
        for name, param in model.named_parameters()
        if _is_knob(name)
    ):
        raise AssertionError("AdamW did not change any SDA knob")
    return train_ce


def _source_in_ring(which: torch.Tensor, archive_cap: int) -> float:
    qpos = _query_positions()
    source = 2 * which
    distance = qpos.unsqueeze(0) - source
    if not bool((distance > 0).all()):
        raise AssertionError("query is not strictly after its source key")
    inside = distance < archive_cap
    return float(inside.to(dtype=torch.float32).mean())


def _best_in_ring_cosine(block, hidden: torch.Tensor) -> float:
    """Max cosine inside this block's ring, including the current position."""
    if hidden.ndim != 3:
        raise AssertionError("expected prenorm hidden (batch, time, d_model)")
    batch, time, _ = hidden.shape
    qkv = block.qkv(hidden).view(batch, time, 3, block.n_head, block.d_head)
    query, key, _value = (tensor.permute(0, 2, 1, 3) for tensor in qkv.unbind(dim=2))
    scores = torch.einsum("bhtd,bhsd->bhts", _unit(query), _unit(key))
    valid = causal_band(time, block.archive_cap, hidden.device)
    best = scores.masked_fill(~valid, -1e9).amax(dim=-1)
    if not torch.isfinite(best).all():
        raise AssertionError("in-ring cosine was non-finite")
    return float(best.max())


def _query_hit_rate(block, hidden: torch.Tensor, tokens: torch.Tensor) -> float:
    """Key-id hit from this block's past-ring argmax. Self is not a retrieval."""
    if hidden.ndim != 3:
        raise AssertionError("expected prenorm hidden (batch, time, d_model)")
    batch, time, _ = hidden.shape
    qkv = block.qkv(hidden).view(batch, time, 3, block.n_head, block.d_head)
    query, key, _value = (tensor.permute(0, 2, 1, 3) for tensor in qkv.unbind(dim=2))
    scores = torch.einsum("bhtd,bhsd->bhts", _unit(query), _unit(key))
    valid = causal_band(time, block.archive_cap, hidden.device)
    past = valid & ~torch.eye(time, device=hidden.device, dtype=torch.bool)
    qpos = _query_positions()
    # past[t, s]; each query row needs one earlier source. An all-masked argmax is index 0.
    if not bool(past.index_select(0, qpos).any(dim=1).all()):
        raise AssertionError("a query has an empty past ring; hit_rate would be an empty argmax")
    chosen = scores.masked_fill(~past, -1e9).argmax(dim=-1).index_select(dim=2, index=qpos)
    source_token = torch.gather(
        tokens.view(batch, 1, time).expand(batch, block.n_head, time),
        2,
        chosen,
    )
    current = tokens.index_select(1, qpos).view(batch, 1, -1).expand_as(source_token)
    return float((source_token == current).to(dtype=torch.float32).mean())


def _rule_kib(model: SDALM) -> tuple[int, float]:
    nbytes = model.rule_state_nbytes(1)
    item = torch.empty((), dtype=torch.float32).element_size()
    cfg = model.config
    d_head = cfg.d_model // cfg.n_head
    entries = recency_state_entries(d_head, cfg.archive_cap) * cfg.n_head * cfg.n_layer
    if nbytes != entries * item:
        raise AssertionError("KiB is not entry count times float32 item size")
    if item == 8:
        raise AssertionError("float32 item size was counted as 8")
    return nbytes, nbytes / 1024.0


@torch.no_grad()
def _evaluate(model: SDALM, inputs: torch.Tensor, targets: torch.Tensor,
              which: torch.Tensor) -> dict[str, float | list[float]]:
    model.eval()
    if model.training:
        raise AssertionError("eval stayed in train mode")
    if not all(layer.sda.hard_gate_eval for layer in model.layers):
        raise AssertionError("eval hard gate is off")
    qpos = _query_positions()
    hidden = model.embed(inputs)
    hits: list[float] = []
    accepted: list[float] = []
    cosines: list[float] = []
    for layer in model.layers:
        prenorm = layer.norm1(hidden)
        rate = layer.sda.acceptance_rate(prenorm)
        if not isinstance(rate, float) or not 0.0 <= rate <= 1.0:
            raise AssertionError("accepted_rate missing or out of range")
        hit = _query_hit_rate(layer.sda, prenorm, inputs)
        if not 0.0 <= hit <= 1.0:
            raise AssertionError("hit_rate out of range")
        accepted.append(rate)
        hits.append(hit)
        cosines.append(_best_in_ring_cosine(layer.sda, prenorm))
        hidden = layer(hidden)
    logits = model.lm_head(model.norm(hidden))
    if not torch.isfinite(logits).all():
        raise AssertionError("eval logits were non-finite")
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
    )
    answer_logits = logits.index_select(1, qpos)
    answer_targets = targets.index_select(1, qpos)
    answer_loss = torch.nn.functional.cross_entropy(
        answer_logits.reshape(-1, answer_logits.size(-1)),
        answer_targets.reshape(-1),
    )
    probs = torch.softmax(answer_logits, dim=-1)
    one_hot = torch.nn.functional.one_hot(answer_targets, probs.size(-1)).to(dtype=probs.dtype)
    mse = (probs - one_hot).pow(2).sum(dim=-1).mean()
    if not torch.isfinite(loss) or not torch.isfinite(answer_loss) or not torch.isfinite(mse):
        raise AssertionError("non-finite eval metric")
    beta = torch.stack([torch.sigmoid(layer.sda.beta_logit.detach()) for layer in model.layers])
    tau = torch.stack([torch.tanh(layer.sda.tau_logit.detach()) for layer in model.layers])
    nbytes, kib = _rule_kib(model)
    return {
        "eval_ce": float(loss),
        "eval_ce_answer": float(answer_loss),
        "mse": float(mse),
        "hit_rate": float(sum(hits) / len(hits)),
        "hit_rate_per_layer": hits,
        "accepted_rate": float(sum(accepted) / len(accepted)),
        "accepted_rate_per_layer": accepted,
        "best_in_ring_cosine_max": float(max(cosines)),
        "best_in_ring_cosine_max_per_layer": cosines,
        "source_in_ring_rate": _source_in_ring(which, model.config.archive_cap),
        "state_nbytes": nbytes,
        "kib": kib,
        "beta_mean": float(beta.mean()),
        "tau_mean": float(tau.mean()),
        "tau_min": float(tau.min()),
        "tau_max": float(tau.max()),
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise AssertionError("empty aggregate")
    return float(sum(values) / len(values))


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        raise AssertionError("sample sd needs two or more pairs")
    center = _mean(values)
    var = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(var))


def _critic_health(summaries: list[dict]) -> dict:
    """Decide whether next-token CE can be read. Uses acceptance only."""
    cells = []
    for row in summaries:
        rates = [float(value) for value in row["per_seed_accepted_rate"]]
        mean = float(row["accepted_rate"])
        every = all(value > 0.0 for value in rates)
        cosine_below = all(
            float(cosine) < float(tau)
            for cosine, tau in zip(row["per_seed_best_in_ring_cosine_max"], row["per_seed_tau_min"])
        )
        cells.append({
            "cell": row["cell"],
            "accepted_rate": mean,
            "every_seed_positive": every,
            "mean_at_least_floor": mean >= PREFER_ACCEPT_MEAN,
            "cosine_below_tau_every_seed": cosine_below,
            "per_seed_accepted_rate": rates,
        })
    each_positive = all(cell["accepted_rate"] > 0.0 for cell in cells)
    prefer = all(cell["every_seed_positive"] and cell["mean_at_least_floor"] for cell in cells)
    if not each_positive:
        effect = "UNIDENTIFIABLE"
        interpretable = False
        follow = "stopped_for_human_review"
    else:
        effect = "exploratory_calib"
        interpretable = True
        follow = "ce_recorded" if prefer else "ce_recorded_prefer_gate_missed"
    return {
        "checked_before_ce": True,
        "rule": (
            "each cell accepted_rate > 0 before CE is read; prefer every CALIB eval "
            "seed > 0 and mean >= 0.05. A zero cell is UNIDENTIFIABLE, not a null effect"
        ),
        "prefer_mean_floor": PREFER_ACCEPT_MEAN,
        "cells": cells,
        "each_cell_accepted_rate_positive": each_positive,
        "prefer_gate": prefer,
        "passed": each_positive and prefer,
        "rho_effect": effect,
        "ce_interpretable": interpretable,
        "follow_up": follow,
        "tau_relock": False,
        "null_effect_claimed": False,
    }


def run() -> dict:
    _assert_seed_bands()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise AssertionError(f"could not lock interop threads before the grid: {exc}") from exc
    cells = _caps()
    gate = _require_gate()
    order = ("small", "base", "large")
    runs: list[dict] = []
    for train_seed, eval_seed in zip(TRAIN_SEEDS, EVAL_SEEDS):
        if train_seed == eval_seed:
            raise AssertionError("train and eval seeds must differ")
        train_in, train_tg, _which_train = _load_train_batch(train_seed, 8000)
        eval_in, eval_tg, which_eval = _load_eval_batch(eval_seed, 8000)
        time = int(train_in.shape[1])
        if time != int(eval_in.shape[1]):
            raise AssertionError("train and eval batches differ in length")
        if time > 128:
            raise AssertionError("MQAR sequence exceeds max_seq_len")
        small_cap = int(cells["small"]["archive_cap"])
        base_cap = int(cells["base"]["archive_cap"])
        large_cap = int(cells["large"]["archive_cap"])
        if not (time > small_cap and time > base_cap):
            raise AssertionError(
                "sequence is not longer than the small and base caps, so those rings do not evict"
            )
        if large_cap <= time:
            raise AssertionError(
                "large cap does not exceed this MQAR length; the memo's no-eviction note would be stale"
            )
        models = {name: _build(int(cells[name]["archive_cap"]), train_seed) for name in order}
        _assert_same_init(models)
        befores = {name: _snapshot(model) for name, model in models.items()}
        for name in order:
            spec = cells[name]
            model = models[name]
            train_ce = _train(model, train_in, train_tg, befores[name])
            metrics = _evaluate(model, eval_in, eval_tg, which_eval)
            record = {
                "cell": name,
                "rho": spec["rho"],
                "role": spec["role"],
                "archive_cap": spec["archive_cap"],
                "clipped_to_core": spec["clipped_to_core"],
                "fraction_of_C": spec["fraction_of_C"],
                "ring_evicts": time > int(spec["archive_cap"]),
                "C": C_HORIZON,
                "C_applies_to": "large",
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "train_query_mix": "uniform",
                "eval_query_mix": "exact_key",
                "eval_pi_exact": EVAL_PI_EXACT,
                "eval_sigma": EVAL_SIGMA,
                "eval_source_cap": EVAL_SOURCE_CAP,
                "which": which_eval.tolist(),
                "train_ce": train_ce,
                "ppl": float(math.exp(metrics["eval_ce"])),
                "param_count": count_parameters(model),
                "seq_len": time,
                "device": "cpu",
                "dtype": "float32",
                **metrics,
            }
            runs.append(record)
        del models

    by_cell: dict[str, list[dict]] = {name: [] for name in order}
    for record in runs:
        by_cell[record["cell"]].append(record)
    summaries = []
    for name in order:
        group = by_cell[name]
        if len(group) != len(TRAIN_SEEDS):
            raise AssertionError(f"{name} is missing a CALIB pair")
        kibs = {row["kib"] for row in group}
        if len(kibs) != 1:
            raise AssertionError(f"{name} KiB changed across seeds")
        caps = {row["archive_cap"] for row in group}
        if caps != {cells[name]["archive_cap"]}:
            raise AssertionError(f"{name} archive_cap changed during the run")
        evicts = {row["ring_evicts"] for row in group}
        if len(evicts) != 1:
            raise AssertionError(f"{name} eviction flag changed across seeds")
        eval_ce = [row["eval_ce"] for row in group]
        summaries.append({
            "cell": name,
            "rho": cells[name]["rho"],
            "role": cells[name]["role"],
            "archive_cap": cells[name]["archive_cap"],
            "clipped_to_core": cells[name]["clipped_to_core"],
            "fraction_of_C": cells[name]["fraction_of_C"],
            "ring_evicts": group[0]["ring_evicts"],
            "n_pairs": len(group),
            "next_token_ce": _mean(eval_ce),
            "next_token_ce_sd": _sample_sd(eval_ce),
            "ppl": float(math.exp(_mean(eval_ce))),
            "mse": _mean([row["mse"] for row in group]),
            "hit_rate": _mean([row["hit_rate"] for row in group]),
            "hit_rate_sd": _sample_sd([row["hit_rate"] for row in group]),
            "accepted_rate": _mean([row["accepted_rate"] for row in group]),
            "best_in_ring_cosine_max": max(row["best_in_ring_cosine_max"] for row in group),
            "tau_min": min(row["tau_min"] for row in group),
            "source_in_ring_rate": _mean([row["source_in_ring_rate"] for row in group]),
            "kib": group[0]["kib"],
            "state_nbytes": group[0]["state_nbytes"],
            "per_seed_eval_ce": eval_ce,
            "per_seed_hit_rate": [row["hit_rate"] for row in group],
            "per_seed_accepted_rate": [row["accepted_rate"] for row in group],
            "per_seed_best_in_ring_cosine_max": [row["best_in_ring_cosine_max"] for row in group],
            "per_seed_tau_min": [row["tau_min"] for row in group],
            "per_seed_mse": [row["mse"] for row in group],
        })
    kibs = [row["kib"] for row in summaries]
    summary_caps = [int(row["archive_cap"]) for row in summaries]
    if summary_caps != [8, 32, 96]:
        raise AssertionError(f"summary caps left 8 < 32 < 96: {summary_caps}")
    if not (kibs[0] < kibs[1] < kibs[2]):
        raise AssertionError("KiB did not follow archive_cap small < base < large")
    if [bool(row["ring_evicts"]) for row in summaries] != [True, True, False]:
        raise AssertionError("eviction left the length-47 pattern: small and base evict, large does not")
    if any(row["seq_len"] != 47 for row in runs):
        raise AssertionError("MQAR length left 47; the large-cap eviction note would be stale")
    for train_seed in TRAIN_SEEDS:
        pair = [row for row in runs if row["train_seed"] == train_seed]
        if len(pair) != 3:
            raise AssertionError("a CALIB pair is missing a cell")
        if any(row["which"] != pair[0]["which"] for row in pair):
            raise AssertionError("eval which changed with archive_cap")
        if any(row["eval_pi_exact"] != 1.0 or row["eval_sigma"] != 0.0 for row in pair):
            raise AssertionError("eval mix drifted inside a pair")
        if any(row["train_query_mix"] != "uniform" or row["eval_query_mix"] != "exact_key" for row in pair):
            raise AssertionError("train and eval mixes were swapped")
    health = _critic_health(summaries)
    ce_values = [row["next_token_ce"] for row in summaries]
    summary = {
        "result": "ofat_ok",
        "exit": 0,
        "claim": "calib_ofat_only",
        "not_lm_efficacy": True,
        "not_comprehensive_win": True,
        "device": "cpu",
        "dtype": "float32",
        "torch_num_threads": 1,
        "C": C_HORIZON,
        "C_defined_by_protocol": False,
        "C_applies_to": "large",
        "C_note": (
            "DESIGN.md and EVAL_PROTOCOL_JA.md do not assign an integer to C. "
            "This round sets C = 128 for the large cell only: archive_cap = round(0.75 * 128) = 96. "
            "Base stays the Core point 32 and is not round(0.5 * C). "
            "Small stays the explicit low cell 8. round(0.25 * 128) = 32 would collide with that "
            "Core point, so small is not 0.25 * C. round(0.25 * C) = 8 only at C = 32, which is not "
            "the horizon used for large. An earlier grid used C = 32 and large = 24, which is narrower "
            "than the Core cap, so the factor was not monotonic. The caps here are 8 < 32 < 96."
        ),
        "core_archive_cap": CORE_ARCHIVE_CAP,
        "small_archive_cap": SMALL_ARCHIVE_CAP,
        "large_archive_cap": round(0.75 * C_HORIZON),
        "archive_caps": [8, 32, 96],
        "monotonic_archive_cap": True,
        "small_is_round_quarter_C": False,
        "base_is_half_C": False,
        "large_formula": "round(0.75 * C)",
        "large_clipped_to_core": False,
        "seq_len": runs[0]["seq_len"],
        "eviction_note": (
            "MQAR length is 47. Caps 8 and 32 are shorter than that prefix, so those rings evict. "
            "Cap 96 is wider than the prefix, so the large ring keeps every past position. "
            "KiB still follows the nominal archive_cap."
        ),
        "shape": {"vocab": 8000, "d_model": 256, "n_layer": 6, "n_head": 4, "d_ff": 1024},
        "variant": "recency",
        "archive_policy": "scaffold ring of the last archive_cap positions (continuous keys; not a second Recency-B implementation)",
        "beta_init": BETA_INIT,
        "tau_init": TAU_INIT,
        "tau_relock": False,
        "train_gate": "soft",
        "eval_gate": "hard",
        "train_query_mix": "uniform",
        "eval_query_mix": "exact_key",
        "eval_pi_exact": EVAL_PI_EXACT,
        "eval_sigma": EVAL_SIGMA,
        "eval_source_cap": EVAL_SOURCE_CAP,
        "eval_mix_note": (
            "Train which is uniform over the 20 pairs. CALIB eval which is exact-key: "
            "pi_exact=1, sigma=0, query token equals the source key. Sources are drawn "
            "from pairs inside cap 8 when any exist, and that sequence is shared by "
            "all three cells. The last query is 8 steps after the final pair key, so "
            "cap 8 does not hold it; caps 32 and 96 do. Tau stays 0.75."
        ),
        "adamw": "sda_knobs_only",
        "steps": STEPS,
        "early_stop": False,
        "lr": LR,
        "weight_decay": 0.0,
        "batch_size": BATCH_SIZE,
        "n_pairs": N_PAIRS,
        "n_queries": N_QUERIES,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "seed_band": "CALIB",
        "test_opened": False,
        "holdout_opened": False,
        "lm_band_opened": False,
        "gate": gate,
        "definitions": {
            "next_token_ce": CE_DEFINITION,
            "ppl": "exp(mean over the four CALIB pairs of full-sequence next-token CE)",
            "mse": MSE_DEFINITION,
            "hit_rate": HIT_DEFINITION,
            "accepted_rate": ACCEPTED_DEFINITION,
            "kib": KIB_DEFINITION,
            "source_in_ring_rate": (
                "fraction of MQAR queries whose source key position is still inside the past ring; "
                "geometry of the batch, not a retrieval mask"
            ),
            "best_in_ring_cosine_max": (
                "max over layers, heads, and positions of the best in-ring cosine on that "
                "eval batch, including the current position. Diagnostic for the hard gate. "
                "Not a second accepted_rate"
            ),
        },
        "critic_health": health,
        "rho_effect": health["rho_effect"],
        "ce_interpretable": health["ce_interpretable"],
        "null_effect_claimed": False,
        "follow_up": health["follow_up"],
        "eval_ce_max_abs_diff_across_cells": float(max(ce_values) - min(ce_values)),
        "eval_ce_note": (
            "stored with the grid. Readable as a rho measurement only when ce_interpretable "
            "is true. A shared CE with accepted_rate 0 is UNIDENTIFIABLE, not a null effect"
        ),
        "cells": summaries,
        "command": "python research/sdalm_10m/ablation_rho.py",
    }
    return {"summary": summary, "runs": runs}


def _write(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, separators=(",", ":"), allow_nan=False) for record in payload["runs"]]
    JSONL_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps(payload["summary"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    parsed = [json.loads(line) for line in JSONL_PATH.read_text(encoding="utf-8").splitlines()]
    if len(parsed) != len(payload["runs"]):
        raise AssertionError("jsonl lost a cell")
    summary = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if summary["result"] != "ofat_ok" or summary["exit"] != 0:
        raise AssertionError("summary log was not a finished grid")
    if summary["test_opened"] or summary["holdout_opened"] or summary["lm_band_opened"]:
        raise AssertionError("summary claims a sealed band was opened")
    if [row["cell"] for row in summary["cells"]] != ["small", "base", "large"]:
        raise AssertionError("summary cells left small, base, large order")
    if [row["archive_cap"] for row in summary["cells"]] != [8, 32, 96]:
        raise AssertionError("logged caps are not the monotonic grid 8, 32, 96")
    if summary["C"] != 128 or summary["C_applies_to"] != "large":
        raise AssertionError("logged C is not the large-only horizon 128")
    if summary["small_is_round_quarter_C"] or summary["base_is_half_C"] or summary["large_clipped_to_core"]:
        raise AssertionError("a cap was recorded as a colliding fraction of C")
    if summary["monotonic_archive_cap"] is not True:
        raise AssertionError("log does not mark the caps monotonic")
    if summary["not_lm_efficacy"] is not True or summary["not_comprehensive_win"] is not True:
        raise AssertionError("non-claims were dropped")
    if summary["claim"] != "calib_ofat_only":
        raise AssertionError("claim left calib_ofat_only")
    if summary["eval_pi_exact"] != 1.0 or summary["eval_sigma"] != 0.0:
        raise AssertionError("logged eval mix is not pi_exact=1, sigma=0")
    if summary["eval_source_cap"] != 8 or summary["train_query_mix"] != "uniform":
        raise AssertionError("logged mixes left the locked train/eval split")
    if summary["eval_query_mix"] != "exact_key":
        raise AssertionError("eval mix was not exact_key")
    if summary["tau_relock"] is not False or summary["tau_init"] != 0.75:
        raise AssertionError("tau was re-locked")
    if summary["null_effect_claimed"] is not False:
        raise AssertionError("a null rho effect was claimed")
    health = summary["critic_health"]
    if health["checked_before_ce"] is not True or health["tau_relock"] is not False:
        raise AssertionError("critic health gate was not applied before CE")
    if health["null_effect_claimed"] is not False:
        raise AssertionError("health gate claimed a null effect")
    if health["rho_effect"] != summary["rho_effect"] or health["ce_interpretable"] != summary["ce_interpretable"]:
        raise AssertionError("summary label drifted from the health gate")
    if not health["each_cell_accepted_rate_positive"]:
        if summary["rho_effect"] != "UNIDENTIFIABLE" or summary["ce_interpretable"] is not False:
            raise AssertionError("a closed gate was labeled as a rho result")
        if summary["follow_up"] != "stopped_for_human_review":
            raise AssertionError("closed gate did not stop for human review")
    elif summary["ce_interpretable"] is not True or summary["rho_effect"] != "exploratory_calib":
        raise AssertionError("an open gate was labeled unidentifiable")


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        raise SystemExit("ablation_rho.py takes no arguments; C, caps, and seeds are locked")
    payload = run()
    _write(payload)
    summary = payload["summary"]
    print(
        "ablation_rho device=cpu dtype=float32 "
        f"C={summary['C']} threads=1 steps={STEPS} early_stop=false"
    )
    print(
        "seeds_train=" + ",".join(str(seed) for seed in summary["train_seeds"])
        + " seeds_eval=" + ",".join(str(seed) for seed in summary["eval_seeds"])
    )
    print("gate=PASS adamw_group=sda_knobs_only variant=sda_recency")
    print(
        "archive_cap=8,32,96 monotonic=true C_applies_to=large small=fixed-8 "
        "eval_pi_exact=1 eval_sigma=0 train_query_mix=uniform tau_relock=false"
    )
    health = summary["critic_health"]
    print(
        "critic_health "
        f"each_cell_positive={str(health['each_cell_accepted_rate_positive']).lower()} "
        f"prefer_gate={str(health['prefer_gate']).lower()} "
        f"rho_effect={summary['rho_effect']} "
        f"ce_interpretable={str(summary['ce_interpretable']).lower()} "
        f"null_effect_claimed=false follow_up={summary['follow_up']}"
    )
    for row in summary["cells"]:
        if row["cell"] == "large":
            rho = f"{float(row['rho']):.2f}"
        elif row["cell"] == "base":
            rho = "core"
        else:
            rho = "fixed-8"
        print(
            f"cell={row['cell']} rho={rho} archive_cap={row['archive_cap']} "
            f"accepted_rate={row['accepted_rate']:.6f} "
            f"best_cosine_max={row['best_in_ring_cosine_max']:.6f} "
            f"tau_min={row['tau_min']:.6f} "
            f"source_in_ring_rate={row['source_in_ring_rate']:.6f} kib={row['kib']:.4f}"
        )
        if summary["ce_interpretable"]:
            print(
                f"cell={row['cell']} ce={row['next_token_ce']:.6f} ppl={row['ppl']:.4f} "
                f"mse={row['mse']:.6f} hit_rate={row['hit_rate']:.6f}"
            )
    if not summary["ce_interpretable"]:
        print("ce_not_reported reason=accepted_rate_zero rho_effect=UNIDENTIFIABLE")
    print("log=research/sdalm_10m/logs/ablation_rho_calib.json")
    print("jsonl=research/sdalm_10m/logs/ablation_rho_calib.jsonl")
    print("ofat_ok=true claim=calib_ofat_only")


if __name__ == "__main__":
    main()
