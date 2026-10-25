"""Float16 dual-write gate versus a float64 reference. CPU only.

Judgement uses synthetic key/value streams drawn on the fly from the CALIB
seeds 20250001–20250008. It does not read LM/MQAR batches (20300001–).
Using that band is leakage and exits non-zero.

Locked thresholds live at the top of this file. ``--smoke-test-seed`` may
repeat 20261023 as a diagnostic. That run does not edit the thresholds and
does not enter the pass/fail judgement.

Exit 0 is PASS. Any other status is FAIL (or invalid input).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

# Locked judgement thresholds. main() must not assign these.
ATOL = 1e-3
RTOL = 1e-3
OVERWRITE_EPS_FACTOR = 8  # f16 overwrite if cosine >= 1 - FACTOR * finfo(float16).eps
TAU = 0.75
BETA = 0.05
F64_OVERWRITE_TOL = 1e-9

CALIB_SEEDS = tuple(range(20250001, 20250009))
LM_BAND_START = 20300001
SMOKE_ONLY_SEED = 20261023

# On-the-fly stream shape. Not a numerical threshold.
D_KEY = 8
D_VALUE = 4
N_BANK = 6
N_WRITES = 8
CAPACITY = 4

F16_OVERWRITE_TOL = OVERWRITE_EPS_FACTOR * float(np.finfo(np.float16).eps)


def _reject_lm_band(seed: int) -> None:
    if int(seed) >= LM_BAND_START:
        raise ValueError(
            f"seed {seed} is in the LM/MQAR band {LM_BAND_START}–; "
            "using it for this gate is leakage"
        )


def _unit(x: np.ndarray) -> np.ndarray:
    dt = x.dtype
    peak = np.max(np.abs(x))
    if peak == 0:
        raise ValueError("zero vector")
    y = np.asarray(x / peak, dtype=dt)
    norm = np.asarray(np.sqrt(np.sum(y * y, dtype=dt)), dtype=dt)
    return np.asarray(y / norm, dtype=dt)


def make_stream(seed: int) -> dict[str, np.ndarray]:
    """Orthonormal bank, recency writes, and queries. Determined by ``seed`` only."""
    _reject_lm_band(seed)
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(size=(D_KEY, N_BANK + 1))
    q_factor, _ = np.linalg.qr(raw)
    bank = np.asarray(q_factor[:, :N_BANK].T, dtype=np.float64)
    write_ids = rng.integers(0, N_BANK, size=N_WRITES)
    write_ids = np.asarray(write_ids, dtype=np.int64)
    write_ids[1] = write_ids[0]
    write_ids[-1] = write_ids[min(2, N_WRITES - 1)]
    values = np.asarray(rng.normal(size=(N_WRITES, D_VALUE)), dtype=np.float64)
    keys = bank[write_ids]
    extra = np.asarray(q_factor[:, N_BANK], dtype=np.float64)
    query_ids = np.concatenate([np.arange(N_BANK, dtype=np.int64), np.array([-1], dtype=np.int64)])
    query_keys = np.concatenate([bank, extra[None, :]], axis=0)
    return {
        "keys": keys,
        "values": values,
        "write_ids": write_ids,
        "query_keys": query_keys,
        "query_ids": query_ids,
    }


def _state_kib(dtype: np.dtype, archive_n: int) -> float:
    entries = D_KEY * D_VALUE + int(archive_n) * (D_KEY + D_VALUE)
    return entries * np.dtype(dtype).itemsize / 1024.0


def run_path(stream: dict[str, np.ndarray], *, dtype: np.dtype, overwrite_tol: float) -> dict:
    """Recency-B dual write: fast state F and archive K, V, both stored in ``dtype``."""
    dt = np.dtype(dtype)
    keys = np.asarray(stream["keys"], dtype=dt)
    values = np.asarray(stream["values"], dtype=dt)
    query_keys = np.asarray(stream["query_keys"], dtype=dt)
    write_ids = np.asarray(stream["write_ids"])
    query_ids = np.asarray(stream["query_ids"])
    beta = np.asarray(BETA, dtype=dt)
    fast = np.zeros((D_KEY, D_VALUE), dtype=dt)
    arch_k = np.zeros((CAPACITY, D_KEY), dtype=dt)
    arch_v = np.zeros((CAPACITY, D_VALUE), dtype=dt)
    slot_ids = np.full(CAPACITY, -2, dtype=np.int64)
    order: list[int] = []
    size = 0
    for key, value, key_id in zip(keys, values, write_ids):
        k = _unit(np.asarray(key, dtype=dt))
        v = np.asarray(value, dtype=dt)
        pred = np.asarray(k @ fast, dtype=dt)
        err = np.asarray(v - pred, dtype=dt)
        fast = np.asarray(fast + beta * np.outer(k, err), dtype=dt)
        slot = None
        if size:
            scores = np.asarray(arch_k[:size] @ k, dtype=dt)
            nearest = int(np.argmax(scores))
            if float(scores[nearest]) >= 1.0 - overwrite_tol:
                slot = nearest
        if slot is None:
            if size < CAPACITY:
                slot = size
                size += 1
            else:
                slot = order.pop(0)
        else:
            order.remove(slot)
        arch_k[slot] = k
        arch_v[slot] = v
        slot_ids[slot] = int(key_id)
        order.append(slot)
    accept = np.zeros(len(query_ids), dtype=bool)
    hit = np.zeros(len(query_ids), dtype=bool)
    outputs = np.zeros((len(query_ids), D_VALUE), dtype=dt)
    for index, (query, query_id) in enumerate(zip(query_keys, query_ids)):
        q = _unit(np.asarray(query, dtype=dt))
        scores = np.asarray(arch_k[:size] @ q, dtype=dt)
        nearest = int(np.argmax(scores))
        taken = bool(float(scores[nearest]) >= TAU)
        accept[index] = taken
        hit[index] = bool(int(slot_ids[nearest]) == int(query_id))
        pred = np.asarray(q @ fast, dtype=dt)
        out = pred
        if taken:
            corr = np.asarray(arch_v[nearest] - arch_k[nearest] @ fast, dtype=dt)
            out = np.asarray(pred + corr, dtype=dt)
        outputs[index] = out
    return {
        "F": fast,
        "K": arch_k[:size].copy(),
        "V": arch_v[:size].copy(),
        "o": outputs,
        "accept": accept,
        "hit": hit,
        "slot_ids": slot_ids[:size].copy(),
        "archive_n": size,
        "dtype": dt,
    }


def method_metrics(result: dict) -> dict[str, float]:
    """Rates from this path's own masks. Do not copy another method's rates in."""
    return {
        "hit_rate": float(np.mean(result["hit"])),
        "accepted_rate": float(np.mean(result["accept"])),
        "kib": _state_kib(result["dtype"], result["archive_n"]),
    }


def compare(reference: dict, candidate: dict) -> list[str]:
    reasons: list[str] = []
    if not np.array_equal(reference["accept"], candidate["accept"]):
        mismatch = int(np.sum(reference["accept"] != candidate["accept"]))
        reasons.append(f"accept mask mismatch {mismatch}/{len(reference['accept'])}")
    if not np.array_equal(reference["hit"], candidate["hit"]):
        mismatch = int(np.sum(reference["hit"] != candidate["hit"]))
        reasons.append(f"hit mask mismatch {mismatch}/{len(reference['hit'])}")
    for name in ("o", "F", "K", "V"):
        left = np.asarray(reference[name], dtype=np.float64)
        right = np.asarray(candidate[name], dtype=np.float64)
        if left.shape != right.shape:
            reasons.append(f"{name} shape {left.shape} vs {right.shape}")
            continue
        if not np.allclose(left, right, atol=ATOL, rtol=RTOL):
            delta = float(np.max(np.abs(left - right)))
            reasons.append(f"{name} max_abs={delta:.3e} atol={ATOL} rtol={RTOL}")
    return reasons


def _evaluate_seed(seed: int) -> list[str]:
    stream = make_stream(seed)
    reference = run_path(stream, dtype=np.float64, overwrite_tol=F64_OVERWRITE_TOL)
    candidate = run_path(stream, dtype=np.float16, overwrite_tol=F16_OVERWRITE_TOL)
    return compare(reference, candidate)


def _summarize(seed: int) -> dict[str, dict[str, float]]:
    stream = make_stream(seed)
    reference = run_path(stream, dtype=np.float64, overwrite_tol=F64_OVERWRITE_TOL)
    candidate = run_path(stream, dtype=np.float16, overwrite_tol=F16_OVERWRITE_TOL)
    return {
        "f64_reference": method_metrics(reference),
        "f16_dual_write": method_metrics(candidate),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test-seed",
        type=int,
        default=None,
        help=f"optional diagnostic seed (typical {SMOKE_ONLY_SEED}); not used for judgement",
    )
    args = parser.parse_args(argv)
    if args.smoke_test_seed is not None:
        _reject_lm_band(args.smoke_test_seed)

    # Snapshot so a later edit that writes the thresholds fails the run.
    locked = (ATOL, RTOL, OVERWRITE_EPS_FACTOR, TAU, BETA, F64_OVERWRITE_TOL, F16_OVERWRITE_TOL)
    failures: list[str] = []
    per_seed_metrics = []
    for seed in CALIB_SEEDS:
        _reject_lm_band(seed)
        reasons = _evaluate_seed(seed)
        per_seed_metrics.append({"seed": seed, "methods": _summarize(seed)})
        if reasons:
            failures.append(f"seed {seed}: " + "; ".join(reasons))

    smoke_report = None
    if args.smoke_test_seed is not None:
        smoke_reasons = _evaluate_seed(args.smoke_test_seed)
        smoke_report = {
            "seed": args.smoke_test_seed,
            "judgement": False,
            "ok": not smoke_reasons,
            "reasons": smoke_reasons,
        }
        print(
            f"smoke_seed={args.smoke_test_seed} judgement=false "
            f"ok={str(not smoke_reasons).lower()}"
        )

    if locked != (ATOL, RTOL, OVERWRITE_EPS_FACTOR, TAU, BETA, F64_OVERWRITE_TOL, F16_OVERWRITE_TOL):
        failures.append("thresholds changed during the run")

    passed = not failures
    summary = {
        "result": "PASS" if passed else "FAIL",
        "exit": 0 if passed else 1,
        "seeds": list(CALIB_SEEDS),
        "n_seeds": len(CALIB_SEEDS),
        "atol": ATOL,
        "rtol": RTOL,
        "overwrite_eps_factor": OVERWRITE_EPS_FACTOR,
        "f16_overwrite_tol": F16_OVERWRITE_TOL,
        "input": "synthetic_kv_calib_only",
        "failures": failures,
        "smoke": smoke_report,
        "metrics_per_method": per_seed_metrics,
    }
    if passed:
        print("PASS")
    else:
        print("FAIL: " + " | ".join(failures))
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"FAIL: {exc}")
        print(json.dumps({
            "result": "FAIL",
            "exit": 2,
            "reason": str(exc),
            "input": "rejected",
        }, separators=(",", ":")))
        sys.exit(2)
