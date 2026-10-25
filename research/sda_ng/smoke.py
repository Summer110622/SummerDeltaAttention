"""CALIB-only CPU smoke for SDA-NG on the reconstructed hard face.

Exact-key queries (σ=0). Compares the frozen soft gate with the hard-gate
SDA-Recency baseline on the same seeds. Does not run TEST seeds.

Exit code 2 if frozen-τ mean_g is not strictly positive (gate stuck shut).
"""
from __future__ import annotations

import json
import os
import time

# Must precede NumPy's OpenBLAS load.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

import numpy as np

from hard_face import make_hard_data
from reference import (
    BETA,
    QUANTILE_P,
    TAU_FROZEN,
    TEMPERATURE,
    T_EPS,
    RecencySDA,
    calib_tau,
    recency_positions,
)

# Same split as research/experiment.py. TEST is listed only so the smoke can
# refuse it. Do not iterate TEST_SEEDS.
CALIB_SEEDS = [20250001 + i for i in range(8)]
TEST_SEEDS = [20261023 + i for i in range(16)]
HIT_MSE = 1e-3
OUT = Path(__file__).resolve().parent / "smoke_log.json"


def _replay(data) -> RecencySDA:
    memory = RecencySDA(
        data["D"], data["V"], capacity=data["B"], beta=BETA,
        tau=TAU_FROZEN, temperature=TEMPERATURE, eps=T_EPS,
    )
    for key_id, key, value in zip(data["key_ids"], data["keys"], data["values"]):
        memory.write(int(key_id), key, value)
    positions = recency_positions(data["key_ids"], data["B"])
    expected = [int(data["key_ids"][p]) for p in positions.tolist()]
    if memory.archived_ids() != expected:
        raise RuntimeError(f"Recency-B diverged from recency_positions on seed {data['seed']}")
    return memory


def _eval(memory: RecencySDA, data, *, mode: str, tau: float) -> dict:
    out, info = memory.read(data["query_keys"], mode=mode, tau=tau, details=True)
    target = data["targets"]
    per = np.mean((out - target) ** 2, axis=1)
    cos = np.asarray(info["cosine"], dtype=np.float64)
    weight = np.asarray(info["weight"], dtype=np.float64)
    kind = data["query_kind"]
    archived = set(memory.archived_ids())
    in_archive = np.asarray([int(i) in archived for i in data["query_ids"]], dtype=bool)
    below = cos < tau
    rewrite = kind == "rewrite"

    def _mean(mask, values):
        if not np.any(mask):
            return None
        return float(np.mean(values[mask]))

    return {
        "mse": float(np.mean(per)),
        "mean_g": float(np.mean(weight)),
        "frac_g_gt_0.05": float(np.mean(weight > 0.05)),
        "mean_g_below_tau": _mean(below, weight),
        "frac_g_gt_0.05_below_tau": _mean(below, weight > 0.05),
        "n_below_tau": int(np.sum(below)),
        "mean_g_in_archive": _mean(in_archive, weight),
        "mean_g_miss": _mean(~in_archive, weight),
        "mse_in_archive": _mean(in_archive, per),
        "mse_miss": _mean(~in_archive, per),
        "rewrite_hit": float(np.mean(per[rewrite] < HIT_MSE)),
        "once_hit": float(np.mean(per[~rewrite] < HIT_MSE)),
        "frac_in_archive": float(np.mean(in_archive)),
        "mean_cos": float(np.mean(cos)),
        "max_cos": float(np.max(cos)),
        "min_cos": float(np.min(cos)),
        "cosine": cos,
        "weight": weight,
    }


def _pool(rows: list[dict], key: str):
    vals = [row[key] for row in rows if row[key] is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _pack(rows: list[dict], *, tau: float, label: str) -> dict:
    return {
        "label": label,
        "tau": tau,
        "mean_g": _pool(rows, "mean_g"),
        "frac_g_gt_0.05": _pool(rows, "frac_g_gt_0.05"),
        "mean_g_below_tau": _pool(rows, "mean_g_below_tau"),
        "frac_g_gt_0.05_below_tau": _pool(rows, "frac_g_gt_0.05_below_tau"),
        "mean_g_in_archive": _pool(rows, "mean_g_in_archive"),
        "mean_g_miss": _pool(rows, "mean_g_miss"),
        "mse": _pool(rows, "mse"),
        "mse_in_archive": _pool(rows, "mse_in_archive"),
        "mse_miss": _pool(rows, "mse_miss"),
        "rewrite_hit_mse_lt_1e-3": _pool(rows, "rewrite_hit"),
        "once_hit_mse_lt_1e-3": _pool(rows, "once_hit"),
        "frac_query_in_archive": _pool(rows, "frac_in_archive"),
        "mean_cos": _pool(rows, "mean_cos"),
        "n_below_tau_per_seed_mean": _pool(rows, "n_below_tau"),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> int:
    if set(CALIB_SEEDS) & set(TEST_SEEDS):
        raise RuntimeError("CALIB and TEST seeds overlap")
    t0 = time.perf_counter()
    per_seed = []
    soft_rows = []
    hard_rows = []
    all_cos = []
    cached = []
    for seed in CALIB_SEEDS:
        data = make_hard_data(seed)
        memory = _replay(data)
        soft = _eval(memory, data, mode="soft", tau=TAU_FROZEN)
        hard = _eval(memory, data, mode="hard", tau=TAU_FROZEN)
        if not np.allclose(soft["cosine"], hard["cosine"]):
            raise RuntimeError("soft and hard scans disagreed on cosine")
        all_cos.append(soft["cosine"])
        cached.append((data, memory, soft, hard))
        soft_rows.append(soft)
        hard_rows.append(hard)

    cos_pool = np.concatenate(all_cos)
    tau_q = calib_tau(cos_pool, QUANTILE_P)
    q_soft_rows = []
    q_hard_rows = []
    for data, memory, soft, hard in cached:
        q_soft = _eval(memory, data, mode="soft", tau=tau_q)
        q_hard = _eval(memory, data, mode="hard", tau=tau_q)
        q_soft_rows.append(q_soft)
        q_hard_rows.append(q_hard)
        per_seed.append({
            "seed": int(data["seed"]),
            "soft_frozen": {k: v for k, v in soft.items() if k not in ("cosine", "weight")},
            "hard_frozen": {k: v for k, v in hard.items() if k not in ("cosine", "weight")},
            "soft_quantile": {k: v for k, v in q_soft.items() if k not in ("cosine", "weight")},
            "hard_quantile": {k: v for k, v in q_hard.items() if k not in ("cosine", "weight")},
            "mse_soft_minus_hard_frozen": float(soft["mse"] - hard["mse"]),
        })

    soft_pack = _pack(soft_rows, tau=TAU_FROZEN, label="soft g=sigmoid((cos-tau)/T)")
    hard_pack = _pack(hard_rows, tau=TAU_FROZEN, label="hard g=1[cos>=tau] SDA-Recency baseline")
    mean_g = soft_pack["mean_g"]
    frac = soft_pack["frac_g_gt_0.05"]
    if mean_g is None or not np.isfinite(mean_g) or mean_g <= 0.0:
        status = "fail_gate_stuck_shut"
        code = 2
    else:
        status = "ok" if (frac or 0.0) > 0.0 else "ok_mean_g_but_frac_gt_0.05_is_zero"
        code = 0

    deltas = [row["mse_soft_minus_hard_frozen"] for row in per_seed]
    log = {
        "name": "SDA-NG",
        "status": status,
        "pr_opened": False,
        "test_seeds_ran": False,
        "claim_boundary": (
            "Synthetic reconstructed hard face, CALIB exact-key queries only. "
            "Not an LM result. Not a reproduction of eqmem_hard_v2 TEST MSE 0.575. "
            "Not evidence of a universal core."
        ),
        "formula": {
            "cos_j": "max_i <q, k_i>",
            "T": "max(temperature, eps)",
            "g": "sigmoid((cos_j - tau) / T)",
            "out": "F.T @ q + g * (v_j - F.T @ k_j)",
            "sigmoid": "1 / (1 + exp(-x))",
        },
        "frozen": {
            "beta": BETA,
            "tau": TAU_FROZEN,
            "temperature": TEMPERATURE,
            "eps": T_EPS,
            "quantile_p": QUANTILE_P,
            "hard_accept_target_of_quantile": 1.0 - QUANTILE_P,
        },
        "calib_quantile": {
            "tau": tau_q,
            "fit_on": "pooled top-1 cosines of these same CALIB queries (in-sample diagnostic)",
            "numpy_quantile": "linear (NumPy default)",
            "note": (
                "On this exact-key face a point mass at cos=1 can put the p=0.3 "
                "quantile on that mass, so the hard-accept target of ~0.7 need not "
                "be realized. Reported as observed, not as a tuned success."
            ),
        },
        "data": {
            "face": "hard rewrite+distractor, reconstructed; eqmem_hard.py absent on this branch",
            "sigma": 0.0,
            "queries": "exact key q=k",
            "n": 2048,
            "U": 1740,
            "B": 1024,
            "D": 32,
            "V": 16,
            "Q_per_seed": 128,
            "query_split": "64 rewrite + 64 once-only",
            "seeds": CALIB_SEEDS,
            "n_seeds": len(CALIB_SEEDS),
        },
        "soft_frozen_tau": soft_pack,
        "hard_frozen_tau": hard_pack,
        "mse_soft_minus_hard_frozen": {
            "mean": float(np.mean(deltas)),
            "std_ddof1": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "per_seed": deltas,
            "reading": "Positive means the soft gate has higher MSE than the hard gate on this CALIB set.",
        },
        "soft_calib_quantile_tau": _pack(q_soft_rows, tau=tau_q, label="soft at in-sample CALIB quantile tau"),
        "hard_calib_quantile_tau": _pack(q_hard_rows, tau=tau_q, label="hard at in-sample CALIB quantile tau"),
        "per_seed": per_seed,
        "environment": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "numpy": np.__version__,
            "seconds": time.perf_counter() - t0,
        },
    }
    OUT.write_text(json.dumps(_jsonable(log), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "exit": code,
        "tau": TAU_FROZEN,
        "temperature": TEMPERATURE,
        "mean_g": soft_pack["mean_g"],
        "frac_g_gt_0.05": soft_pack["frac_g_gt_0.05"],
        "mean_g_below_tau": soft_pack["mean_g_below_tau"],
        "mse_soft": soft_pack["mse"],
        "mse_hard": hard_pack["mse"],
        "mse_soft_minus_hard": float(np.mean(deltas)),
        "tau_quantile": tau_q,
        "log": str(OUT),
    }, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
