"""Paired CPU microbenchmark of SDA confidence pruning; not an LRA evaluation."""
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from summer_delta_attention import SummerDeltaAttention


CAPACITIES = (1024, 8192)
QUERY_COUNTS = (4, 128)
HIT_FRACTIONS = (0.0, 0.5, 1.0)
SEEDS = tuple(range(5))
REPEATS = 7
DIM, VALUE_DIM = 32, 16


def main():
    print(json.dumps({"event": "config", "variant": "SDA confidence-pruned exact batch reads",
                      "capacities": CAPACITIES, "query_counts": QUERY_COUNTS,
                      "hit_fractions": HIT_FRACTIONS, "seeds": SEEDS, "repeats": REPEATS,
                      "key_dim": DIM, "value_dim": VALUE_DIM, "dtype": "float32",
                      "threshold": 0.75, "beta": 0.05,
                      "timing": "read only; same prepopulated memory; alternating order; amortized calls",
                      "scope": "random-input implementation benchmark, no task accuracy claim",
                      "commit": os.environ.get("ORX_COMMIT"),
                      "python": platform.python_version(), "numpy": np.__version__,
                      "platform": platform.platform(),
                      "threads": {k: os.environ.get(k) for k in
                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}),
          flush=True)
    rows = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            memory = SummerDeltaAttention(DIM, VALUE_DIM, capacity=capacity, dtype=np.float32, seed=seed)
            for key, value in zip(rng.normal(size=(capacity, DIM)),
                                  rng.normal(size=(capacity, VALUE_DIM))):
                memory.write(key, value)
            for count in QUERY_COUNTS:
                for hit_fraction in HIT_FRACTIONS:
                    queries = rng.normal(size=(count, DIM)).astype(np.float32)
                    hits = round(count * hit_fraction)
                    queries[:hits] = memory._keys[rng.choice(capacity, hits, replace=False)]
                    baseline, info = memory.read_batch(queries, prune_rejected=False, details=True)
                    pruned, pruned_info = memory.read_batch(queries, details=True, prune_rejected=True)
                    np.testing.assert_allclose(memory.read_batch(queries, prune_rejected=True),
                                               baseline, atol=3e-6, rtol=3e-6)
                    scalar = np.stack([memory.read(query) for query in queries])
                    np.testing.assert_allclose(pruned, baseline, atol=3e-6, rtol=3e-6)
                    np.testing.assert_allclose(pruned, scalar, atol=3e-6, rtol=3e-6)
                    np.testing.assert_array_equal(info["accepted"], pruned_info["accepted"])
                    times = {"baseline": [], "pruned": []}
                    methods = {"baseline": lambda: memory.read_batch(queries, prune_rejected=False),
                               "pruned": lambda: memory.read_batch(queries, prune_rejected=True)}
                    start = perf_counter()
                    methods["baseline"]()
                    loops = max(1, min(256, int(0.005 / max(perf_counter() - start, 1e-9))))
                    for repeat in range(REPEATS):
                        names = ("baseline", "pruned") if repeat % 2 == 0 else ("pruned", "baseline")
                        for name in names:
                            start = perf_counter()
                            for _ in range(loops):
                                methods[name]()
                            times[name].append((perf_counter() - start) * 1000 / loops)
                    medians = {name: float(np.median(value)) for name, value in times.items()}
                    row = {"capacity": capacity, "queries": count, "hit_fraction": hit_fraction,
                           "seed": seed, "median_ms": medians,
                           "speedup": medians["baseline"] / medians["pruned"],
                           "max_abs_difference": float(np.max(np.abs(pruned - baseline))),
                           "max_scalar_abs_difference": float(np.max(np.abs(pruned - scalar))),
                           "accepted_fraction": float(info["accepted"].mean()),
                           "payload_bytes": memory.bytes_of_state_and_archive,
                           "loops_per_repeat": loops, "times_ms": times}
                    rows.append(row)
                    print(json.dumps({"event": "measurement", **row}), flush=True)
    summaries = []
    for capacity in CAPACITIES:
        for count in QUERY_COUNTS:
            for hit_fraction in HIT_FRACTIONS:
                selected = [r for r in rows if (r["capacity"], r["queries"], r["hit_fraction"])
                            == (capacity, count, hit_fraction)]
                result = {"event": "summary", "capacity": capacity, "queries": count,
                          "hit_fraction": hit_fraction, "seeds": len(selected),
                          "baseline_median_ms": float(np.median([r["median_ms"]["baseline"] for r in selected])),
                          "pruned_median_ms": float(np.median([r["median_ms"]["pruned"] for r in selected])),
                          "median_paired_speedup": float(np.median([r["speedup"] for r in selected])),
                          "min_paired_speedup": min(r["speedup"] for r in selected),
                          "max_paired_speedup": max(r["speedup"] for r in selected),
                          "max_abs_difference": max(r["max_abs_difference"] for r in selected)}
                summaries.append(result)
                print(json.dumps(result), flush=True)
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    (output / "sda_pruning.json").write_text(json.dumps({"rows": rows, "summaries": summaries}, indent=2))
    print(json.dumps({"event": "complete", "rows": len(rows), "correctness": "all comparisons passed"}), flush=True)


if __name__ == "__main__":
    main()
