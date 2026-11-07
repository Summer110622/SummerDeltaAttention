"""Optional scalar-vs-batch latency benchmark; not a language-model evaluation.

Run from the repository root after installation:
    OPENBLAS_NUM_THREADS=1 python -m research.benchmark_reads

Writes/setup are excluded. Both methods read the same already-populated model.
Each timed call includes normalization, search, correction and output allocation.
No timing assertions are made: hardware, BLAS and workload determine speedup.
"""
import argparse
import json
import os
import platform
from time import perf_counter

import numpy as np

from summer_delta_attention import SummerDeltaAttention


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=positive_int, nargs="+", default=[512, 4096])
    parser.add_argument("--queries", type=positive_int, default=256)
    parser.add_argument("--key-dim", type=positive_int, default=64)
    parser.add_argument("--value-dim", type=positive_int, default=64)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--archive-block-size", type=positive_int, default=4096)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    print(json.dumps({"configuration": vars(args), "python": platform.python_version(),
                      "numpy": np.__version__, "platform": platform.platform(),
                      "processor": platform.processor(),
                      "variant": "scalar read versus tiled read_batch",
                      "threads": {name: os.environ.get(name) for name in
                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}),
          flush=True)
    np.show_config()
    np.show_runtime()
    results = []
    for capacity in args.capacity:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            memory = SummerDeltaAttention(args.key_dim, args.value_dim,
                                          capacity=capacity, dtype=args.dtype, seed=seed)
            keys = rng.normal(size=(capacity, args.key_dim))
            values = rng.normal(size=(capacity, args.value_dim))
            for k, v in zip(keys, values):
                memory.write(k, v)
            # Equal counts (up to one query) of exact-key hits and random queries.
            queries = rng.normal(size=(args.queries, args.key_dim))
            queries[::2] = keys[rng.integers(capacity, size=len(queries[::2]))]

            def scalar():
                return np.stack([memory.read(q) for q in queries])

            def batch():
                return memory.read_batch(queries, batch_size=args.batch_size,
                                         archive_block_size=args.archive_block_size)

            expected, actual = scalar(), batch()  # correctness check and warmup
            tolerance = 2e-5 if args.dtype == "float32" else 1e-11
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            times = {"scalar": [], "batch": []}
            methods = [("scalar", scalar), ("batch", batch)]
            for repeat in range(args.repeats):
                for name, method in methods[::1 if repeat % 2 == 0 else -1]:
                    start = perf_counter()
                    method()
                    times[name].append(perf_counter() - start)
            medians = {name: float(np.median(t)) for name, t in times.items()}
            result = {"capacity": capacity, "archive_size": memory.archive_size,
                      "seed": seed, "seconds": times, "median_seconds": medians,
                      "median_speedup": medians["scalar"] / medians["batch"],
                      "max_abs_difference": float(np.max(np.abs(actual - expected)))}
            results.append(result)
            print(json.dumps(result), flush=True)
    for capacity in args.capacity:
        rows = [row for row in results if row["capacity"] == capacity]
        print(json.dumps({"summary": {"capacity": capacity,
                          "queries": args.queries, "seeds": args.seeds,
                          "scalar_median_seconds": float(np.median([
                              row["median_seconds"]["scalar"] for row in rows])),
                          "batch_median_seconds": float(np.median([
                              row["median_seconds"]["batch"] for row in rows])),
                          "median_paired_speedup": float(np.median([
                              row["median_speedup"] for row in rows])),
                          "max_abs_difference": max(row["max_abs_difference"] for row in rows),
                          "correctness": "all comparisons passed"}}), flush=True)


if __name__ == "__main__":
    main()
