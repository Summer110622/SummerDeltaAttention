"""Rebuild M-L2 face tables from existing research/*/raw_results.csv. No new arch."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
T95 = 2.131449545559323


def paired_ci(d):
    d = np.asarray(d, float)
    m = float(d.mean())
    half = T95 * float(d.std(ddof=1)) / np.sqrt(len(d))
    return m, m - half, m + half


def main():
    raw_v2 = pd.read_csv(ROOT / "eqmem_hard_v2" / "raw_results.csv")
    wide = raw_v2[raw_v2.noise == 0].pivot_table(index="seed", columns="method", values="mse")
    new, exp = "SDA-Recency (new)", "Expanded DeltaNet (eq-mem)"
    m, lo, hi = paired_ci(wide[exp] - wide[new])
    pd.DataFrame(
        [
            dict(
                face="hard",
                N=2048,
                B=1024,
                winner="SDA-Recency",
                mse_winner=float(wide[new].mean()),
                mse_expanded=float(wide[exp].mean()),
                delta_exp_minus_winner=m,
                ci_lo=lo,
                ci_hi=hi,
                memory_kib=388.0,
                source="eqmem_hard_v2",
            )
        ]
    ).to_csv(OUT / "table_hard.csv", index=False)

    raw_h = pd.read_csv(ROOT / "hybrid_rf" / "raw_results.csv")
    iid = raw_h[(raw_h.suite == "iid") & (raw_h.noise == 0)]
    iid_rows = []
    for (N, B), g in iid.groupby(["N", "B"]):
        w = g.pivot_table(index="seed", columns="method", values="mse")
        m, lo, hi = paired_ci(w["Expanded DeltaNet (eq-mem)"] - w["SDA-Recency"])
        iid_rows.append(
            dict(
                face="iid",
                N=int(N),
                B=int(B),
                winner="Expanded DeltaNet (eq-mem)",
                mse_winner=float(w["Expanded DeltaNet (eq-mem)"].mean()),
                mse_sda_recency=float(w["SDA-Recency"].mean()),
                delta_exp_minus_recency=m,
                ci_lo=lo,
                ci_hi=hi,
                source="hybrid_rf",
            )
        )
    pd.DataFrame(iid_rows).to_csv(OUT / "table_iid.csv", index=False)

    raw_a = pd.read_csv(ROOT / "recency_adverse_v2" / "raw_results.csv")
    a0 = raw_a[raw_a.noise == 0]
    adv_rows = []
    for (N, B), g in a0.groupby(["N", "B"]):
        w = g.pivot_table(index="seed", columns="method", values="mse")
        m, lo, hi = paired_ci(w["Expanded DeltaNet (eq-mem)"] - w["SDA (recency B)"])
        adv_rows.append(
            dict(
                face="adverse_v2",
                N=int(N),
                B=int(B),
                winner="Expanded DeltaNet (eq-mem)",
                mse_winner=float(w["Expanded DeltaNet (eq-mem)"].mean()),
                mse_sda_recency=float(w["SDA (recency B)"].mean()),
                mse_sda_uniform=float(w["SDA (uniform B)"].mean()),
                delta_exp_minus_recency=m,
                ci_lo=lo,
                ci_hi=hi,
                source="recency_adverse_v2",
            )
        )
    pd.DataFrame(adv_rows).to_csv(OUT / "table_adverse_v2.csv", index=False)
    print("wrote", OUT / "table_hard.csv", OUT / "table_iid.csv", OUT / "table_adverse_v2.csv")


if __name__ == "__main__":
    main()
