"""Supplementary figures from research/summary_results.csv (no new experiments).

Usage: OPENBLAS_NUM_THREADS=1 python research/plot_extra.py [--n 2048] [--noise 0.0] [--out research/extra]
Outputs: fig_retention_sensitivity.{png,pdf}, fig_ablation.{png,pdf}
Data: 16-seed synthetic exact-key recall benchmark (not LM training). No latency is plotted: no latency data exists.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

P = Path(__file__).resolve().parent
CURVES = ["SDA v5-Gated (calibrated)", "SDA v5-Gated (original)", "Indexed KV (zero fallback)",
          "Archive softmax (same retained KV)", "KDA channel-gated proxy", "No-decay DeltaNet (calibrated beta)"]

def load(path=P / "summary_results.csv"):
    return pd.read_csv(path)

def retention_sensitivity(S, n, noise, out):
    d = S[(S.n == n) & (S.noise == noise)]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m in CURVES:
        g = d[d.method == m].sort_values("retention")
        if g.empty: continue
        ax.errorbar(g.retention, g.mean_mse, yerr=1.96 * g["sem"], marker="o", ms=4, capsize=2, label=m)
    ax.set_xlabel("archive retention ratio"); ax.set_ylabel("MSE (mean ± 1.96·SEM, 16 seeds)")
    ax.set_title(f"Retention sensitivity (n={n}, noise={noise}, synthetic)")
    ax.legend(fontsize=7); ax.grid(alpha=.2)
    _save(fig, out, "fig_retention_sensitivity")

def ablation(S, n, noise, retention, out):
    d = S[(S.n == n) & (S.noise == noise) & (S.retention == retention) & S.method.isin(CURVES[:4])]
    d = d.set_index("method").loc[[m for m in CURVES[:4] if m in set(d.method)]]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(d.index, d.mean_mse, xerr=1.96 * d["sem"], color="#4C72B0", capsize=3)
    ax.invert_yaxis(); ax.set_xlabel("MSE (lower is better)")
    ax.set_title(f"Ablation at equal retained KV (n={n}, noise={noise}, retention={retention})")
    _save(fig, out, "fig_ablation")

def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight"); fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--n", type=int, default=2048)
    a.add_argument("--noise", type=float, default=0.0); a.add_argument("--retention", type=float, default=0.5)
    a.add_argument("--out", type=Path, default=P / "extra"); x = a.parse_args()
    S = load(); retention_sensitivity(S, x.n, x.noise, x.out); ablation(S, x.n, x.noise, x.retention, x.out)
