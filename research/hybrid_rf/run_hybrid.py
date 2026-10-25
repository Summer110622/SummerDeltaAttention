"""SDA-Hybrid-RF eval: (hard) eqmem_hard n=2048 rho=.5 ; (iid) uniform past ; (old) v2 old-skew pi=0. eqmem_hard.py unmodified.
Hybrid beta,tau calibrated jointly on CALIB seeds per suite/cell. SDA-Recency beta/tau frozen (0.05,0.75). Expanded beta CALIB-calibrated."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
R0 = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(R0)); sys.path.insert(0, str(R0/'hybrid_rf'))
import eqmem_hard as eh, experiment as ex, hybrid as hy, data_iid
from experiment import D, V, CALIB_SEEDS, TEST_SEEDS, queries_for, state_update
OUT = Path(__file__).resolve().parent; cfg = json.loads((R0/'configuration.json').read_text())
BETA, TAU = cfg['sda']['beta'], cfg['sda']['threshold']; SIGS = [0., .4]
def expanded_beta(sets, P):
    res = {b: np.mean([np.mean((eh.expanded_read(q, eh.expanded_update(K, Vv, b, P), P)-y)**2) for K, Vv, _, qs in sets for q, y in qs]) for b in hy.BETAS}
    return min(res, key=res.get)
def suite_sets(suite, seeds, N, B):
    out = []
    for s in seeds:
        if suite == 'hard':
            d = eh.make_hard_data(s, N); K, Vv = d['keys'], d['values']
            qs = [(eh.hard_queries(d, g), Vv[d['tgt_pos']]) for g in SIGS]
            out.append((K, Vv, (lambda P, kid=d['key_id']: eh.recency_positions(kid, P)), qs, d))
        else:
            K, Vv, ids, z, perm = data_iid.data(s, N, B, 'old' if suite == 'old' else 'uniform')
            qs = [(queries_for(K, ids, z, g), Vv[ids]) for g in SIGS]
            out.append((K, Vv, (lambda P, N=N: np.arange(N-P, N)), qs, dict(ids=ids)))
    return out
rows, meta = [], {}
cells = [('hard', 2048, 1024)] + [(s, N, B) for s in ['iid', 'old'] for N in [512, 2048] for B in [128, 256]]
for suite, N, B in cells:
    t0 = time.time(); R1, M = eh.n_blocks(B); Mf = M//8; Px = eh.proj(R1)
    Rh = hy.R_default(R1); Ph = hy.split(Mf, Rh); Pr = hy.proj(Rh)
    cal = suite_sets(suite, CALIB_SEEDS, N, B)
    bx = expanded_beta([c[:4] for c in cal], Px)
    (bh, th), grid = hy.calibrate([c[:4] for c in cal], Rh, Ph)
    hk = (Rh*D*V + D*Rh + Ph*(D+V))*8/1024
    meta[f'{suite}_N{N}_B{B}'] = dict(R_exp=int(R1), exp_beta=bx, R_hyb=Rh, P_hyb=Ph, hyb_kib=hk, exp_kib=R1*D*V*8/1024, hyb_beta=bh, hyb_tau=th)
    for seed, (K, Vv, pf, qs, d) in zip(TEST_SEEDS, suite_sets(suite, TEST_SEEDS, N, B)):
        S = state_update(K, Vv, beta=BETA); Sx = eh.expanded_update(K, Vv, bx, Px); Sh = hy.update(K, Vv, bh, Pr)
        recB = pf(B); poolH = pf(Ph)
        for g, (q, y) in zip(SIGS, qs):
            mr, sr = eh.archive_read(q, K, Vv, recB); ar = sr >= TAU
            hr, _, gh = hy.read(q, K, Vv, Sh, Pr, poolH, th)
            hs, _, _ = hy.read(q, K, Vv, Sh, Pr, poolH, th, mode='switch')
            preds = {'Hybrid-RF': hr, 'Hybrid-RF (switch abl.)': hs, 'SDA-Recency': q@S + ar[:, None]*(Vv[mr]-K[mr]@S),
                     'Expanded DeltaNet (eq-mem)': eh.expanded_read(q, Sx, Px)}
            if suite == 'hard': tp = d['tgt_pos']; inP = np.isin(tp, poolH); seg = {'rewrite': d['is_rw']}
            else: tp = d['ids']; inP = np.isin(tp, poolH); seg = {}
            for k, p in preds.items():
                per = np.mean((p-y)**2, 1); ok = per < 1e-3
                r = dict(suite=suite, N=N, B=B, noise=g, seed=seed, method=k, mse=per.mean(),
                         mse_inPool=per[inP].mean() if inP.any() else np.nan, mse_outPool=per[~inP].mean() if (~inP).any() else np.nan,
                         frac_inPool=inP.mean(), hyb_accept=gh.mean())
                if suite == 'hard': r.update(rewrite_hit=ok[d['is_rw']].mean(), stable_hit=ok[~d['is_rw']].mean())
                rows.append(r)
    print('done', suite, N, B, meta[f'{suite}_N{N}_B{B}'], round(time.time()-t0, 1), flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT/'raw_results.csv', index=False)
g = ['suite', 'N', 'B', 'noise', 'method']
agg = df.groupby(g, as_index=False).mean(numeric_only=True).drop(columns='seed'); agg.to_csv(OUT/'summary_results.csv', index=False)
pr = []
for (s, N, B, sig), w in df.groupby(['suite', 'N', 'B', 'noise']):
    w = w.pivot(index='seed', columns='method', values='mse')
    for a_ in ['Expanded DeltaNet (eq-mem)', 'SDA-Recency', 'Hybrid-RF (switch abl.)']:
        dd = w[a_]-w['Hybrid-RF']; se = dd.std()/4
        pr.append(dict(suite=s, N=N, B=B, noise=sig, delta=f'{a_} − Hybrid', mean=dd.mean(), ci_lo=dd.mean()-ex.T95_DF15*se, ci_hi=dd.mean()+ex.T95_DF15*se))
pr = pd.DataFrame(pr); pr.to_csv(OUT/'paired_results.csv', index=False)
(OUT/'configuration.json').write_text(json.dumps(dict(sda_beta=BETA, sda_tau=TAU, cells=meta), indent=1, default=float))
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(agg.round(4).to_string(index=False)); print(pr.round(4).to_string(index=False))
