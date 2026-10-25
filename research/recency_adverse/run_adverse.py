"""M-D3b recency-adverse stress (spec recency-adverse-stress-v1). Imports eqmem_hard/experiment unmodified.
SDA beta/tau frozen from configuration.json. Expanded beta calibrated on CALIB seeds only (baseline)."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(R))
import eqmem_hard as eh, experiment as ex
from experiment import D, V, CALIB_SEEDS, TEST_SEEDS, make_data, queries_for, state_update
OUT = Path(__file__).resolve().parent
cfg = json.loads((R/'configuration.json').read_text()); BETA, TAU = cfg['sda']['beta'], cfg['sda']['threshold']
def calib_expanded(N, B):
    R1, _ = eh.n_blocks(B); P = eh.proj(R1); res = {}
    for b in [.05, .1, .2, .4, .7, 1.]:
        e = []
        for s in CALIB_SEEDS:
            K, Vv, ids, z, _ = make_data(s, N); S = eh.expanded_update(K, Vv, b, P)
            for sig in [0., .4]: e.append(np.mean((eh.expanded_read(queries_for(K, ids, z, sig), S, P)-Vv[ids])**2))
        res[b] = float(np.mean(e))
    return min(res, key=res.get), res, R1, P
rows = []; meta = {}
for N in [512, 2048]:
    for B in [128, 256]:
        bx, grid, R1, P = calib_expanded(N, B); M = (D*V+B*(D+V))*8
        meta[f'N{N}_B{B}'] = dict(expanded_beta=bx, grid=grid, R_plus_1=int(R1), sda_kib=M/1024, exp_kib=R1*D*V*8/1024)
        for seed in TEST_SEEDS:
            K, Vv, ids, z, perm = make_data(seed, N)
            S = state_update(K, Vv, beta=BETA); Sd = state_update(K, Vv, **cfg['delta']); Sx = eh.expanded_update(K, Vv, bx, P)
            ret = {'recency': np.arange(N-B, N), 'uniform': perm[:B]}
            for sig in [0., .4]:
                q = queries_for(K, ids, z, sig); y = Vv[ids]
                preds = {}
                for rn, idx in ret.items():
                    m, s = eh.archive_read(q, K, Vv, idx); a = s >= TAU
                    preds[f'SDA ({rn} B)'] = (q@S + a[:, None]*(Vv[m]-K[m]@S), m, a, M)
                    preds[f'Indexed KV ({rn} B)'] = (np.where(a[:, None], Vv[m], 0.), m, a, B*(D+V)*8)
                preds['No-decay DeltaNet'] = (q@Sd, None, None, D*V*8)
                preds['Expanded DeltaNet (eq-mem)'] = (eh.expanded_read(q, Sx, P), None, None, R1*D*V*8)
                preds['Zero'] = (np.zeros_like(y), None, None, 0)
                inB_rec = ids >= N-B; inB_uni = np.isin(ids, perm[:B])
                for k, (p, h, a, mem) in preds.items():
                    per = np.mean((p-y)**2, 1); inB = inB_uni if 'uniform' in k else inB_rec
                    rows.append(dict(N=N, B=B, noise=sig, seed=seed, method=k, mse=per.mean(),
                        mse_inB=per[inB].mean() if inB.any() else np.nan, mse_outB=per[~inB].mean() if (~inB).any() else np.nan,
                        frac_outB=1-inB.mean(), mean_age=np.mean(N-1-ids), memory_kib=mem/1024,
                        hit_rate=np.mean(h == ids) if h is not None else np.nan, accept=a.mean() if a is not None else np.nan))
        print('done', N, B, bx, flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT/'raw_results.csv', index=False)
g = ['N', 'B', 'noise', 'method']
agg = df.groupby(g, as_index=False).agg(mean_mse=('mse', 'mean'), sd=('mse', 'std'), mse_inB=('mse_inB', 'mean'), mse_outB=('mse_outB', 'mean'),
    frac_outB=('frac_outB', 'mean'), memory_kib=('memory_kib', 'mean'), hit_rate=('hit_rate', 'mean'), accept=('accept', 'mean'))
agg['sem'] = agg.sd/4; agg.drop(columns='sd').to_csv(OUT/'summary_results.csv', index=False)
pr = []
for (N, B, sig), w in df.groupby(['N', 'B', 'noise']):
    w = w.pivot(index='seed', columns='method', values='mse')
    for a_, b_ in [('SDA (uniform B)', 'SDA (recency B)'), ('Expanded DeltaNet (eq-mem)', 'SDA (recency B)'), ('Expanded DeltaNet (eq-mem)', 'SDA (uniform B)'),
                   ('Indexed KV (recency B)', 'SDA (recency B)'), ('No-decay DeltaNet', 'SDA (recency B)')]:
        dd = w[a_]-w[b_]; se = dd.std()/4
        pr.append(dict(N=N, B=B, noise=sig, delta=f'{a_} − {b_}', mean=dd.mean(), ci_lo=dd.mean()-ex.T95_DF15*se, ci_hi=dd.mean()+ex.T95_DF15*se))
pr = pd.DataFrame(pr); pr.to_csv(OUT/'paired_results.csv', index=False)
(OUT/'configuration.json').write_text(json.dumps(dict(sda_beta=BETA, tau=TAU, cells=meta), indent=1))
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(agg.drop(columns='sd').round(4).to_string(index=False)); print(pr.round(4).to_string(index=False))
