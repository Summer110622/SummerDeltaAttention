"""eqmem_hard_v2: SDA-Recency candidate vs Expanded/Recency/Indexed/SDA-reservoir. Imports eqmem_hard (unmodified)."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(R))
import eqmem_hard as eh, experiment as ex
from experiment import D, V, TEST_SEEDS, state_update, mse
OUT = Path(__file__).resolve().parent
NEW = 'SDA-Recency (new)'
def run(n=2048, rho=.5, noises=(0., .4)):
    t0 = time.time(); cfg = json.loads((R/'configuration.json').read_text())
    B = round(rho*n); R1, M = eh.n_blocks(B); P = eh.proj(R1)
    bx, grid = eh.calibrate_expanded(n, rho); tau = cfg['sda']['threshold']
    rows = []
    for seed in TEST_SEEDS:
        d = eh.make_hard_data(seed, n); K, Vv = d['keys'], d['values']
        S = state_update(K, Vv, beta=cfg['sda']['beta']); Sx = eh.expanded_update(K, Vv, bx, P)
        res = d['perm'][:B]; rec = eh.recency_positions(d['key_id'], B)
        for sig in noises:
            q = eh.hard_queries(d, sig); y = Vv[d['tgt_pos']]
            m, s = eh.archive_read(q, K, Vv, res); a = s >= tau
            mr, sr = eh.archive_read(q, K, Vv, rec); ar = sr >= tau
            preds = {NEW: (q@S + ar[:, None]*(Vv[mr]-K[mr]@S), mr, ar, M),
                     eh.SDA: (q@S + a[:, None]*(Vv[m]-K[m]@S), m, a, M),
                     'Indexed KV (reservoir)': (np.where(a[:, None], Vv[m], 0.), m, a, B*(D+V)*8),
                     'Expanded DeltaNet (eq-mem)': (eh.expanded_read(q, Sx, P), None, None, R1*D*V*8),
                     'Recency KV (eq-B)': (Vv[mr], mr, None, B*(D+V)*8)}
            for k, (p, h, acc, mem) in preds.items():
                per = np.mean((p-y)**2, 1); ok = per < 1e-3
                rows.append(dict(noise=sig, seed=seed, method=k, mse=per.mean(), memory_kib=mem/1024,
                    hit_rate=np.mean(h == d['tgt_pos']) if h is not None else np.nan,
                    accept=acc.mean() if acc is not None else np.nan,
                    rewrite_hit=ok[d['is_rw']].mean(), stable_hit=ok[~d['is_rw']].mean()))
    df = pd.DataFrame(rows); df.to_csv(OUT/'raw_results.csv', index=False)
    agg = df.groupby(['noise', 'method'], as_index=False).agg(mean_mse=('mse', 'mean'), sd=('mse', 'std'),
        memory_kib=('memory_kib', 'mean'), hit_rate=('hit_rate', 'mean'), accept=('accept', 'mean'),
        rewrite_hit=('rewrite_hit', 'mean'), stable_hit=('stable_hit', 'mean'))
    agg['sem'] = agg.sd/np.sqrt(len(TEST_SEEDS)); agg.to_csv(OUT/'summary_results.csv', index=False)
    pr = []
    for sig in noises:
        w = df[df.noise == sig].pivot(index='seed', columns='method', values='mse')
        for ref in ['Expanded DeltaNet (eq-mem)', 'Recency KV (eq-B)', 'Indexed KV (reservoir)', eh.SDA]:
            dd = w[ref]-w[NEW]; se = dd.std()/np.sqrt(len(dd))
            pr.append(dict(noise=sig, delta=f'{ref} − new', mean=dd.mean(), ci_lo=dd.mean()-ex.T95_DF15*se, ci_hi=dd.mean()+ex.T95_DF15*se))
    pr = pd.DataFrame(pr); pr.to_csv(OUT/'paired_results.csv', index=False)
    (OUT/'configuration.json').write_text(json.dumps(dict(n=n, rho=rho, B=B, R_plus_1=int(R1), expanded_beta=bx, grid=grid, committed=cfg), indent=1, default=str))
    pd.set_option('display.width', 250)
    print('beta_x', bx); print(agg.drop(columns='sd').to_string(index=False)); print(pr.to_string(index=False)); print('time', time.time()-t0)
if __name__ == '__main__': run()
