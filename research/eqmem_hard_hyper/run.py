"""eqmem_hard re-Go for SDA-Hyper-EqMem (Arch 案H). Imports research/eqmem_hard.py (unchanged).
Usage: OPENBLAS_NUM_THREADS=1 .venv/bin/python research/eqmem_hard_hyper/run.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, '/workspace/sda-lab/sda_hyper')
import eqmem_hard as eh, experiment as ex
from experiment import D, V, CALIB_SEEDS, TEST_SEEDS, state_update, T95_DF15
from sda_hyper.hyper_eqmem import SDAHyperEqMem

N, RHO = 2048, .5; B = round(RHO*N); NOISES = [0., .4]
VARIANTS = {'HyperEqMem (spec P990 L3 b7 C4)': dict(P=990, L=3, b=7, C=4),
            'HyperEqMem (C16 P946, ablation)': dict(P=946, L=3, b=7, C=16)}

def build(d, beta, cfgv):
    m = SDAHyperEqMem(D, V, beta=beta, **cfgv)
    for k, v in zip(d['keys'], d['values']): m.write(k, v)
    return m

def predict(m, q, tau, mg):
    out = []; hits = []
    for x in q:
        p, j, _, _ = m.read(x, tau, mg); out.append(p); hits.append(j)
    return np.array(out), hits

def calibrate(cfgv):
    res = {}
    datas = [eh.make_hard_data(s, N) for s in CALIB_SEEDS]
    for beta in [0., .05, .2]:
        ms = [build(d, beta, cfgv) for d in datas]
        for tau in [.5, .65, .75, .85]:
            for mg in [0., .02, .05, .1]:
                e = [ex.mse(predict(m, eh.hard_queries(d, s), tau, mg)[0], d['values'][d['tgt_pos']])
                     for m, d in zip(ms, datas) for s in NOISES]
                res[(beta, tau, mg)] = float(np.mean(e))
    best = min(res, key=res.get); return best, res

def main():
    t0 = time.time()
    base = json.loads((HERE.parent/'eqmem_hard'/'configuration.json').read_text())
    cfg = base['committed']; beta_x = base['expanded_beta']; R1 = base['R_plus_1']; P = eh.proj(R1)
    cal = {}
    for name, v in VARIANTS.items():
        best, grid = calibrate(v); cal[name] = best; print(name, 'calib', best, grid[best])
    rows = []
    for seed in TEST_SEEDS:
        d = eh.make_hard_data(seed, N); K, Vv = d['keys'], d['values']
        S_sda = state_update(K, Vv, beta=cfg['sda']['beta']); S_x = eh.expanded_update(K, Vv, beta_x, P)
        chosen = d['perm'][:B]; rec = eh.recency_positions(d['key_id'], B); tau0 = cfg['sda']['threshold']
        hms = {nm: build(d, cal[nm][0], v) for nm, v in VARIANTS.items()}
        for sig in NOISES:
            q = eh.hard_queries(d, sig); y = Vv[d['tgt_pos']]
            mi, s = eh.archive_read(q, K, Vv, chosen); acc = s >= tau0
            mr, _ = eh.archive_read(q, K, Vv, rec)
            preds = {eh.SDA: (q@S_sda + acc[:, None]*(Vv[mi]-K[mi]@S_sda), (D*V+B*(D+V))*8),
                     'Expanded DeltaNet (eq-mem)': (eh.expanded_read(q, S_x, P), R1*D*V*8),
                     'Recency KV (eq-B)': (Vv[mr], B*(D+V)*8)}
            for nm, m in hms.items():
                preds[nm] = (predict(m, q, cal[nm][1], cal[nm][2])[0], m.memory_bytes())
            for meth, (p, mem) in preds.items():
                per = np.mean((p-y)**2, 1); ok = per < 1e-3
                rows.append(dict(n=N, noise=sig, retention=RHO, seed=seed, method=meth, mse=float(per.mean()),
                                 memory_kib=mem/1024, rewrite_hit=float(ok[d['is_rw']].mean()),
                                 stable_hit=float(ok[~d['is_rw']].mean()), rewrite_mse=float(per[d['is_rw']].mean()),
                                 stable_mse=float(per[~d['is_rw']].mean())))
    df = pd.DataFrame(rows); df.to_csv(HERE/'raw_results.csv', index=False)
    agg = df.groupby(['noise', 'method'], as_index=False).agg(mean_mse=('mse', 'mean'), sd=('mse', 'std'),
          memory_kib=('memory_kib', 'mean'), rewrite_hit=('rewrite_hit', 'mean'), stable_hit=('stable_hit', 'mean'),
          rewrite_mse=('rewrite_mse', 'mean'), stable_mse=('stable_mse', 'mean'))
    agg['sem'] = agg.sd/4; agg.to_csv(HERE/'summary_results.csv', index=False)
    pr = []
    for sig, g in df.groupby('noise'):
        w = g.pivot(index='seed', columns='method', values='mse')
        for nm in VARIANTS:
            for other in ['Expanded DeltaNet (eq-mem)', 'Recency KV (eq-B)', eh.SDA]:
                dd = (w[other]-w[nm]).to_numpy(); h = T95_DF15*dd.std(ddof=1)/4
                pr.append(dict(noise=sig, candidate=nm, comparator=other, mean_diff=dd.mean(), ci_low=dd.mean()-h, ci_high=dd.mean()+h))
    pr = pd.DataFrame(pr); pr.to_csv(HERE/'paired_results.csv', index=False)
    (HERE/'configuration.json').write_text(json.dumps(dict(variants=VARIANTS, calib={k: list(v) for k, v in cal.items()},
        expanded_beta=beta_x, R_plus_1=R1), indent=2))
    pd.set_option('display.width', 250)
    print(agg.to_string(index=False)); print(pr.to_string(index=False)); print('time', time.time()-t0)

if __name__ == '__main__': main()
