"""SDA-Cond2 (M-D3e): Routed multi-F + LRU exact pool, write-statistic 2-mode switch (no residual).
rho_rw = mean(is_rw over last W writes), is_rw = key_id seen earlier in sequence. Read: rho>=theta -> Exact if cos>=tau_R else F; else Exact if cos>=tau_E else F.
Budget: hard = 388 KiB with R=64,P=309 (spec). iid/old cells: Expanded's per-cell budget, R=round(64/97*R_exp), rest to P (same scaling rule as hybrid_rf).
ONE global config chosen on CALIB_SEEDS (all faces & cells pooled, sigma 0 and 0.4, no face labels). TEST only for reporting."""
import sys, json, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
R0 = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(R0)); sys.path.insert(0, str(R0/'hybrid_rf'))
import eqmem_hard as eh, experiment as ex, hybrid as hy, data_iid
from experiment import D, V, CALIB_SEEDS, TEST_SEEDS, queries_for, state_update
OUT = Path(__file__).resolve().parent; cfg = json.loads((R0/'configuration.json').read_text())
BETA, TAU = cfg['sda']['beta'], cfg['sda']['threshold']; SIGS = [0., .4]
G_B = [.2, .4, .7]; G_TH = [.05, .1, .2]; G_TR = [.5, .7, .85]; G_W = [128, 256]
def tauE_opts(tr): return sorted({tr, .9, .95})
def rho(kid, W):
    seen, rw = set(), []
    for k in kid: rw.append(k in seen); seen.add(k)
    return float(np.mean(rw[-W:]))
def sets(suite, seeds, N, B):
    out = []
    for s in seeds:
        if suite == 'hard':
            d = eh.make_hard_data(s, N); K, Vv, kid = d['keys'], d['values'], d['key_id']
            qs = [(eh.hard_queries(d, g), Vv[d['tgt_pos']]) for g in SIGS]; pf = lambda P, kid=kid: eh.recency_positions(kid, P); tp = d['tgt_pos']
        else:
            K, Vv, ids, z, perm = data_iid.data(s, N, B, 'old' if suite == 'old' else 'uniform'); kid = np.arange(N)
            qs = [(queries_for(K, ids, z, g), Vv[ids]) for g in SIGS]; pf = lambda P, N=N: np.arange(N-P, N); tp = ids; d = {}
        out.append(dict(K=K, V=Vv, kid=kid, qs=qs, pf=pf, tp=tp, d=d))
    return out
def cell_geom(suite, B):
    R1, M = eh.n_blocks(B)
    if suite == 'hard': Rc, Pc = 64, 309
    else: Rc = max(1, round(64/97*R1)); Pc = hy.split(M//8, Rc)
    return R1, M, Rc, Pc
def prep(st, beta, Pr, P):
    """precompute f, pool candidate value, cos for each sigma."""
    S = hy.update(st['K'], st['V'], beta, Pr); pool = np.sort(st['pf'](P))[::-1]; out = []
    for q, y in st['qs']:
        f = np.einsum('qd,qdv->qv', q, S[hy.route(q, Pr)]); s = q@st['K'][pool].T; c = np.argmax(s, 1)
        j = pool[c]; out.append(dict(f=f, vj=st['V'][j], c=s[np.arange(len(q)), c], y=y, inP=np.isin(st['tp'], pool)))
    return out
def cond2(p, r, th, tr, te):
    t = tr if r >= th else te; g = p['c'] >= t
    return np.where(g[:, None], p['vj'], p['f']), g, r >= th
cells = [('hard', 2048, 1024)] + [(s, N, B) for s in ['iid', 'old'] for N in [512, 2048] for B in [128, 256]]
# ---- CALIB (global) ----
t0 = time.time(); cal = {}
for suite, N, B in cells:
    R1, M, Rc, Pc = cell_geom(suite, B); Pr = hy.proj(Rc); ss = sets(suite, CALIB_SEEDS, N, B)
    cal[(suite, N, B)] = dict(rho={W: [rho(st['kid'], W) for st in ss] for W in G_W}, pre={b: [prep(st, b, Pr, Pc) for st in ss] for b in G_B})
print('calib prep', round(time.time()-t0, 1), flush=True)
grid = {}
for b, th, tr, W in itertools.product(G_B, G_TH, G_TR, G_W):
    for te in tauE_opts(tr):
        e = [np.mean((cond2(p, r, th, tr, te)[0]-p['y'])**2) for c in cal.values() for r, ps in zip(c['rho'][W], c['pre'][b]) for p in ps]
        grid[(b, th, tr, te, W)] = float(np.mean(e))
best = min(grid, key=grid.get); bb, bth, btr, bte, bW = best
print('best', best, grid[best], flush=True)
calib_rho = {f'{k[0]}_N{k[1]}_B{k[2]}': float(np.mean(v['rho'][bW])) for k, v in cal.items()}; del cal
# ---- TEST ----
rows, meta = [], {}
for suite, N, B in cells:
    R1, M, Rc, Pc = cell_geom(suite, B); Pr = hy.proj(Rc); Px = eh.proj(R1); ssc = sets(suite, CALIB_SEEDS, N, B)
    bx = min(hy.BETAS, key=lambda b: np.mean([np.mean((eh.expanded_read(q, eh.expanded_update(st['K'], st['V'], b, Px), Px)-y)**2) for st in ssc for q, y in st['qs']]))
    meta[f'{suite}_N{N}_B{B}'] = dict(R_exp=int(R1), exp_beta=bx, exp_kib=R1*D*V*8/1024, R_cond2=Rc, P_cond2=Pc, cond2_kib=(Rc*D*V+D*Rc+Pc*(D+V))*8/1024)
    for seed, st in zip(TEST_SEEDS, sets(suite, TEST_SEEDS, N, B)):
        K, Vv = st['K'], st['V']; S = state_update(K, Vv, beta=BETA); Sx = eh.expanded_update(K, Vv, bx, Px)
        recB = st['pf'](B); r = rho(st['kid'], bW); pre = prep(st, bb, Pr, Pc)
        for g, (q, y), p in zip(SIGS, st['qs'], pre):
            o, gate, modeR = cond2(p, r, bth, btr, bte); mr, sr = eh.archive_read(q, K, Vv, recB)
            preds = {'SDA-Cond2': o, 'SDA-Recency': q@S + (sr >= TAU)[:, None]*(Vv[mr]-K[mr]@S), 'Expanded DeltaNet (eq-mem)': eh.expanded_read(q, Sx, Px)}
            for k, pr_ in preds.items():
                per = np.mean((pr_-y)**2, 1); ok = per < 1e-3; inP = p['inP']
                rr = dict(suite=suite, N=N, B=B, noise=g, seed=seed, method=k, mse=per.mean(), rho_rw=r, modeR=float(modeR), pool_hit=gate.mean(),
                          pool_hit_correct=(gate & inP).mean(), frac_inPool=inP.mean(),
                          mse_inPool=per[inP].mean() if inP.any() else np.nan, mse_outPool=per[~inP].mean() if (~inP).any() else np.nan)
                if suite == 'hard': rw = st['d']['is_rw']; rr.update(rewrite_hit=ok[rw].mean(), stable_hit=ok[~rw].mean(), rewrite_mse=per[rw].mean())
                rows.append(rr)
    print('done', suite, N, B, meta[f'{suite}_N{N}_B{B}'], flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT/'raw_results.csv', index=False)
agg = df.groupby(['suite', 'N', 'B', 'noise', 'method'], as_index=False).mean(numeric_only=True).drop(columns='seed'); agg.to_csv(OUT/'summary_results.csv', index=False)
pr = []
for (s, N, B, sig), w in df.groupby(['suite', 'N', 'B', 'noise']):
    w = w.pivot(index='seed', columns='method', values='mse')
    for a_ in ['Expanded DeltaNet (eq-mem)', 'SDA-Recency']:
        dd = w[a_]-w['SDA-Cond2']; se = dd.std()/4
        pr.append(dict(suite=s, N=N, B=B, noise=sig, delta=f'{a_} − Cond2', mean=dd.mean(), ci_lo=dd.mean()-ex.T95_DF15*se, ci_hi=dd.mean()+ex.T95_DF15*se))
pr = pd.DataFrame(pr); pr.to_csv(OUT/'paired_results.csv', index=False)
pd.DataFrame([dict(beta=k[0], theta_rw=k[1], tau_R=k[2], tau_E=k[3], W=k[4], calib_mse=v) for k, v in grid.items()]).to_csv(OUT/'calib_grid.csv', index=False)
(OUT/'configuration.json').write_text(json.dumps(dict(frozen=dict(beta=bb, theta_rw=bth, tau_R=btr, tau_E=bte, W=bW, calib_mse=grid[best]), selection='global, CALIB_SEEDS only, all faces/cells/sigmas pooled, no face labels',
    calib_rho_rw=calib_rho, sda_recency=dict(beta=BETA, tau=TAU), cells=meta, holdout='not run'), indent=1, default=float))
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(agg[['suite','N','B','noise','method','mse','modeR','pool_hit','rewrite_hit','mse_outPool']].round(4).to_string(index=False)); print(pr.round(4).to_string(index=False))
