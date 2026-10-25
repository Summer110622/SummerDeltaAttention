"""Equal-memory hard suite (today task 1): rewrite + hard-negative distractors.

Reuses research/experiment.py helpers (seeds, state_update, queries_for, committed
SDA/KDA/Delta hyperparameters). Writes only to research/eqmem_hard/.
Synthetic NumPy float64 CPU; no LM claims.
Usage: OPENBLAS_NUM_THREADS=1 python research/eqmem_hard.py [--n 2048 --rho 0.5]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as ex
from experiment import D, V, Q, CALIB_SEEDS, TEST_SEEDS, state_update, queries_for, mse, T95_DF15

OUT = Path(__file__).resolve().parent/'eqmem_hard'
SDA = 'SDA v5-Gated (calibrated)'

def make_hard_data(seed, n):
    """Sequence of n writes: U=floor(.85n) unique keys (incl. Q distractors), n-U rewrites.
    Each query target gets one hard-negative key at 15-30 deg, written after the target's last write.
    Half the queries target rewritten keys. Ground truth = latest value."""
    rng = np.random.default_rng(seed + 7_000_000)  # derived stream; separate from base make_data
    U = int(np.floor(.85*n)); n_base = U - Q; n_rw = n - U
    base = rng.normal(size=(n_base, D)); base /= np.linalg.norm(base, axis=1, keepdims=True)
    rw_ids = rng.choice(n_base, size=n_rw, replace=True)
    rewritten = np.unique(rw_ids)
    nonrw = np.setdiff1d(np.arange(n_base), rewritten)
    tgt = np.concatenate([rng.choice(rewritten, Q//2, replace=False), rng.choice(nonrw, Q-Q//2, replace=False)])
    # distractors
    ang = np.deg2rad(rng.uniform(15, 30, size=Q))
    r = rng.normal(size=(Q, D)); kt = base[tgt]
    r -= (r*kt).sum(1, keepdims=True)*kt; r /= np.linalg.norm(r, axis=1, keepdims=True)
    dk = np.cos(ang)[:, None]*kt + np.sin(ang)[:, None]*r
    # order: base keys in random order, rewrites at random later positions, distractor after target's last write
    events = [('b', i) for i in rng.permutation(n_base)]
    for i in rw_ids:
        first = next(j for j, e in enumerate(events) if e == ('b', i))
        events.insert(rng.integers(first+1, len(events)+1), ('r', i))
    for qi, i in enumerate(tgt):
        last = max(j for j, e in enumerate(events) if e[1] == i and e[0] in 'br')
        events.insert(rng.integers(last+1, len(events)+1), ('d', qi))
    keys = np.empty((n, D)); values = rng.normal(size=(n, V)); key_id = np.empty(n, int)
    latest = {}
    for t, (kind, i) in enumerate(events):
        if kind == 'd': keys[t] = dk[i]; key_id[t] = n_base + i
        else: keys[t] = base[i]; key_id[t] = i; latest[i] = t
    is_rw = np.isin(tgt, rewritten)
    target_pos = np.array([latest[i] for i in tgt])
    z = rng.normal(size=(Q, D)); perm = rng.permutation(n)
    return dict(keys=keys, values=values, key_id=key_id, tgt_pos=target_pos, is_rw=is_rw, z=z, perm=perm, U=U)

def hard_queries(d, sigma):
    q = d['keys'][d['tgt_pos']] + sigma*d['z']/np.sqrt(D)
    return q/np.linalg.norm(q, axis=1, keepdims=True)

def archive_read(q, keys, values, idx):
    """top1 cosine over retained positions; ties broken toward latest position."""
    idx = np.sort(idx)[::-1]
    s = q@keys[idx].T; c = np.argmax(s, 1)
    return idx[c], s[np.arange(len(q)), c]

def route(k, P): return np.argmax(k@P, axis=-1)

def expanded_update(keys, values, beta, P):
    S = np.zeros((P.shape[1], D, V)); blk = route(keys, P)
    for k, v, b in zip(keys, values, blk):
        S[b] += beta*np.outer(k, v - k@S[b])
    return S

def expanded_read(q, S, P):
    b = route(q, P); return np.einsum('qd,qdv->qv', q, S[b])

def recency_positions(key_id, B):
    """LRU-B: keep the B most recently written distinct keys, each at its latest write."""
    last = {}
    for t, k in enumerate(key_id): last[k] = t
    pos = np.array(sorted(last.values()))
    return pos[-B:]

def n_blocks(B):
    M = (D*V + B*(D+V))*8
    R = (M//8 - D*V)//(D*V)
    return R+1, M

def proj(R1): return np.random.default_rng(424242).normal(size=(D, R1))

def calibrate_expanded(n, rho):
    B = round(rho*n); R1, _ = n_blocks(B); P = proj(R1); res = {}
    for beta in [.05, .1, .2, .4, .7, 1.]:
        e = []
        for s in CALIB_SEEDS:
            d = make_hard_data(s, n); S = expanded_update(d['keys'], d['values'], beta, P)
            for sig in [0., .4]:
                e.append(mse(expanded_read(hard_queries(d, sig), S, P), d['values'][d['tgt_pos']]))
        res[beta] = float(np.mean(e))
    best = min(res, key=res.get)
    return best, res

def run(n, rho, noises):
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    cfg = json.loads((Path(ex.__file__).parent/'configuration.json').read_text())
    B = round(rho*n); R1, M_sda = n_blocks(B); P = proj(R1)
    beta_x, grid = calibrate_expanded(n, rho)
    meta = dict(n=n, rho=rho, B=B, R_plus_1=int(R1), expanded_beta=beta_x, expanded_calib_grid=grid,
                committed=cfg, memory_sda_kib=M_sda/1024, memory_expanded_kib=R1*D*V*8/1024)
    (OUT/'configuration.json').write_text(json.dumps(meta, indent=2, default=str))
    rows = []; qrows = []
    for seed in TEST_SEEDS:
        d = make_hard_data(seed, n); K, Vv = d['keys'], d['values']
        S_sda = state_update(K, Vv, beta=cfg['sda']['beta'])
        S_delta = state_update(K, Vv, **cfg['delta'])
        S_kda = state_update(K, Vv, **cfg['kda'], channelwise=True)
        S_x = expanded_update(K, Vv, beta_x, P)
        chosen = d['perm'][:B]; rec = recency_positions(d['key_id'], B)
        tau = cfg['sda']['threshold']
        for sig in noises:
            q = hard_queries(d, sig); y = Vv[d['tgt_pos']]
            m, s = archive_read(q, K, Vv, chosen); acc = s >= tau
            mr, _ = archive_read(q, K, Vv, rec)
            preds = {
                SDA: (q@S_sda + acc[:, None]*(Vv[m] - K[m]@S_sda), m, (D*V+B*(D+V))*8),
                'Indexed KV (reservoir, zero fallback)': (np.where(acc[:, None], Vv[m], 0.), m, B*(D+V)*8),
                'No-decay DeltaNet (calibrated beta)': (q@S_delta, None, D*V*8),
                'KDA channel-gated proxy': (q@S_kda, None, D*V*8),
                'Expanded DeltaNet (eq-mem)': (expanded_read(q, S_x, P), None, R1*D*V*8),
                'Recency KV (eq-B)': (Vv[mr], mr, B*(D+V)*8),
            }
            for meth, (p, hit, mem) in preds.items():
                per = np.mean((p-y)**2, 1); ok = per < 1e-3
                rows.append(dict(n=n, noise=sig, retention=rho, seed=seed, method=meth, mse=float(per.mean()),
                                 memory_kib=mem/1024,
                                 hit_rate=float(np.mean(hit == d['tgt_pos'])) if hit is not None else np.nan,
                                 accepted_rate=float(acc.mean()) if meth.startswith(('SDA', 'Indexed')) else np.nan,
                                 rewrite_hit=float(ok[d['is_rw']].mean()), stable_hit=float(ok[~d['is_rw']].mean()),
                                 rewrite_mse=float(per[d['is_rw']].mean()), stable_mse=float(per[~d['is_rw']].mean())))
    df = pd.DataFrame(rows); df.to_csv(OUT/'raw_results.csv', index=False)
    agg = df.groupby(['n', 'noise', 'retention', 'method'], as_index=False).agg(
        mean_mse=('mse', 'mean'), sd_mse=('mse', 'std'), memory_kib=('memory_kib', 'mean'),
        hit_rate=('hit_rate', 'mean'), accept=('accepted_rate', 'mean'))
    agg['sem'] = agg.sd_mse/np.sqrt(len(TEST_SEEDS)); agg.to_csv(OUT/'summary_results.csv', index=False)
    rw = df.groupby(['n', 'noise', 'method'], as_index=False)[['rewrite_hit', 'stable_hit', 'rewrite_mse', 'stable_mse']].mean()
    rw.to_csv(OUT/'rewrite_hit.csv', index=False)
    paired = ex.paired_vs_sda(df, ref=SDA); paired.to_csv(OUT/'paired_results.csv', index=False)
    print(json.dumps(meta, indent=1, default=str))
    for sig in noises:
        print(f'--- sigma={sig}'); print(agg[agg.noise == sig][['method', 'mean_mse', 'sem', 'memory_kib', 'hit_rate', 'accept']].sort_values('mean_mse').to_string(index=False))
        print(paired[paired.noise == sig].to_string(index=False)); print(rw[rw.noise == sig].to_string(index=False))
    print('time', time.time()-t0)

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--n', type=int, default=2048); a.add_argument('--rho', type=float, default=.5)
    a.add_argument('--noise', type=float, nargs='*', default=[0., .4]); a.add_argument('--hard', default='rewrite,distractor')
    x = a.parse_args(); assert set(x.hard.split(',')) == {'rewrite', 'distractor'}, 'only combined hard mode implemented'
    run(x.n, x.rho, x.noise)
