"""SDA-Hybrid-RF (Arch M-D3d): Recency LRU pool (exact k,v) + Routed multi-F (R blocks DxV, fixed projection).
Memory (float64) counted = R*D*V + D*R (projection, counted as a handicap; Expanded does not count its projection) + P*(D+V)."""
import numpy as np
from experiment import D, V
def split(M_floats, R):
    P = (M_floats - R*D*V - D*R)//(D+V); assert P > 0; return int(P)
def R_default(R1_expanded):  # split A: 48/97 of Expanded's blocks
    return max(1, round(48/97*R1_expanded))
def proj(R): return np.random.default_rng(515151).normal(size=(D, R))
def route(x, Pr): return np.argmax(x@Pr, axis=-1)
def update(keys, values, beta, Pr):
    S = np.zeros((Pr.shape[1], D, V)); blk = route(keys, Pr)
    for k, v, b in zip(keys, values, blk): S[b] += beta*np.outer(k, v - k@S[b])
    return S
def read(q, keys, values, S, Pr, pool, tau, mode='residual'):
    """pool = retained positions (LRU distinct latest P). Ties toward latest position."""
    f = np.einsum('qd,qdv->qv', q, S[route(q, Pr)])
    idx = np.sort(pool)[::-1]; s = q@keys[idx].T; c = np.argmax(s, 1)
    j = idx[c]; cs = s[np.arange(len(q)), c]; g = cs >= tau
    kj = keys[j]; fj = np.einsum('qd,qdv->qv', kj, S[route(kj, Pr)])
    corr = values[j] - fj if mode == 'residual' else values[j] - f
    return f + g[:, None]*corr, j, g
BETAS = [.05, .1, .2, .4, .7, 1.]; TAUS = [.5, .6, .7, .75, .8, .85, .9, .95]
def calibrate(calib_sets, R, P):
    """calib_sets: list of (keys, values, pool_fn(P), [(q,y),...]). Joint grid on CALIB seeds only."""
    Pr = proj(R); res = {}
    for b in BETAS:
        Ss = [update(K, Vv, b, Pr) for K, Vv, _, _ in calib_sets]
        for t in TAUS:
            e = [np.mean((read(q, K, Vv, S, Pr, pf(P), t)[0]-y)**2) for (K, Vv, pf, qs), S in zip(calib_sets, Ss) for q, y in qs]
            res[(b, t)] = float(np.mean(e))
    best = min(res, key=res.get); return best, {f'{k[0]}_{k[1]}': v for k, v in res.items()}
