"""i.i.d. suite helpers shared by recency_adverse_v2 and hybrid_rf. mode='uniform' (v1: ids~Unif{0..N-1}) or 'old' (v2: ids~Unif{0..N-B-1})."""
import numpy as np
from experiment import make_data, Q
def data(seed, N, B, mode):
    K, Vv, ids, z, perm = make_data(seed, N)
    if mode == 'old': ids = np.random.default_rng(seed + 9_000_000).integers(0, N-B, size=Q)
    return K, Vv, ids, z, perm
