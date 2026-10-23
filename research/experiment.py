"""Reproducible synthetic benchmark for Summer Delta Attention v5-Gated vs KDA.

This benchmarks the *KDA recurrent update*, not a trained Kimi Linear model. Its
per-channel retention gate is an explicit (non-learned) input-dependent proxy.
NumPy CPU float64, post-write query phase; no GPU timing or ANN speed claim.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
D, V, Q = 32, 16, 128
CALIB_SEEDS = [20250001 + i for i in range(8)]
TEST_SEEDS = [20261023 + i for i in range(16)]
N_MAIN = 512
NS = [128, 512, 2048]
NOISES = [0., .4, 1.]
BUDGETS = [.1,.25,.5,1.]
SCALE = 32.


def make_data(seed, n):
    rng=np.random.default_rng(seed)
    keys=rng.normal(size=(n,D))
    keys/=np.linalg.norm(keys,axis=1,keepdims=True)
    values=rng.normal(size=(n,V))
    ids=rng.integers(0,n,size=Q)
    # Independent random draws for query noise and memory sampling across settings
    z=rng.normal(size=(Q,D))
    permutation=rng.permutation(n)
    return keys, values, ids, z, permutation


def queries_for(keys, ids, z, sigma):
    q=keys[ids] + sigma*z/np.sqrt(D)
    return q/np.linalg.norm(q,axis=1,keepdims=True)


def state_update(keys,values,beta,decay_rate=0.,gate_eta=0.,channelwise=False):
    S=np.zeros((D,V),dtype=np.float64)
    profile=np.linspace(.5,1.5,D) if channelwise else np.ones(D)
    for k,v in zip(keys,values):
        if decay_rate:
            # decay alpha[t,j] = exp(-decay_rate * profile[j] * (1+eta*sqrt(d)*|k[t,j]|))
            a=np.exp(-decay_rate*profile*(1+gate_eta*np.sqrt(D)*np.abs(k)))
            S*=a[:,None]
        residual=v-k@S
        S+=beta*np.outer(k,residual)
    return S


def mse(pred,target):
    return float(np.mean((pred-target)**2))


def calibrate():
    # Tune each method on separate synthetic development seeds, no test leakage.
    dev=[make_data(seed,N_MAIN) for seed in CALIB_SEEDS]
    beta_grid=[.05,.1,.2,.4,.7,1.]
    decay_grid=[0.,.0005,.002,.008,.03]
    eta_grid=[0.,1.]
    best_kda=(float('inf'),None)
    best_kda_positive=(float('inf'),None)
    best_delta=(float('inf'),None)
    cache={}
    for beta in beta_grid:
        for rate in decay_grid:
            for eta in eta_grid:
                if rate==0 and eta==1: continue
                errors=[]
                for keys,values,ids,z,_ in dev:
                    S=state_update(keys,values,beta,rate,eta,channelwise=True)
                    # evaluate both exact and moderately noisy queries
                    for sigma in [0.,.4]:
                        q=queries_for(keys,ids,z,sigma)
                        errors.append(mse(q@S,values[ids]))
                m=float(np.mean(errors))
                cache[(beta,rate,eta)]=m
                if m<best_kda[0]:best_kda=(m,dict(beta=beta,decay_rate=rate,gate_eta=eta))
                if rate>0 and m<best_kda_positive[0]:best_kda_positive=(m,dict(beta=beta,decay_rate=rate,gate_eta=eta))
                if rate==0 and m<best_delta[0]:best_delta=(m,dict(beta=beta))
    # SDA beta + similarity threshold calibrated at 50% retention (reference budget).
    best_sda=(float('inf'),None)
    for beta in beta_grid:
        for tau in [.65,.75,.8,.85,.9]:
            errors=[]
            for keys,values,ids,z,perm in dev:
                S=state_update(keys,values,beta)
                chosen=perm[:round(.5*N_MAIN)]
                for sigma in [0.,.4]:
                    q=queries_for(keys,ids,z,sigma)
                    p=predictions(q,keys,values,S,chosen,tau)
                    errors.append(mse(p['SDA tuned'],values[ids]))
            m=float(np.mean(errors))
            if m<best_sda[0]:best_sda=(m,dict(beta=beta,threshold=tau))
    return dict(kda=best_kda_positive[1],kda_boundary=best_kda[1],delta=best_delta[1],sda=best_sda[1],dev_mse=dict(kda_positive=best_kda_positive[0],kda_boundary=best_kda[0],delta=best_delta[0],sda=best_sda[0]),seed_splits=dict(calibration=CALIB_SEEDS,test=TEST_SEEDS))


def predictions(q,keys,values,state,indices,threshold):
    if len(indices)==0:
        out=q@state
        return {'SDA tuned':out,'KV only (zero fallback)':np.zeros_like(out), 'SDA legacy':out,
                'softmax':np.zeros_like(out),
                'hit':np.zeros(len(q),dtype=bool),'accepted':np.zeros(len(q),dtype=bool)}
    scores=q@keys[indices].T
    cols=np.argmax(scores,axis=1)
    match=indices[cols]
    sim=scores[np.arange(len(q)),cols]
    accepted=sim>=threshold
    vhat=values[match]
    out_fast=q@state
    correction=vhat-keys[match]@state
    # Equal-memory control: the same sharp softmax as the Full-KV reference, but
    # restricted to the retained archive (no gate, no fast state).
    logits=SCALE*scores
    weights=np.exp(logits-logits.max(axis=1,keepdims=True)); weights/=weights.sum(axis=1,keepdims=True)
    return {'SDA tuned':out_fast+accepted[:,None]*correction,
            'softmax':weights@values[indices],
            'KV only (zero fallback)':np.where(accepted[:,None],vhat,0.),
            'hit':match,'accepted':accepted}


T95_DF15=2.131449545559323  # two-sided 95% Student-t critical value, 15 d.o.f. (16 seeds)


def paired_vs_sda(df,ref='SDA v5-Gated (calibrated)'):
    """Per-seed paired differences (comparator - SDA); positive means SDA has lower MSE."""
    rows=[]
    for (n,noise,ret),g in df.groupby(['n','noise','retention']):
        wide=g.pivot(index='seed',columns='method',values='mse')
        assert len(wide)==len(TEST_SEEDS)==16, 'critical value assumes 16 seeds'
        for other in wide.columns:
            if other==ref: continue
            d=(wide[other]-wide[ref]).to_numpy()
            half=T95_DF15*d.std(ddof=1)/np.sqrt(len(d))
            rows.append(dict(n=n,noise=noise,retention=ret,comparator=other,
                             mean_diff=float(d.mean()),ci95_low=float(d.mean()-half),ci95_high=float(d.mean()+half)))
    return pd.DataFrame(rows)


def run():
    t0=time.time()
    cfg=calibrate()
    (OUT/'configuration.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    rows=[]
    for n in NS:
        for seed in TEST_SEEDS:
            keys,values,ids,z,permutation=make_data(seed,n)
            S_kda=state_update(keys,values,**cfg['kda'],channelwise=True)
            S_kda_boundary=state_update(keys,values,**cfg['kda_boundary'],channelwise=True)
            S_delta=state_update(keys,values,**cfg['delta'])
            S_sda=state_update(keys,values,beta=cfg['sda']['beta'])
            S_old=state_update(keys,values,beta=1.)
            assert np.isfinite(S_kda).all()
            for sigma in NOISES:
                q=queries_for(keys,ids,z,sigma)
                targets=values[ids]
                logits=SCALE*(q@keys.T)
                prob=np.exp(logits-logits.max(axis=1,keepdims=True)); prob/=prob.sum(axis=1,keepdims=True)
                full=prob@values
                preds_base={'KDA channel-gated proxy':q@S_kda,
                            'KDA unrestricted (alpha=1 boundary)':q@S_kda_boundary,
                            'No-decay DeltaNet (calibrated beta)':q@S_delta,
                            'Full KV softmax (sharp)':full,
                            # Values are N(0,1), so this is the trivial MSE~1 floor.
                            'Zero predictor':np.zeros_like(targets)}
                for frac in BUDGETS:
                    b=round(n*frac)
                    chosen=permutation[:b]
                    p=predictions(q,keys,values,S_sda,chosen,cfg['sda']['threshold'])
                    p_old=predictions(q,keys,values,S_old,chosen,.8)
                    pred={**preds_base, 'SDA v5-Gated (calibrated)':p['SDA tuned'],
                          'SDA v5-Gated (original)':p_old['SDA tuned'],
                          'Indexed KV (zero fallback)':p['KV only (zero fallback)'],
                          'Archive softmax (same retained KV)':p['softmax']}
                    for method,output in pred.items():
                        if method.startswith('Zero'):
                            mem=0
                        elif method.startswith('KDA') or method.startswith('No-decay'):
                            mem= D*V*8
                        elif method.startswith('Full'):
                            mem=n*(D+V)*8
                        elif method.startswith('SDA'):
                            mem=(D*V+b*(D+V))*8
                        else:
                            mem=b*(D+V)*8
                        rows.append(dict(n=n,noise=sigma,retention=frac,seed=seed,method=method,
                                         mse=mse(output,targets),
                                         memory_kib=mem/1024,
                                         hit_rate=float(np.mean(p['hit']==ids)),
                                         accepted_rate=float(np.mean(p['accepted']))))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'raw_results.csv',index=False)
    agg=df.groupby(['n','noise','retention','method'],as_index=False).agg(mean_mse=('mse','mean'),sd_mse=('mse','std'), memory_kib=('memory_kib','mean'),mean_hit=('hit_rate','mean'),mean_accept=('accepted_rate','mean'))
    agg['sem']=agg.sd_mse/np.sqrt(len(TEST_SEEDS))
    agg.to_csv(OUT/'summary_results.csv',index=False)
    paired=paired_vs_sda(df)
    paired.to_csv(OUT/'paired_results.csv',index=False)
    print('Paired (comparator - SDA calibrated), n512 noise0 rho0.5')
    print(paired[(paired.n==512)&(paired.noise==0)&(paired.retention==.5)].to_string(index=False))
    print('Config:',json.dumps(cfg,indent=2))
    print('Main n512 noise0 rho0.5')
    print(agg[(agg.n==512)&(agg.noise==0)&(agg.retention==.5)][['method','mean_mse','sem','memory_kib']].sort_values('mean_mse').to_string(index=False))
    print('Main n512 noise0.4 rho0.5')
    print(agg[(agg.n==512)&(agg.noise==.4)&(agg.retention==.5)][['method','mean_mse','sem','memory_kib']].sort_values('mean_mse').to_string(index=False))
    print('Run time',time.time()-t0)
    return agg

if __name__=='__main__':run()
