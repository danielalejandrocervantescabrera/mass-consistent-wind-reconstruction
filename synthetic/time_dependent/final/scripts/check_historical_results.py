#!/usr/bin/env python3
"""Check archived original temporal checkpoints, metadata and virgin test statistics.
No model training or test-set tuning happens here.
"""
import csv
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent

def load_csv(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))

runs = [
    load_csv(ROOT/'out_time_obs_sweep/summary/timeobs_n16384_s0_eval_summary.csv')[0],
    load_csv(ROOT/'out_time_final_16384/summary/timefinal16384_s1_eval_summary.csv')[0],
    load_csv(ROOT/'out_time_final_16384/summary/timefinal16384_s2_eval_summary.csv')[0],
]
expected={
    'traj':'parabolic', 'nObs':'16384', 'obsSampling':'lhs', 'obsSeed':'12345',
    'evalSeed':'54321', 'width':'60', 'depth':'2', 'nEpoch':'2000', 'nBatch':'25',
    'innerBatchSize':'512', 'bdyBatchSize':'64', 'beta1':'1.0', 'beta2':'1.0',
    'step_size':'200', 'gamma':'0.5', 'epsilon':'0.02',
}
for seed, row in enumerate(runs):
    assert int(row['seed'])==seed, (seed,row['seed'])
    for key,val in expected.items():
        assert row[key]==val, (seed,key,row[key],val)
for k in runs[0]:
    if k in ('seed', 'best_epoch','best_val_mse_uv','best_val_mse_uvw_diag',
             'best_val_relL2_diag','test_mse_uv','test_mse_uvw','test_relL2',
             'test_rms_div','test_rms_bottom','test_rel_uv','test_rel_uvw',
             'train_seconds','checkpoint_path','observation_file','loss_csv','time_metrics_csv'):
        continue
    assert len({r[k] for r in runs})==1, ('unexpected run config variation',k)

ckpt_hashes=[]
for line in (ROOT/'reference/HISTORICAL_CHECKPOINTS_SHA256.txt').read_text().splitlines():
    if not line or line.startswith('#'):continue
    digest, relpath = line.split(None,1)
    p=ROOT/relpath
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest, p
    ckpt_hashes.append(p)
assert len(ckpt_hashes)==3

try:
    import torch
    for seed,p in enumerate(ckpt_hashes):
        ck=torch.load(p,map_location='cpu',weights_only=True)
        assert ck['epoch']==int(runs[seed]['best_epoch'])
        assert abs(float(ck['val_mse_uv'])-float(runs[seed]['best_val_mse_uv']))<1e-10
        assert len(ck['model_state_dict'])==14
    print('CHECKPOINTS: 3 readable original state dictionaries; metadata agrees')
except ImportError:
    print('CHECKPOINTS: SHA256 passed; torch not installed, state dictionaries not checked')

virgin = load_csv(ROOT/'reference/independent/time_final_16384_virgin_runs.csv')
agg = load_csv(ROOT/'reference/independent/time_final_16384_virgin_mean.csv')[0]
assert [int(r['seed']) for r in virgin]==[0,1,2]
for r in virgin:
    assert int(r['final_eval_seed'])==20260929
    assert int(r['nTest'])==50000 and int(r['nTestBottom'])==20000
    assert int(r['checkpoint_epoch'])==int(runs[int(r['seed'])]['best_epoch'])

metrics=['mse_uv','mse_uvw','relL2_uv','relL2_uvw','rms_div','rms_bottom']
for m in metrics:
    v=np.array([float(r[m]) for r in virgin])
    assert np.isclose(np.mean(v),float(agg[m+'_mean']),atol=1e-14,rtol=1e-12), m
    assert np.isclose(np.std(v,ddof=1),float(agg[m+'_std']),atol=1e-14,rtol=1e-12), m
    print(f'{m:>12}: {v.mean():.9e} +/- {v.std(ddof=1):.9e}')

slices=load_csv(ROOT/'reference/independent/time_final_16384_virgin_by_time_runs.csv')
sliceagg=load_csv(ROOT/'reference/independent/time_final_16384_virgin_by_time_mean.csv')
assert len(slices)==27 and len(sliceagg)==9
for row in sliceagg:
    group=[r for r in slices if r['t']==row['t']]
    assert len(group)==3 and {int(r['seed']) for r in group}=={0,1,2}
    for m in metrics:
        v=np.array([float(r[m]) for r in group])
        assert np.isclose(v.mean(),float(row[m+'_mean']),rtol=1e-12,atol=1e-14)
        assert np.isclose(v.std(ddof=1),float(row[m+'_std']),rtol=1e-12,atol=1e-14)
print('PASS: archived checkpoint metadata + 50,000-point independent test + 9 time slices')
