#!/usr/bin/env python3
"""Audit archived historical Melissa checkpoints, NOT a recomputation of scientific metrics."""
from pathlib import Path
import hashlib,sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from train_melissa_reconstruction_final import ResidualNet

EXPECTED_SEEDS=(1234,2345,3456)
sha_records={}
for line in (ROOT/'reference/checkpoints/SHA256SUMS.txt').read_text().splitlines():
    if line.strip():
        sha, rel=line.split(None,1)
        sha_records[rel.strip()]=sha.strip()
assert len(sha_records)==3, 'Expected exactly 3 archived checkpoints'
results=[]
for seed in EXPECTED_SEEDS:
    p=ROOT/f'reference/checkpoints/seed_{seed}/final_model.pt'
    summary=ROOT/f'reference/models/seed_{seed}/summary.json'
    assert p.is_file() and summary.is_file(), f'Missing archival files for seed {seed}'
    rel=p.relative_to(ROOT).as_posix()
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha_records[rel]==digest, f'SHA256 mismatch for {p}'
    desc=json.loads(summary.read_text())
    assert desc['seed']==seed
    assert desc['architecture']['width']==80 and desc['architecture']['depth']==5
    assert desc['architecture']['n_parameters']==65443
    assert desc['architecture']['fourier_features'] is False
    assert desc['training']['epochs']==2500 and desc['training']['batches_per_epoch']==25
    assert desc['final_epoch']==2499
    assert desc['sampling']['real_hourly_times']==61
    assert desc['sampling']['generated_times']==721
    assert desc['sampling']['real_batch_size']==256
    assert desc['sampling']['interp_batch_size']==256
    assert desc['loss']['beta_div']==0.03 and desc['loss']['beta_bdy']==1.0
    state=torch.load(p,map_location='cpu',weights_only=True)
    assert isinstance(state,dict),f'Not a model state dictionary: {p}'
    model=ResidualNet(width=80,depth=5)
    model.load_state_dict(state,strict=True)
    assert sum(t.numel() for t in model.parameters())==65443
    for k,v in model.state_dict().items():
        assert torch.isfinite(v).all(),f'Nonfinite model parameter: seed={seed}, {k}'
    with torch.no_grad():
        y=model(torch.tensor([[0.5,0.5,0.5,0.5]],dtype=torch.float32))
    assert tuple(y.shape)==(1,3) and torch.isfinite(y).all()
    results.append(desc)
    print(f'PASS seed={seed}: SHA256, strict state load, 65443 finite parameters; final epoch 2499')

def row(label,fn,scale=1.,unit=''):
    v=np.array([fn(x) for x in results],dtype=np.float64)*scale
    print(f'  {label:28s} {v.mean():.9g} +/- {v.std(ddof=1):.9g} {unit}')
print('\nHISTORICAL JSON METRICS (recorded at training; NOT independently recalculated):')
row('observed-hour uv rel L2',lambda r:r['full_data_uv_metrics']['real']['relL2_uv'],100,'%')
row('observed-hour uv RMSE',lambda r:r['full_data_uv_metrics']['real']['RMSE_uv_mps'],unit='m/s')
row('RMS physical divergence',lambda r:r['physics_metrics']['rms_div_phys_sinv'],unit='s^-1')
row('RMS bottom velocity',lambda r:r['physics_metrics']['rms_bdy_phys_mps'],unit='m/s')
row('RMS interior w',lambda r:r['physics_metrics']['rms_w_phys_mps'],unit='m/s')
print('\nPASS: 3 archived checkpoints and their training metadata were verified.')
print('NEXT: scripts/07_recompute_historical.sh to process 61 hours and recompute derived diagnostics.')
print('NOTE: independent observed vertical wind data are NOT available for validation.')
