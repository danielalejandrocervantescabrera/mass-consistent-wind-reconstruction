#!/usr/bin/env python3
"""Report reference or new-run metrics without conflating them."""
from pathlib import Path
import argparse,json,numpy as np
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--prepared-only',action='store_true')
p.add_argument('--model-root',type=Path,default=root/'outputs/models')
p.add_argument('--analysis-root',type=Path,default=root/'outputs/analysis')
a=p.parse_args()
prep=root/'data/processed/manifest.json'
if prep.exists():
 m=json.loads(prep.read_text()); v=m['preprocessing']
 assert m['n_power_snapshots_available']==m['n_train_snapshots']==61
 assert all(not m.get(k) for k in ('dev_key','val_key','test_key_reserved'))
 assert (v['nx'],v['ny'],len(v['z_targets_m']),v['n_generated_times'])==(40,40,15,721)
 assert v['wahba_lambda']==0.001 and v['dt_minutes']==5.0
 print('Prepared data: PASS (61 hours, 721 times, 40x40x15, lambda=0.001)')
else: print('Prepared dataset absent; only archived reference is available.')
if a.prepared_only:raise SystemExit(0)
runs=[a.model_root/f'seed_{s}/summary.json' for s in (1234,2345,3456)]
new=all(f.exists() for f in runs)
if not new:runs=[root/f'reference/models/seed_{s}/summary.json' for s in (1234,2345,3456)]
values=[json.loads(f.read_text()) for f in runs]
assert [x['seed'] for x in values]==[1234,2345,3456]
print('Summary type:', ('HISTORICAL CHECKPOINT METADATA (NOT recalculated)' if a.model_root==root/'reference/checkpoints' else 'NEW TRAINING METADATA (NOT recalculated)') if new else 'ARCHIVED REFERENCE (NOT recomputed training)')
def show(label,f,scale=1,unit=''):
 x=np.array([f(r) for r in values])*scale
 print(f'{label:26s} {x.mean():.7g} +/- {x.std(ddof=1):.5g} {unit}')
show('real uv relL2',lambda r:r['full_data_uv_metrics']['real']['relL2_uv'],100,'%')
show('real uv RMSE',lambda r:r['full_data_uv_metrics']['real']['RMSE_uv_mps'],unit='m/s')
show('interior divergence RMS',lambda r:r['physics_metrics']['rms_div_phys_sinv'],unit='s^-1')
show('bottom w RMS',lambda r:r['physics_metrics']['rms_bdy_phys_mps'],unit='m/s')
show('interior w RMS',lambda r:r['physics_metrics']['rms_w_phys_mps'],unit='m/s')
for filename,output in [('vertical_velocity_summary.json','outputs/analysis/vertical_seed_2345'),('w_seed_stability_summary.json','outputs/analysis/seed_stability')]:
 fname='w_seed_stability_summary.json' if filename.startswith('w_seed') else filename
 p=a.analysis_root/Path(output).name/fname
 if not p.exists(): p=root/'reference/analysis'/filename
 d=json.loads(p.read_text())
 if filename.startswith('vertical'):
  c=d['statistics_over_real_hourly_times']['corr_wtheta_increment_vs_wkin_obs']
  print('Kinematic correlation:',c['mean'],'+/-',c['std'])
 else:print('w seed spread:',100*d['ensemble']['relative_rms_spread'],'%')
 print('  source:', 'archived reference' if 'reference' in p.parts else 'new-run output')
print('Caution: these diagnostics do not independently validate atmospheric vertical velocity.')
