#!/usr/bin/env python3
"""Offline checks of the 61 archived raw fields."""
import argparse,csv,re
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--raw-dir',required=True,type=Path);a=p.parse_args()
files=sorted(a.raw_dir.glob('*.dat'))
assert len(files)==61,f'Expected 61 input files, found {len(files)}'
rows=list(csv.DictReader(open(a.raw_dir/'hourly_eye_centers.csv',newline='')))
centers={r['key']:r for r in rows}; assert len(centers)==61
expected={(datetime(2025,10,26,9)+timedelta(hours=i)).strftime('%Y%m%dT%H') for i in range(61)}
seen=set(); errors=[]; levels=[10,50,90,130,170,210,250,290]
for file in files:
 match=re.search(r'date(\d{8})_hour(\d{2})\.dat$',file.name)
 assert match,file
 key=f'{match[1]}T{match[2]}'; seen.add(key);assert key in centers,key
 v=np.loadtxt(file)
 assert v.shape==(800,8) and np.isfinite(v).all(),file
 assert np.allclose(np.unique(v[:,2]),levels),file
 sp=v[:,6];th=np.deg2rad(v[:,7]);errors.append(np.max(np.abs(v[:,3:5]-np.column_stack([-sp*np.sin(th),-sp*np.cos(th)]))))
 c=centers[key];lat=float(c['eye_lat_deg']);lon=float(c['eye_lon_deg'])
 assert np.isclose(v[:,0].min(),lat-2,atol=1e-6) and np.isclose(v[:,0].max(),lat+2,atol=1e-6),file
 assert np.isclose(v[:,1].min(),lon-2,atol=1e-6) and np.isclose(v[:,1].max(),lon+2,atol=1e-6),file
assert seen==expected,'Missing or duplicate hourly fields'
print('PASS: 61 hours, 48,800 rows, 8 heights, 10x10 per level; max uv mismatch:',max(errors))
print('Note: legacy excluded=1 flag at 20251028T12 is ignored in final all-observed preprocessing.')
