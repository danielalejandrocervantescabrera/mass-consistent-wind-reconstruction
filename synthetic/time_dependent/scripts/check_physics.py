#!/usr/bin/env python3
"""Numerically verify divergence zero and bottom w=0 for the analytic vortex.
No training involved; this is a formulation check, not a paper reproduction.
"""
import importlib.util
from pathlib import Path
import numpy as np
script = Path(__file__).resolve().parents[1] / 'src/time_dependent_vortex_reference.py'
spec = importlib.util.spec_from_file_location('vortex_ref',script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
p = dict(traj='parabolic',x0=0,y0=0,vx=20,vy=0,ax=0,ay=30,epsilon=0.02)
rng = np.random.default_rng(20261022)
X = rng.uniform([-20,-20,0.2,-1],[20,20,6.8,1],size=(500,4))
h=1e-3
v=mod.uexact_time(X,p).numpy()
div=np.zeros(len(X))
for i in range(3):
    Xp=X.copy();Xm=X.copy();Xp[:,i]+=h;Xm[:,i]-=h
    vp=mod.uexact_time(Xp,p).numpy()
    vm=mod.uexact_time(Xm,p).numpy()
    div += (vp[:,i]-vm[:,i])/(2*h)
Xb=X.copy(); Xb[:,2]=0.0
wb=mod.uexact_time(Xb,p).numpy()[:,2]
err = np.max(np.abs(div))
assert err < 2e-5, f'divergence error too high: {err}'
assert np.max(np.abs(wb)) < 1e-12
print(f'PASS exact-field divergence (central difference): max|div|={err:.3g}')
print('PASS lower-boundary w(z=0)=0')
