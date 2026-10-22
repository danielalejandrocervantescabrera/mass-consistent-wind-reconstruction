#!/usr/bin/env python3
"""Corrected/self-contained Sasaki-RBF baseline for DeepWind Example I.

Exact field on Omega=[-7,7]x[-7,7]x[0,7]:
    u = (2 y exp(-r^2/49) - eps*x*z/2,
        -2 x exp(-r^2/49) - eps*y*z/2,
         eps*z^2/2),
where r^2=x^2+y^2+z^2.

Background used by the classical mass-consistent adjustment:
    u0 = (u_exact, v_exact, 0).
Its divergence is -eps*z, so lambda solves Delta(lambda)=eps*z.
The corrected field is u_S = u0 + grad(lambda).

Boundary treatment follows the historical good.py implementation:
  * zero correction normal flux on bottom and lateral faces;
  * open top boundary lambda=0.

Modernized only for reproducible/fair metrics:
  * vectorized IMQ-RBF matrices;
  * MSE(u,v), MSE(u,v,w), relative L2;
  * RMS(div), not signed mean(div);
  * RMS bottom w (target is zero);
  * fixed random test seed and CSV summary;
  * Gram condition number and linear-system residual are reported.
"""

import argparse
import csv
import os
import time
import numpy as np


def exact_field(xyz: np.ndarray, epsilon: float) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r2 = x*x + y*y + z*z
    ga = np.exp(-r2 / 49.0)
    out = np.empty((xyz.shape[0], 3), dtype=np.float64)
    out[:, 0] =  2.0*y*ga - 0.5*epsilon*x*z
    out[:, 1] = -2.0*x*ga - 0.5*epsilon*y*z
    out[:, 2] =  0.5*epsilon*z*z
    return out


def background_field(xyz: np.ndarray, epsilon: float) -> np.ndarray:
    out = exact_field(xyz, epsilon)
    out[:, 2] = 0.0
    return out


def mse(a, b):
    return float(np.mean((a-b)**2))


def rms(a):
    return float(np.sqrt(np.mean(a*a)))


def rel_l2(a, b):
    return float(np.linalg.norm(a-b) / (np.linalg.norm(b) + 1e-14))


def uniform_regions(bounds, n_centers_target: int):
    a,b,c,d,e,f = bounds
    n = max(3, int(np.ceil(n_centers_target ** (1.0/3.0))))
    xs=np.linspace(a,b,n); ys=np.linspace(c,d,n); zs=np.linspace(e,f,n)
    interior=np.array([[x,y,z] for z in zs[1:-1] for y in ys[1:-1] for x in xs[1:-1]], dtype=np.float64)
    bottom=np.array([[x,y,e] for y in ys for x in xs], dtype=np.float64)
    top=np.array([[x,y,f] for y in ys for x in xs], dtype=np.float64)
    V1=np.array([[a,y,z] for z in zs[1:-1] for y in ys], dtype=np.float64)
    V2=np.array([[b,y,z] for z in zs[1:-1] for y in ys], dtype=np.float64)
    V3=np.array([[x,c,z] for z in zs[1:-1] for x in xs[1:-1]], dtype=np.float64)
    V4=np.array([[x,d,z] for z in zs[1:-1] for x in xs[1:-1]], dtype=np.float64)
    all_pts=np.vstack([interior,V1,V2,V3,V4,bottom,top])
    if all_pts.shape[0] != n**3:
        raise RuntimeError(f"center partition error: {all_pts.shape[0]} != {n**3}")
    return dict(n_side=n, interior=interior, V1=V1, V2=V2, V3=V3, V4=V4, bottom=bottom, top=top, all=all_pts)


def deltas(dsites, ctrs):
    D=dsites[:,None,:]-ctrs[None,:,:]
    return D, np.sum(D*D,axis=2)


def imq(ep, dsites, ctrs):
    _,r2=deltas(dsites,ctrs)
    return 1.0/np.sqrt(1.0+ep*ep*r2)


def grad_imq(ep, dsites, ctrs):
    D,r2=deltas(dsites,ctrs)
    A=1.0+ep*ep*r2
    fac=-(ep*ep)/(A**1.5)
    return D*fac[:,:,None]


def second_imq(ep, dsites, ctrs):
    D,r2=deltas(dsites,ctrs)
    A=1.0+ep*ep*r2
    common=(ep*ep)/(A**2.5)
    dxx=common*(3.0*ep*ep*D[:,:,0]**2-A)
    dyy=common*(3.0*ep*ep*D[:,:,1]**2-A)
    dzz=common*(3.0*ep*ep*D[:,:,2]**2-A)
    return dxx,dyy,dzz


def lap_matrix(ep, dsites, ctrs):
    a,b,c=second_imq(ep,dsites,ctrs)
    return a+b+c


def neumann_matrix(ep, normal, dsites, ctrs):
    G=grad_imq(ep,dsites,ctrs)
    return G @ np.asarray(normal,dtype=np.float64)


def build_system(n_centers, ep, epsilon):
    bounds=[-7.,7.,-7.,7.,0.,7.]
    R=uniform_regions(bounds,n_centers)
    ctrs=R['all']
    E=lap_matrix(ep,R['interior'],ctrs)
    NL=neumann_matrix(ep,[-1,0,0],R['V1'],ctrs)
    NR=neumann_matrix(ep,[ 1,0,0],R['V2'],ctrs)
    NBa=neumann_matrix(ep,[0,-1,0],R['V3'],ctrs)
    NF=neumann_matrix(ep,[0, 1,0],R['V4'],ctrs)
    NB=neumann_matrix(ep,[0,0,-1],R['bottom'],ctrs)
    # Historical good.py uses an open top boundary: lambda=0.
    NT=imq(ep,R['top'],ctrs)
    gram=np.vstack([E,NL,NR,NBa,NF,NB,NT])
    rhs=np.concatenate([
        epsilon*R['interior'][:,2],
        np.zeros(R['V1'].shape[0]), np.zeros(R['V2'].shape[0]),
        np.zeros(R['V3'].shape[0]), np.zeros(R['V4'].shape[0]),
        np.zeros(R['bottom'].shape[0]), np.zeros(R['top'].shape[0])
    ])
    if gram.shape != (ctrs.shape[0],ctrs.shape[0]):
        raise RuntimeError(f"non-square system {gram.shape}")
    return gram,rhs,ctrs,R['n_side']


def eval_grad(ep, xyz, ctrs, coeff, chunk=2500):
    out=np.empty((len(xyz),3),dtype=np.float64)
    for i in range(0,len(xyz),chunk):
        q=xyz[i:i+chunk]; G=grad_imq(ep,q,ctrs)
        out[i:i+len(q),0]=G[:,:,0]@coeff
        out[i:i+len(q),1]=G[:,:,1]@coeff
        out[i:i+len(q),2]=G[:,:,2]@coeff
    return out


def eval_lap(ep, xyz, ctrs, coeff, chunk=2500):
    out=np.empty(len(xyz),dtype=np.float64)
    for i in range(0,len(xyz),chunk):
        q=xyz[i:i+chunk]; a,b,c=second_imq(ep,q,ctrs)
        out[i:i+len(q)]=(a+b+c)@coeff
    return out


def main():
    ap=argparse.ArgumentParser(description='Corrected Sasaki/RBF baseline for DeepWind Example I')
    ap.add_argument('--nCenters',type=int,default=100,help='target number of tensor-grid RBF centers; rounded up to n_side^3')
    ap.add_argument('--ep',type=float,default=0.01,help='IMQ shape parameter; historical good.py used 0.01')
    ap.add_argument('--epsilon',type=float,default=0.1)
    ap.add_argument('--nTest',type=int,default=10000)
    ap.add_argument('--nBottomEval',type=int,default=5000)
    ap.add_argument('--evalSeed',type=int,default=54321)
    ap.add_argument('--chunk',type=int,default=2500)
    ap.add_argument('--outdir',default='out_sasaki_example1')
    ap.add_argument('--tag',default='sasaki_example1')
    args=ap.parse_args()

    os.makedirs(args.outdir,exist_ok=True)
    sd=os.path.join(args.outdir,'summary'); pd=os.path.join(args.outdir,'predictions')
    os.makedirs(sd,exist_ok=True); os.makedirs(pd,exist_ok=True)

    t=time.time(); gram,rhs,ctrs,n_side=build_system(args.nCenters,args.ep,args.epsilon); build_seconds=time.time()-t
    t=time.time(); coeff=np.linalg.solve(gram,rhs); solve_seconds=time.time()-t
    cond=float(np.linalg.cond(gram))
    sysres=float(np.linalg.norm(gram@coeff-rhs)/(np.linalg.norm(rhs)+1e-14))

    rng=np.random.RandomState(args.evalSeed)
    X=rng.uniform([-7.,-7.,0.],[7.,7.,7.],size=(args.nTest,3))
    Ue=exact_field(X,args.epsilon)
    grad=eval_grad(args.ep,X,ctrs,coeff,args.chunk)
    Ua=background_field(X,args.epsilon)+grad
    lap=eval_lap(args.ep,X,ctrs,coeff,args.chunk)
    div=-args.epsilon*X[:,2]+lap

    Xb=np.column_stack([rng.uniform(-7,7,args.nBottomEval),rng.uniform(-7,7,args.nBottomEval),np.zeros(args.nBottomEval)])
    wb=eval_grad(args.ep,Xb,ctrs,coeff,args.chunk)[:,2]

    m_uv=mse(Ua[:,:2],Ue[:,:2]); m_uvw=mse(Ua,Ue); rel=rel_l2(Ua,Ue)
    rd=rms(div); rb=rms(wb)

    pred_path=os.path.join(pd,f'{args.tag}_predictions.csv')
    np.savetxt(pred_path,np.column_stack([X,Ue,Ua]),delimiter=',',header='x,y,z,u_exact,v_exact,w_exact,u_sasaki,v_sasaki,w_sasaki',comments='')
    sum_path=os.path.join(sd,f'{args.tag}_eval_summary.csv')
    fields=['method','epsilon','ep','nCenters_requested','nCenters_actual','n_side','evalSeed','nTest','test_mse_uv','test_mse_uvw','test_relL2_field','test_rms_div','test_rms_bottom_bc','gram_condition','system_rel_residual','build_seconds','solve_seconds','prediction_file']
    row=dict(method='sasaki_rbf',epsilon=args.epsilon,ep=args.ep,nCenters_requested=args.nCenters,nCenters_actual=len(ctrs),n_side=n_side,evalSeed=args.evalSeed,nTest=args.nTest,test_mse_uv=m_uv,test_mse_uvw=m_uvw,test_relL2_field=rel,test_rms_div=rd,test_rms_bottom_bc=rb,gram_condition=cond,system_rel_residual=sysres,build_seconds=build_seconds,solve_seconds=solve_seconds,prediction_file=pred_path)
    with open(sum_path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerow(row)

    print('=== Sasaki/RBF Example I ===')
    print(f'centers requested/actual = {args.nCenters}/{len(ctrs)} (n_side={n_side})')
    print(f'ep = {args.ep}, epsilon = {args.epsilon}')
    print(f'cond(Gram) = {cond:.6e}, system residual = {sysres:.6e}')
    print(f'MSE(u,v) = {m_uv:.6e}')
    print(f'MSE(u,v,w) = {m_uvw:.6e}')
    print(f'relL2 = {rel:.6e}')
    print(f'RMS(div) = {rd:.6e}')
    print(f'RMS(bottom w) = {rb:.6e}')
    print(f'summary = {sum_path}')

if __name__=='__main__':
    main()
