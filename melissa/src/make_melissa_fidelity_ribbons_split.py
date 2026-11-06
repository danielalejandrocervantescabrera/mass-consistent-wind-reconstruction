#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication figures for validation-selected Hurricane Melissa checkpoint.

Run from the *melissa/* directory, after copying this script into src/:

    python src/make_melissa_fidelity_ribbons_split.py

Inputs: data/processed/{manifest.json,train_hourly_processed.npz} and
        outputs/split_protocol/models/seed_2345/{summary.json,best_model.pt}.

Outputs (in a SEPARATE directory, to avoid overwriting original figures):
  fig_melissa_horizontal_fidelity.{pdf,png}
  fig_melissa_3d_streamribbons.{pdf,png}
  figure_reproduction_metadata.json

The 3D streamribbons follow instantaneous reconstructed velocity streamlines,
integrated in PHYSICAL meters. Only the displayed z coordinate is exaggerated
by 500. No observed w values are used or claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from make_melissa_paper_figures_split import (
    RitzNet, build_input, decode_key, load_state, read_json,
)


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data/processed')
    p.add_argument('--model-dir', default='outputs/split_protocol/models/seed_2345')
    p.add_argument('--outdir', default='outputs/split_protocol/figures/revised_2345')
    p.add_argument('--snapshot', default='20251028T07', help='Exact hourly key from processed data')
    p.add_argument('--seed', type=int, default=2345)
    p.add_argument('--batch-size', type=int, default=65536)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--z-exaggeration', type=float, default=500.0,
                   help='Affects only 3D plot, NEVER trajectory integration')
    p.add_argument('--ribbon-width-km', type=float, default=5.5)
    p.add_argument('--stream-length-km', type=float, default=260.0,
                   help='Maximum arc length in each integration direction')
    p.add_argument('--stream-step-km', type=float, default=2.5)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return p


def read_inputs(args):
    ddir = Path(args.data_dir)
    npz_path = ddir / 'train_hourly_processed.npz'
    manifest_path = ddir / 'manifest.json'
    model_dir = Path(args.model_dir)
    summary_path = model_dir / 'summary.json'
    weights_path = model_dir / 'best_model.pt'
    for path in (npz_path, manifest_path, summary_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required input does not exist: {path}')

    meta = read_json(manifest_path)
    summary = read_json(summary_path)
    if summary.get('seeds', {}).get('network_and_collocation') != args.seed:
        raise ValueError(f'Seed in summary.json is not {args.seed}')
    best_epoch = int(summary['training']['best_epoch'])
    if args.seed == 2345 and best_epoch != 2500:
        raise ValueError(f'Expected validation-selected epoch 2500; found {best_epoch}')
    common = meta['storm_relative_common_support']
    domain = meta['domain']

    with np.load(npz_path, allow_pickle=False) as d:
        keys = [decode_key(x) for x in d['keys']]
        if args.snapshot not in keys:
            raise ValueError(f'Snapshot {args.snapshot} not present; available: {keys[:4]} ... {keys[-4:]}')
        i = keys.index(args.snapshot)
        hours = np.asarray(d['hours'], dtype=np.float64)
        xi = np.asarray(d['xi_grid_m'], dtype=np.float64)
        eta = np.asarray(d['eta_grid_m'], dtype=np.float64)
        z = np.asarray(d['z_grid_m'], dtype=np.float64)
        # Only the selected original hour is needed.  However, to calculate
        # S_k exactly as in training, we read its full 3D processed field.
        Uref = np.asarray(d['U_phys_mps'][i], dtype=np.float64)
        Vref = np.asarray(d['V_phys_mps'][i], dtype=np.float64)

    if Uref.shape != Vref.shape or Uref.shape != (len(z), len(eta), len(xi)):
        raise ValueError(f'Unexpected U/V shape {Uref.shape}; required (nz,ny,nx)')
    if np.any(np.diff(xi) <= 0) or np.any(np.diff(eta) <= 0) or np.any(np.diff(z) <= 0):
        raise ValueError('Physical coordinate grids must be strictly increasing')
    if not np.all(np.isfinite(Uref)) or not np.all(np.isfinite(Vref)):
        raise ValueError('Reference field contains non-finite values')

    S_real = float(max(np.max(np.abs(Uref)), np.max(np.abs(Vref))))
    Lxi = float(common['xi_max_m'] - common['xi_min_m'])
    Leta = float(common['eta_max_m'] - common['eta_min_m'])
    Lh = math.sqrt(Lxi * Leta)
    Lz = float(domain['Lz_m'])
    if Lz <= 0 or S_real <= 0:
        raise ValueError('Invalid length/velocity scaling')
    # At any ORIGINAL hour, the training scale S(t) equals that hourly S_k.
    # See TemporalVelocityScale in train_melissa_reconstruction_final.py.
    S_t = S_real
    Sw = S_t * Lz / Lh
    t_hat = float(hours[i] / float(domain['total_hours']))
    return dict(xi=xi, eta=eta, z=z, Uref=Uref, Vref=Vref, hour=float(hours[i]),
                t_hat=t_hat, S_t=S_t, Sw=Sw, Lh=Lh, Lz=Lz,
                xi_min=float(common['xi_min_m']), xi_max=float(common['xi_max_m']),
                eta_min=float(common['eta_min_m']), eta_max=float(common['eta_max_m']),
                summary=summary, weights_path=weights_path,
                manifest_path=manifest_path, npz_path=npz_path)


def load_and_predict(data, args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda selected but CUDA is not available')
    device = torch.device('cuda' if (args.device == 'cuda' or
            (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    architecture = data['summary']['architecture']
    model = RitzNet(width=int(architecture['width']), depth=int(architecture['depth'])).to(device)
    model.load_state_dict(load_state(str(data['weights_path']), device), strict=True)
    model.eval()
    nparams = sum(p.numel() for p in model.parameters())
    if nparams != int(architecture['n_parameters']):
        raise RuntimeError(f'Architecture parameter mismatch {nparams} vs summary {architecture["n_parameters"]}')
    X = build_input(data['xi'], data['eta'], data['z'], data['t_hat'],
                    data['xi_min'], data['xi_max'], data['eta_min'],
                    data['eta_max'], data['Lz'])
    norm_out = np.empty((len(X), 3), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(X), args.batch_size):
            b = min(a + args.batch_size, len(X))
            norm_out[a:b] = model(torch.from_numpy(X[a:b]).to(device)).cpu().numpy()
    nz, ny, nx = len(data['z']), len(data['eta']), len(data['xi'])
    norm_out = norm_out.reshape(nz, ny, nx, 3).astype(np.float64)
    U = data['S_t'] * norm_out[..., 0]
    V = data['S_t'] * norm_out[..., 1]
    W = data['Sw'] * norm_out[..., 2]
    if not all(np.all(np.isfinite(arr)) for arr in (U,V,W)):
        raise RuntimeError('Non-finite neural output')
    return U, V, W


def add_label(ax, name):
    ax.text(.025, .975, name, transform=ax.transAxes, va='top', ha='left',
            fontweight='bold', fontsize=11, bbox=dict(facecolor='white', alpha=.75,
            edgecolor='none', pad=1.7))


def save_both(fig, outdir, base, dpi):
    pdf = outdir / f'{base}.pdf'
    png = outdir / f'{base}.png'
    fig.savefig(pdf, bbox_inches='tight', pad_inches=.08)
    fig.savefig(png, dpi=dpi, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    print(f'  {pdf}\n  {png}')


def figure_fidelity(data, U, V, args, outdir):
    xi = data['xi'] / 1000
    eta = data['eta'] / 1000
    k0 = int(np.argmin(abs(data['z'] - 10.0)))
    zval = data['z'][k0]
    Ur, Vr = data['Uref'][k0], data['Vref'][k0]
    Um, Vm = U[k0], V[k0]
    speed_ref = np.hypot(Ur, Vr)
    speed_mod = np.hypot(Um, Vm)
    err = np.hypot(Um - Ur, Vm - Vr)
    vmax = max(float(speed_ref.max()), float(speed_mod.max()))
    emax = max(float(np.percentile(err, 99.5)), 1e-9)
    norm_speed = colors.Normalize(vmin=0, vmax=vmax)
    norm_error = colors.Normalize(vmin=0, vmax=emax)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.65), constrained_layout=True)
    for p, (ax, speed, up, vp) in enumerate(zip(axes, (speed_ref, speed_mod), (Ur, Um), (Vr, Vm))):
        ax.pcolormesh(xi, eta, speed, shading='auto', cmap='turbo', norm=norm_speed,
                      rasterized=True)
        # Arrows show DIRECTION (not a second speed scale); short arrows avoid
        # obscuring the meteorological field and match the original figure.
        ix = np.arange(0, len(xi), max(1, len(xi)//9))
        iy = np.arange(0, len(eta), max(1, len(eta)//9))
        Xq, Yq = np.meshgrid(xi[ix], eta[iy], indexing='xy')
        Uq = up[np.ix_(iy, ix)]
        Vq = vp[np.ix_(iy, ix)]
        magnitude = np.maximum(np.hypot(Uq, Vq), 1e-8)
        arrow_km = min(float(xi[-1]-xi[0]),float(eta[-1]-eta[0])) / 24.0
        ax.quiver(Xq, Yq, arrow_km*Uq/magnitude, arrow_km*Vq/magnitude,
                  angles='xy', scale_units='xy', scale=1, color='#262626',
                  width=.0025, headwidth=3, headlength=4, headaxislength=3.3,
                  alpha=.75, pivot='middle')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xi[0], xi[-1]); ax.set_ylim(eta[0], eta[-1])
        ax.set_xlabel(r'$\xi$ [km]')
        if p == 0: ax.set_ylabel(r'$\eta$ [km]')
        ax.set_title(('Processed horizontal reference', 'Neural horizontal reconstruction')[p], fontsize=10.5)
        add_label(ax, ('(a)', '(b)')[p])

    im_err = axes[2].pcolormesh(xi, eta, err, shading='auto', cmap='viridis',
                                norm=norm_error, rasterized=True)
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].set_xlim(xi[0], xi[-1]); axes[2].set_ylim(eta[0], eta[-1])
    axes[2].set_xlabel(r'$\xi$ [km]')
    axes[2].set_title('Horizontal vector-error magnitude', fontsize=10.5)
    add_label(axes[2], '(c)')

    sm = ScalarMappable(norm=norm_speed, cmap='turbo')
    cb = fig.colorbar(sm, ax=list(axes[:2]), shrink=.88, pad=.018)
    cb.set_label(r'Horizontal speed [m s$^{-1}$]')
    cb2 = fig.colorbar(im_err, ax=axes[2], shrink=.88, pad=.018)
    cb2.set_label(r'Vector error [m s$^{-1}$]')
    fig.suptitle(f'Hurricane Melissa — 28 October 2025, 07:00 UTC; z={zval:g} m; seed {args.seed}',
                 fontsize=12)
    save_both(fig, outdir, 'fig_melissa_horizontal_fidelity', args.dpi)
    relL2 = np.linalg.norm((Um-Ur).ravel())**2 + np.linalg.norm((Vm-Vr).ravel())**2
    den = np.linalg.norm(Ur.ravel())**2 + np.linalg.norm(Vr.ravel())**2
    relL2 = math.sqrt(relL2/den) if den>0 else float('nan')
    rmse = math.sqrt(float(np.mean((Um-Ur)**2 + (Vm-Vr)**2)))
    return {'height_m':float(zval), 'snapshot_horizontal_relL2':relL2,
            'snapshot_horizontal_RMSE_mps':rmse,
            'snapshot_horizontal_vector_error_max_mps':float(np.max(err))}


def make_velocity_evaluator(data, U, V, W):
    axis = (data['z'], data['eta'], data['xi'])
    fields = [RegularGridInterpolator(axis, v, method='linear', bounds_error=False,
                                      fill_value=np.nan) for v in (U, V, W)]
    mins = np.array([axis[2][0], axis[1][0], axis[0][0]], dtype=float)
    maxs = np.array([axis[2][-1], axis[1][-1], axis[0][-1]], dtype=float)

    def evaluate(p):
        p = np.asarray(p, dtype=float)
        if np.any(p < mins) or np.any(p > maxs):
            return None
        sample = np.array([[p[2], p[1], p[0]]])
        vel = np.array([float(it(sample)[0]) for it in fields], dtype=float)
        if not np.all(np.isfinite(vel)):
            return None
        velnorm = float(np.linalg.norm(vel))
        if velnorm <= 1e-7:
            return None
        return (vel/velnorm, float(np.hypot(vel[0],vel[1])))
    return evaluate


def stream_one(seed, direction, evaluator, h_m, max_length_m):
    """RK4 in physical (xi, eta, z) meters, arc-length parametrization."""
    path = [np.array(seed, dtype=float)]
    nsteps = int(math.ceil(max_length_m / h_m))
    for _ in range(nsteps):
        p = path[-1]
        a = evaluator(p)
        if a is None: break
        k1 = direction*a[0]
        b = evaluator(p + .5*h_m*k1)
        if b is None: break
        k2 = direction*b[0]
        c = evaluator(p + .5*h_m*k2)
        if c is None: break
        k3 = direction*c[0]
        d = evaluator(p + h_m*k3)
        if d is None: break
        k4 = direction*d[0]
        q = p + h_m*(k1+2*k2+2*k3+k4)/6
        if evaluator(q) is None: break
        path.append(q)
    return np.vstack(path)


def seeds_inside_domain(data):
    """Deterministic starting sites in storm-relative physical coordinates."""
    xy_min = np.array([data['xi'][0], data['eta'][0]])
    xy_max = np.array([data['xi'][-1], data['eta'][-1]])
    halfwidth = float(min(np.min(xy_max), np.min(-xy_min)))
    radius1 = min(95000., .50*halfwidth)
    radius2 = min(145000., .78*halfwidth)
    bottom = float(data['z'][0]); top = float(data['z'][-1])
    out = []
    for k in range(9):
        phi=2*math.pi*(k+.3)/9
        height = bottom + (.25 if k%2==0 else .69)*(top-bottom)
        out.append([radius1*math.cos(phi), radius1*math.sin(phi),height])
    for k in range(9):
        phi=2*math.pi*(k+.8)/9
        height = bottom + (.43 if k%2==0 else .78)*(top-bottom)
        out.append([radius2*math.cos(phi), radius2*math.sin(phi),height])
    return out


def make_ribbon_quads(curve, zfactor, width_km):
    """Graphics-only ribbons; physical trajectories are ALREADY integrated."""
    disp = curve.copy()
    disp[:, :2] /= 1000
    disp[:, 2] *= zfactor/1000
    quads = []
    for a,b in zip(disp[:-1],disp[1:]):
        tangent = b - a
        horiz = np.hypot(tangent[0],tangent[1])
        if horiz < 1e-12:
            continue
        n = np.array([-tangent[1]/horiz, tangent[0]/horiz, 0.]) * (width_km/2)
        quads.append([a-n, a+n, b+n, b-n])
    return quads


def figure_ribbons(data, U, V, W, args, outdir):
    evaluator = make_velocity_evaluator(data, U,V,W)
    starting_points = seeds_inside_domain(data)
    cmap = plt.get_cmap('turbo')
    horizontal_speed = np.hypot(U,V)
    limit = float(np.percentile(horizontal_speed, 99.5))
    norm = colors.Normalize(vmin=0, vmax=max(limit,1e-8))
    fig = plt.figure(figsize=(11.1,8.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    accepted = 0
    for seed in starting_points:
        backward = stream_one(seed, -1, evaluator,
                              args.stream_step_km*1000, args.stream_length_km*1000)
        forward = stream_one(seed, +1, evaluator,
                             args.stream_step_km*1000, args.stream_length_km*1000)
        curve = np.vstack([backward[::-1][:-1], forward])
        if len(curve) < 8:
            continue
        quads = make_ribbon_quads(curve, args.z_exaggeration, args.ribbon_width_km)
        if not quads: continue
        cvals=[]
        for p in .5*(curve[:-1]+curve[1:]):
            result = evaluator(p)
            cvals.append(float('nan') if result is None else result[1])
        # Each quad receives local horizontal wind-speed color.
        rgba = [cmap(norm(s if np.isfinite(s) else 0)) for s in cvals]
        coll=Poly3DCollection(quads, facecolors=rgba, alpha=.86,
                              edgecolor='none', linewidths=0., zsort='average')
        ax.add_collection3d(coll)
        accepted += 1

    if accepted == 0:
        plt.close(fig)
        raise RuntimeError('No streamribbons could be constructed. Check bounds/grid/velocity scaling.')
    ax.set_xlim(data['xi'][0]/1000, data['xi'][-1]/1000)
    ax.set_ylim(data['eta'][0]/1000, data['eta'][-1]/1000)
    ax.set_zlim(data['z'][0]*args.z_exaggeration/1000,
                data['z'][-1]*args.z_exaggeration/1000)
    ax.set_xlabel(r'$\xi$ [km]',labelpad=10)
    ax.set_ylabel(r'$\eta$ [km]',labelpad=10)
    ax.set_zlabel(r'$z$ [m]',labelpad=10)
    ticks_m = np.array([10.,50.,100.,150.,200.,250.,290.])
    ticks_m = ticks_m[(ticks_m>=data['z'][0]) & (ticks_m<=data['z'][-1])]
    ax.set_zticks(ticks_m*args.z_exaggeration/1000)
    ax.set_zticklabels([f'{v:g}' for v in ticks_m])
    ax.set_box_aspect((1.16,1.16,.70))
    ax.view_init(elev=26, azim=-68)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        try: axis.pane.set_facecolor((1,1,1,0))
        except AttributeError: pass
    ax.grid(True, alpha=.22)
    sm = ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    cb=fig.colorbar(sm, ax=ax, shrink=.63, pad=.07, aspect=23)
    cb.set_label(r'Horizontal speed [m s$^{-1}$]')
    ax.set_title('Hurricane Melissa — 28 October 2025, 07:00 UTC\n'
                 f'Instantaneous reconstructed streamribbons; seed {args.seed}; '
                 f'displayed $z$ $\\times${args.z_exaggeration:g}', pad=18, fontsize=11)
    save_both(fig, outdir, 'fig_melissa_3d_streamribbons', args.dpi)
    return {'ribbons_displayed':accepted,
            'vertical_display_factor':args.z_exaggeration,
            'streamline_integration':'RK4 in physical (xi,eta,z) meters; no z exaggeration',
            'ribbon_width_km_illustrative':args.ribbon_width_km}


def main():
    args=argument_parser().parse_args()
    if args.z_exaggeration <= 0 or args.stream_step_km <= 0 or args.stream_length_km <= 0:
        raise ValueError('Plot exaggeration and streamline integration lengths must be positive')
    data=read_inputs(args)
    print(f'Loaded {args.snapshot}, hour {data["hour"]:g}, scale S(t)={data["S_t"]:.6f} m/s')
    print(f'Validation-selected epoch {data["summary"]["training"]["best_epoch"]} '
          f'for seed {args.seed}.')
    U,V,W=load_and_predict(data,args)
    outdir=Path(args.outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    print('Generating horizontal fidelity figure:')
    hstats=figure_fidelity(data,U,V,args,outdir)
    print('Generating three-dimensional streamribbon figure:')
    rstats=figure_ribbons(data,U,V,W,args,outdir)
    hash_md=hashlib.sha256(data['weights_path'].read_bytes()).hexdigest()
    details={'source':'validation-selected processed-space-time split checkpoint',
             'checkpoint':str(data['weights_path']), 'checkpoint_sha256':hash_md,
             'seed':args.seed, 'epoch':int(data['summary']['training']['best_epoch']),
             'snapshot_key':args.snapshot,'hour_from_start':data['hour'],
             'time_normalized':data['t_hat'],'S_t_mps':data['S_t'],
             'vertical_scale_Sw_mps':data['Sw'],
             'grid':{'xi_n':len(data['xi']),'eta_n':len(data['eta']),'z_n':len(data['z'])},
             **hstats,**rstats}
    metadata_path=outdir/'figure_reproduction_metadata.json'
    metadata_path.write_text(json.dumps(details,indent=2)+'\n',encoding='utf8')
    print(f'  {metadata_path}')
    print('Completed without modifying the existing manuscript figure files.')

if __name__=='__main__':
    main()
