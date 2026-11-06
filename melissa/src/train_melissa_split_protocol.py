#!/usr/bin/env python3
"""Space--time Melissa reconstruction with a *processed-data* train/val/test split.

The 61 NASA POWER hourly winds are first used by prepare_melissa_hourly.py to
construct one fixed, densified horizontal reference field (40x40x15x721 by
publication defaults). No original hourly snapshot is withheld. Here the
processed space--time grid is partitioned into disjoint supervised-data subsets
at every time level; the same 61-hour interval is represented by all three.

This tests fidelity to the *processed, interpolated reference*, not to held-out
meteorological snapshots, real w, or future atmospheric states. Distinct physics
collocation points are sampled independently and do not need uv labels.

To retain historical outputs, run this program in outputs/split_protocol/models/.
The prior all-observations trainer is left unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from train_melissa_reconstruction_final import (
    ResidualNet, StormFrame, Scaling, TemporalVelocityScale,
    set_seed, temporal_normalize_targets, uv_metrics_on_rows,
    sample_volume, sample_bottom, divergence_residual, physics_metrics,
)


class TimeStratifiedSplit:
    """An exact, reproducible, disjoint split *within each dense time slice*.

    A bijective affine permutation of local spatial indices allocates precisely
    fixed fractions per time level without materializing 17M indices or arrays.
    The per-time offset changes the assignment with time. The construction
    deliberately avoids holding out entire hours (no temporal extrapolation).
    """
    def __init__(self, n_times, n_per_time, train_fraction=0.70,
                 val_fraction=0.15, seed=260926):
        self.nt = int(n_times)
        self.npt = int(n_per_time)
        if self.nt < 2 or self.npt < 10:
            raise ValueError('Need at least 2 time levels and 10 spatial rows per level.')
        if not (0 < train_fraction < 1 and 0 < val_fraction < 1
                and train_fraction + val_fraction < 1):
            raise ValueError('Fractions must satisfy 0<train<1, 0<val<1, train+val<1.')
        ntr = int(round(self.npt * train_fraction))
        nval = int(round(self.npt * val_fraction))
        ntest = self.npt - ntr - nval
        if min(ntr, nval, ntest) <= 0:
            raise ValueError('The split is too small to provide three nonempty sets.')
        self.intervals = {'train': (0, ntr),
                          'val': (ntr, ntr + nval),
                          'test': (ntr + nval, self.npt)}
        rng = np.random.default_rng(seed)
        # Different random bijection at each time level to avoid systematically
        # assigning the same spatial locations to one split at nearby times.
        aa = []
        inverses = []
        for _ in range(self.nt):
            a = int(rng.integers(1, self.npt))
            while math.gcd(a, self.npt) != 1:
                a = int(rng.integers(1, self.npt))
            aa.append(a)
            inverses.append(pow(a, -1, self.npt))
        self.a_of_t = np.asarray(aa, dtype=np.int64)
        self.inv_a_of_t = np.asarray(inverses, dtype=np.int64)
        self.c_of_t = rng.integers(0, self.npt, size=self.nt, dtype=np.int64)
        self.seed = int(seed)
        self.affine_hash = hashlib.sha256(
            self.a_of_t.tobytes() + self.c_of_t.tobytes()
        ).hexdigest()

    def rows(self, partition, rng, number, replace=False):
        """Sample global dense array row IDs exclusively from one partition."""
        low, high = self.intervals[partition]
        per_time = high - low
        possible = self.nt * per_time
        number = int(number)
        if number <= 0 or (not replace and number > possible):
            raise ValueError(f'Invalid sample count {number} for {partition}: {possible} available.')
        if replace:
            code = rng.integers(0, possible, size=number, dtype=np.int64)
        else:
            code = rng.choice(possible, size=number, replace=False)
        t_idx = code // per_time
        q = (code % per_time) + low
        local = (self.inv_a_of_t[t_idx] * ((q - self.c_of_t[t_idx]) % self.npt)) % self.npt
        return t_idx * self.npt + local

    def split_of(self, row):
        """Audit helper: returns train/val/test for each row ID."""
        row = np.asarray(row, dtype=np.int64)
        t = row // self.npt
        s = row % self.npt
        q = (self.a_of_t[t] * s + self.c_of_t[t]) % self.npt
        t1 = self.intervals['train'][1]
        v1 = self.intervals['val'][1]
        return np.where(q < t1, 'train', np.where(q < v1, 'val', 'test'))

    def metadata(self):
        return {
            'method': 'time-stratified affine-bijection of dense grid rows',
            'split_seed': self.seed,
            'times': self.nt,
            'points_per_time': self.npt,
            'rows_total': self.nt * self.npt,
            'per_time': {k: hi - lo for k, (lo, hi) in self.intervals.items()},
            'counts': {k: self.nt * (hi - lo) for k, (lo, hi) in self.intervals.items()},
            'per_time_affine_parameters_sha256': self.affine_hash,
            'all_partitions_span_all_time_levels': True,
            'holdout_unit': 'processed space-time grid row, not original POWER hour',
            'meteorological_independence': False,
        }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--split_seed', type=int, default=260926)
    ap.add_argument('--train_fraction', type=float, default=.70)
    ap.add_argument('--val_fraction', type=float, default=.15)
    ap.add_argument('--validation_seed', type=int, default=260927)
    ap.add_argument('--test_seed', type=int, default=260928)
    ap.add_argument('--physics_test_seed', type=int, default=260929)
    ap.add_argument('--width', type=int, default=80)
    ap.add_argument('--depth', type=int, default=5)
    ap.add_argument('--nEpoch', type=int, default=2500)
    ap.add_argument('--nBatch', type=int, default=25)
    ap.add_argument('--dataBatchSize', type=int, default=512)
    ap.add_argument('--divBatchSize', type=int, default=2048)
    ap.add_argument('--bottomBatchSize', type=int, default=1024)
    ap.add_argument('--valDataSize', type=int, default=24000)
    ap.add_argument('--testDataSize', type=int, default=100000)
    ap.add_argument('--evalDivSize', type=int, default=8192)
    ap.add_argument('--evalBdySize', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=.01)
    ap.add_argument('--weight_decay', type=float, default=.000001)
    ap.add_argument('--step_size', type=int, default=200)
    ap.add_argument('--gamma', type=float, default=.5)
    ap.add_argument('--grad_clip', type=float, default=1.)
    ap.add_argument('--beta_div', type=float, default=.03)
    ap.add_argument('--beta_bdy', type=float, default=1.)
    ap.add_argument('--eval_every', type=int, default=20)
    ap.add_argument('--log_every', type=int, default=20)
    a = ap.parse_args()
    for name in ['nEpoch', 'nBatch', 'dataBatchSize', 'divBatchSize',
                 'bottomBatchSize', 'valDataSize', 'testDataSize',
                 'evalDivSize', 'evalBdySize', 'eval_every', 'log_every']:
        if getattr(a, name) < 1:
            ap.error(f'--{name} must be positive')
    return a


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = Path(args.data_dir)
    with (data_dir / 'manifest.json').open() as f:
        manifest = json.load(f)
    if int(manifest['n_train_snapshots']) != int(manifest['n_power_snapshots_available']):
        raise RuntimeError('Expected all original snapshots in the interpolated data source.')
    if any(manifest.get(k, '') for k in ('dev_key', 'val_key', 'test_key_reserved')):
        raise RuntimeError('Regenerate dense data with --dev_key "" --val_key "" --test_key "".')
    if int(manifest['n_power_snapshots_available']) != 61:
        print('NOTE: a reduced-sized dataset is being used (smoke/development run).')

    X = np.load(data_dir / 'train_X.npy', mmap_mode='r')
    Y = np.load(data_dir / 'train_Yuv.npy', mmap_mode='r')
    generated_times = np.loadtxt(data_dir / 'train_times.csv', delimiter=',',
                                 skiprows=1, ndmin=2)[:, 1]
    nt = len(generated_times)
    if X.ndim != 2 or X.shape[1] != 4 or Y.shape != (len(X), 2):
        raise ValueError(f'Unexpected dense data shapes: X={X.shape}, Y={Y.shape}')
    if len(X) % nt:
        raise ValueError('Dense data are not time-major arrays of equal-sized time slices.')
    npt = len(X) // nt
    splitter = TimeStratifiedSplit(nt, npt, args.train_fraction,
                                   args.val_fraction, args.split_seed)
    split_meta = splitter.metadata()
    with (outdir / 'split_manifest.json').open('w') as f:
        json.dump(split_meta, f, indent=2)

    # Sample masks are implicit and disjoint. The validation sample is held fixed
    # across epochs/seeds; the test sample is opened only after checkpoint selection.
    val_idx = splitter.rows('val', np.random.default_rng(args.validation_seed),
                            args.valDataSize, replace=False)
    assert np.all(splitter.split_of(val_idx) == 'val')
    # Test split is never accessed in the training loop.

    frame = StormFrame(str(data_dir), manifest)
    scaling = Scaling(manifest, frame)
    tscale = TemporalVelocityScale(str(data_dir), manifest)
    t_min, t_max = float(generated_times.min()), float(generated_times.max())
    model = ResidualNet(args.width, args.depth).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    coeffs = torch.tensor((scaling.cx, scaling.cy, scaling.cz),
                          dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    rng_train = np.random.default_rng(args.seed)
    history = []
    best_val = float('inf')
    best_epoch = None
    best_path = outdir / 'best_model.pt'

    print('MELISSA -- PROCESSED SPACE-TIME SPLIT; NOT FORECASTING', flush=True)
    print(f'Device={device}; seed={args.seed}; parameters={nparams}', flush=True)
    print(f'Dense reference={len(X):,} rows, {nt} times, {npt} points/time', flush=True)
    print(f'Exact split counts={split_meta["counts"]}', flush=True)
    print('Original hourly fields contribute to *reference construction*; '
          'supervised test/validation points are excluded from training.', flush=True)
    print('J_uv = mean[(u_bar-u_ref/S(t))^2+(v_bar-v_ref/S(t))^2]', flush=True)
    print('Metric interpretation: reconstruction of processed field; '
          'not independent atmospheric or vertical-w validation.', flush=True)

    start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(args.nEpoch):
        model.train()
        accu = {'loss': 0., 'uv': 0., 'div': 0., 'bdy': 0.}
        for _ in range(args.nBatch):
            rows = splitter.rows('train', rng_train, args.dataBatchSize, replace=True)
            Xearth = np.array(X[rows], dtype=np.float32, copy=True)
            Yglobal = np.array(Y[rows], dtype=np.float32, copy=True)
            Xstorm = frame.earth_to_storm(Xearth)
            Ybar, _ = temporal_normalize_targets(Yglobal, Xstorm[:, 3], tscale)
            Xb = torch.from_numpy(Xstorm).to(device)
            Yb = torch.from_numpy(Ybar).to(device)
            pr = model(Xb)[:, :2]
            loss_uv = ((pr - Yb) ** 2).sum(dim=1).mean()

            Xp = torch.from_numpy(sample_volume(rng_train, args.divBatchSize,
                                                 t_min, t_max)).to(device)
            _, rr = divergence_residual(model, Xp, coeffs, create_graph=True)
            loss_div = (rr ** 2).mean()
            Xbot = torch.from_numpy(sample_bottom(rng_train, args.bottomBatchSize,
                                                   t_min, t_max)).to(device)
            loss_bdy = (model(Xbot)[:, 2] ** 2).mean()
            loss = loss_uv + args.beta_div * loss_div + args.beta_bdy * loss_bdy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for k, v in [('loss', loss), ('uv', loss_uv),
                         ('div', loss_div), ('bdy', loss_bdy)]:
                accu[k] += float(v.item())
        scheduler.step()
        record = {'epoch': epoch+1, 'train_loss': accu['loss']/args.nBatch,
                  'train_uv': accu['uv']/args.nBatch,
                  'train_div': accu['div']/args.nBatch,
                  'train_bdy': accu['bdy']/args.nBatch,
                  'lr': optimizer.param_groups[0]['lr']}
        if (epoch+1) % args.eval_every == 0 or epoch == 0 or epoch == args.nEpoch-1:
            model.eval()
            val = uv_metrics_on_rows(model, frame, X, Y, val_idx, scaling,
                                     tscale, device, batch=8192)
            record.update({f'val_{k}': v for k, v in val.items()})
            if val['mse_bar_pair'] < best_val:
                best_val = val['mse_bar_pair']
                best_epoch = epoch+1
                torch.save(model.state_dict(), best_path)
        history.append(record)
        if (epoch+1) % args.log_every == 0 or epoch == 0 or epoch == args.nEpoch-1:
            vt = record.get('val_mse_bar_pair')
            print(f'Epoch {epoch+1:5d} | train={record["train_loss"]:.4e} '
                  f'uv={record["train_uv"]:.4e} div={record["train_div"]:.4e} '
                  f'bdy={record["train_bdy"]:.4e}' +
                  (f' | val_uv={vt:.5e} best={best_val:.5e} (epoch {best_epoch})'
                   if vt is not None else ''), flush=True)

    train_seconds = time.perf_counter() - start
    if best_epoch is None:
        raise RuntimeError('No validation checkpoint was saved.')
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    # The test is evaluated ONCE, after checkpoint selection. It was not used to
    # choose architecture, epoch, scale, physics weights, or hyperparameters.
    test_idx = splitter.rows('test', np.random.default_rng(args.test_seed),
                             args.testDataSize, replace=False)
    assert np.all(splitter.split_of(test_idx) == 'test')
    assert not np.intersect1d(val_idx, test_idx).size
    test = uv_metrics_on_rows(model, frame, X, Y, test_idx, scaling,
                              tscale, device, batch=8192)

    phy_rng = np.random.default_rng(args.physics_test_seed)
    Xphys = torch.from_numpy(sample_volume(phy_rng, args.evalDivSize,
                                            t_min, t_max)).to(device)
    Xbdy = torch.from_numpy(sample_bottom(phy_rng, args.evalBdySize,
                                           t_min, t_max)).to(device)
    phys = physics_metrics(model, Xphys, Xbdy, scaling, tscale, coeffs)
    max_mem = (int(torch.cuda.max_memory_allocated(device))
               if device.type == 'cuda' else 0)

    summary = {
        'protocol': 'processed-space-time holdout, NOT original-hour holdout',
        'scope': 'inverse reconstruction of w under continuity and bottom boundary; not forecasting',
        'processed_reference_is_built_from_all_original_hours': True,
        'no_independent_meteorological_test_or_w_truth': True,
        'split': split_meta,
        'seeds': {'network_and_collocation': args.seed,
                  'partition': args.split_seed, 'validation_sampling': args.validation_seed,
                  'test_sampling': args.test_seed,
                  'physics_test_sampling': args.physics_test_seed},
        'architecture': {'width': args.width, 'depth': args.depth,
                         'n_parameters': nparams},
        'normalization': {'target': 'u/S(t), v/S(t)',
                          'S_t': 'PCHIP(log S_k) computed from preprocessed original hourly snapshots',
                          'physical_divergence': 'S(t)/Lh * r_div',
                          'Lxi_m': scaling.Lxi, 'Leta_m': scaling.Leta,
                          'Lz_m': scaling.Lz, 'Lh_m': scaling.Lh},
        'loss': {'uv': 'mean[(u_bar-u_ref_bar)^2+(v_bar-v_ref_bar)^2] on TRAIN split only',
                 'beta_div': args.beta_div, 'beta_bdy': args.beta_bdy},
        'training': {'epochs_budget': args.nEpoch, 'batches_per_epoch': args.nBatch,
                     'data_batch': args.dataBatchSize, 'div_batch': args.divBatchSize,
                     'boundary_batch': args.bottomBatchSize,
                     'initial_lr': args.lr, 'step_size': args.step_size,
                     'gamma': args.gamma, 'weight_decay': args.weight_decay,
                     'checkpoint_selection': 'minimum horizontal validation MSE',
                     'best_epoch': best_epoch, 'seconds': train_seconds,
                     'max_gpu_allocated_bytes': max_mem},
        'validation': {'sample_size': len(val_idx), 'best_mse_bar_pair': best_val},
        'test': {'sample_size': len(test_idx), **test},
        'test_physics': {'n_interior': args.evalDivSize,
                         'n_boundary': args.evalBdySize,
                         **phys},
        'model_file': 'best_model.pt',
    }
    with (outdir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    cols = ['epoch', 'train_loss', 'train_uv', 'train_div', 'train_bdy', 'lr',
            'val_mse_bar_pair', 'val_mse_bar_uv', 'val_relL2_uv', 'val_RMSE_uv_mps']
    with (outdir / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in history:
            writer.writerow({k: r.get(k, '') for k in cols})
    print(f'BEST EPOCH: {best_epoch} / {args.nEpoch}', flush=True)
    print(f'TEST processed field: relL2={test["relL2_uv"]:.6%}; '
          f'RMSE={test["RMSE_uv_mps"]:.6f} m/s; '
          f'MSEbar(pair)={test["mse_bar_pair"]:.6e}', flush=True)
    print(f'PHYSICS test: RMS(div)={phys["rms_div_phys_sinv"]:.6e} s^-1; '
          f'RMS(w_bottom)={phys["rms_bdy_phys_mps"]:.6e} m/s', flush=True)
    print(f'Training time={train_seconds:.1f} s; wrote {outdir}', flush=True)

if __name__ == '__main__':
    main()
