#!/usr/bin/env python3
"""Recompute Melissa UV fit on all 61 assimilated hourly anchors from frozen weights.

Run from the root of reconstruccion_viento_reproducible_v5:
    python3 /path/to/evaluate_melissa_uv_historico.py

This is an in-sample fit diagnostic, NOT validation on independent observations,
and NOT a forecast. No training or checkpoint changes occur.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def run(repo_root: Path, output_dir: Path, batch: int, device_string: str):
    root = repo_root / 'melissa'
    source_dir = root / 'src'
    sys.path.insert(0, str(source_dir))
    from train_melissa_reconstruction_final import (
        ResidualNet, StormFrame, Scaling, TemporalVelocityScale,
        load_real_times, load_generated_times, classify_time_slices,
        uv_metrics_on_rows,
    )

    data_dir = root / 'data' / 'processed'
    required = ['manifest.json', 'train_X.npy', 'train_Yuv.npy',
                'train_hourly_processed.npz', 'train_times.csv', 'eye_path.npz']
    missing = [str(data_dir / n) for n in required if not (data_dir / n).is_file()]
    if missing:
        raise SystemExit('Prepared data not found:\n' + '\n'.join(missing) +
                         '\nFirst run: bash melissa/scripts/07_recompute_historical.sh')
    with open(data_dir / 'manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)

    if int(manifest.get('n_train_snapshots', -1)) != 61:
        raise RuntimeError('Expected all 61 observed hourly snapshots in the processed dataset')

    # Memory mapping avoids loading the entire 17,304,000-row dataset into RAM.
    X = np.load(data_dir / 'train_X.npy', mmap_mode='r')
    Y = np.load(data_dir / 'train_Yuv.npy', mmap_mode='r')
    if X.shape[0] != Y.shape[0]:
        raise RuntimeError(f'Mismatched X, Y: {X.shape}, {Y.shape}')

    frame = StormFrame(str(data_dir), manifest)
    scaling = Scaling(manifest, frame)
    tscale = TemporalVelocityScale(str(data_dir), manifest)
    real_t = load_real_times(str(data_dir), manifest)
    generated_t = load_generated_times(str(data_dir))
    real_indices, interp_indices = classify_time_slices(generated_t, real_t)
    if len(real_indices) != 61 or len(generated_t) != 721:
        raise RuntimeError(f'Expected 61 real and 721 total times, got {len(real_indices)} and {len(generated_t)}')
    if X.shape[0] % len(generated_t) != 0:
        raise RuntimeError('Data rows not divisible by number of generated time slices')
    npts = X.shape[0] // len(generated_t)
    if npts != 40 * 40 * 15:
        raise RuntimeError(f'Expected 24000 points per time, got {npts}')

    offsets = np.arange(npts, dtype=np.int64)
    rows = (real_indices[:, None].astype(np.int64) * npts + offsets[None, :]).reshape(-1)
    device = torch.device(device_string)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    print(f'Device: {device}; processed rows: {len(X):,}; observed-hour rows: {len(rows):,}')
    print('Computing horizontal fit only at original 61 hourly anchor times (no retraining).')
    print('These hours were assimilated during training; results are not independent validation.\n')

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in (1234, 2345, 3456):
        checkpoint = root / 'reference' / 'checkpoints' / f'seed_{seed}' / 'final_model.pt'
        summary_file = root / 'reference' / 'checkpoints' / f'seed_{seed}' / 'summary.json'
        if not checkpoint.is_file() or not summary_file.is_file():
            raise FileNotFoundError(f'Missing model or summary for seed {seed}')
        with open(summary_file, encoding='utf-8') as f:
            summary = json.load(f)
        arch = summary['architecture']
        model = ResidualNet(width=int(arch['width']), depth=int(arch['depth'])).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()

        metric = uv_metrics_on_rows(
            model, frame, X, Y, rows, scaling, tscale, device, batch=batch,
        )
        old = summary['full_data_uv_metrics']['real']
        rel, rmse = metric['relL2_uv'], metric['RMSE_uv_mps']
        prev_rel, prev_rmse = old['relL2_uv'], old['RMSE_uv_mps']
        print(f'seed={seed} | recomputed relL2_uv={100*rel:.8f}% (stored={100*prev_rel:.8f}%) | '
              f'RMSE_uv={rmse:.8f} m/s (stored={prev_rmse:.8f} m/s)')
        results.append({
            'seed': seed, 'relL2_uv': rel, 'RMSE_uv_mps': rmse,
            'mse_bar_uv': metric['mse_bar_uv'],
            'stored_relL2_uv': prev_rel, 'stored_RMSE_uv_mps': prev_rmse,
            'delta_relL2_uv': rel - prev_rel,
            'delta_RMSE_uv_mps': rmse - prev_rmse,
        })
        del model

    rel = np.asarray([r['relL2_uv'] for r in results], dtype=np.float64)
    rmse = np.asarray([r['RMSE_uv_mps'] for r in results], dtype=np.float64)
    print('\nRECOMPUTED, MEAN ± SAMPLE STD (ddof=1), over 3 original models:')
    print(f'  observed-hour relL2_uv: {100*rel.mean():.8f} ± {100*rel.std(ddof=1):.8f} %')
    print(f'  observed-hour RMSE_uv:  {rmse.mean():.8f} ± {rmse.std(ddof=1):.8f} m/s')

    csv_path = output_dir / 'observed_hour_uv_recomputed.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = output_dir / 'observed_hour_uv_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'scope': 'in-sample horizontal fit at 61 assimilated observed hourly anchors',
            'no_training_performed': True,
            'rows_evaluated_per_seed': int(len(rows)),
            'seeds': results,
            'mean_relL2_uv_pct': float(100 * rel.mean()),
            'std_relL2_uv_pct': float(100 * rel.std(ddof=1)),
            'mean_RMSE_uv_mps': float(rmse.mean()),
            'std_RMSE_uv_mps': float(rmse.std(ddof=1)),
        }, f, indent=2)
    print(f'\nSaved {csv_path}\nSaved {json_path}')
    diffs_rel = max(abs(r['delta_relL2_uv']) for r in results)
    diffs_rmse = max(abs(r['delta_RMSE_uv_mps']) for r in results)
    if diffs_rel < 5e-4 and diffs_rmse < 1e-2:
        print(f'PASS: reconstructed and archived UV metrics agree within 0.0005 relL2 and 0.01 m/s RMSE.')
    else:
        print(f'WARNING: values differ; investigate processing/code versions: max delta relL2={diffs_rel:.3e}, RMSE={diffs_rmse:.3e}')
        raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=Path('.'),
                   help='Root of reconstruccion_viento_reproducible_v5 (default: current directory)')
    p.add_argument('--outdir', type=Path, default=None)
    p.add_argument('--batch', type=int, default=32768)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    root = args.repo_root.resolve()
    out = args.outdir or root / 'melissa' / 'outputs' / 'historical_analysis' / 'uv_recomputed'
    run(root, out, args.batch, args.device)


if __name__ == '__main__':
    main()
