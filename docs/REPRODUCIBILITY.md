# Reproduction protocol

Run commands from the repository root. Python >= 3.10 is recommended. GPU use is optional for the synthetic audit and necessary for efficient processing in the larger hurricane case. PyTorch wheels must match the machine's CUDA environment when CUDA is requested.

## Setup and preflight

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r melissa/requirements.txt
bash scripts/check_sources.sh
```

## A. Time-dependent synthetic case

1. Verify three frozen weights, metadata, and historical tabulations:
   `bash synthetic/time_dependent/final/scripts/07_verify_history.sh`.
2. Recompute the 50,000-point independent evaluation, using evaluation seed `20260929`:
   `bash synthetic/time_dependent/final/scripts/03_evaluate_virgin.sh`.
3. Rebuild trajectory/field figures and export arrays:
   `bash synthetic/time_dependent/final/scripts/04_figures.sh`.
4. A minimal training smoke test:
   `bash synthetic/time_dependent/final/scripts/01_smoke.sh`.

The synthetic models use 16,384 fixed LHS observations of horizontal `(u,v)` only, observation seed 12345, ResNet width 60/depth 2, 2,000 epochs, 25 batches per epoch. Their training seeds were 0, 1, 2; the seed-0 checkpoint was archived under the observation-density sweep directory. The validation selection criterion is horizontal UV MSE only; independent evaluation uses the 50,000 fresh test points and 20,000 bottom-boundary points. Three-model statistical summaries use sample standard deviation (`ddof=1`).

## B. Hurricane Melissa case

1. Verify raw inputs and SHA-256 integrity of the three archived weights: `bash melissa/scripts/06_verify_history.sh`.
2. Rebuild the complete processed dataset from 61 raw hourly snapshots (if not already present) and recompute the vertical diagnostics without retraining: `bash melissa/scripts/07_recompute_historical.sh`.
3. Recompute the horizontal in-sample fit on each of the **61 assimilated hours** using frozen models (seeds 1234, 2345, 3456): `bash melissa/scripts/09_recompute_uv_historical.sh`.
4. Regenerate final figure PDFs/PNGs: `bash melissa/scripts/08_figures_historical.sh`.

New generated arrays and figures remain in `melissa/data/processed/` and `melissa/outputs/` (ignored by Git). The generated dataset contains 721 time slices at a five-minute spacing; the 660 intermediate slices **must not** be treated as independent measurements.

### Historical numerical comparisons

- Recorded horizontal error (61 assimilated hours), independently recomputed from each checkpoint: `5.77781044 ± 1.11904373 %` relative L2; `0.78146723 ± 0.15135422 m/s` RMSE.
- Historical training summary of physical divergence: approximately `(6.57 ± 0.92) × 10⁻⁶ s⁻¹` across the three runs (separate from derived diagnostic sampling).
- Recomputed kinematic increment correlation against a divergence-derived increment: approximately `0.946 ± 0.017`.
- Relative seed variability in reconstructed interior vertical wind: approximately `12.9 %`.

The divergence-derived increment comparison is a **kinematic consistency diagnostic**, not a comparison against true independently measured vertical wind. Original HTTP payloads and some external hurricane-track provenance details are not preserved. We did **not** perform independent holdout evaluation for the Hurricane Melissa inputs, nor a leave-one-hour-out test.

## Environment and versioning

Record local versions before running new experiments:

```bash
bash scripts/record_environment.sh
```

Reproducing archived checkpoint evaluations is distinct from reproducing stochastic training bit-for-bit. New training is optional, computationally expensive, and may be sensitive to software/hardware versions.
