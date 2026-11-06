
## Hurricane Melissa: current manuscript protocol

The manuscript reports a time-stratified 70/15/15
train/validation/test protocol on processed NASA POWER
horizontal-wind fields, evaluated across three
independent network seeds.

The mean held-out horizontal relative error is
**6.48 +/- 1.93%**.

The unobserved vertical component is assessed through
inter-seed stability and continuity-based diagnostics.

See [Melissa split protocol](melissa/docs/SPLIT_PROTOCOL.md)
for experimental details, checkpoints, and limitations.

---

# Mass-Consistent Wind Field Reconstruction

**Reproducibility materials accompanying a research manuscript (under review).**

This repository implements *reconstruction* of three-dimensional wind fields, including a vertical velocity component that is **not directly observed**. The model fits observed horizontal components `(u, v)` while penalizing spatial mass-continuity residuals and the lower-boundary condition on `w`. **It is not a weather-forecasting system.**

Two time-dependent reconstruction experiments are included: a moving synthetic vortex with known ground truth, and the October 26–28, 2025 Hurricane Melissa case. The hurricane workflow extends the time-dependent synthetic formulation by adding a time coordinate to the reconstruction and enforcing spatial divergence at fixed time.

## Contents

| Location | Description |
|---|---|
| `synthetic/time_dependent/final/` | Final synthetic space-time protocol, three frozen checkpoints, independent test and figure scripts. |
| `synthetic/example1/` | Steady synthetic experiment, density/architecture studies and historical tables. |
| `synthetic/baselines/sasaki/` | Classical comparator and numerical conditioning experiments. |
| `melissa/data/raw/` | 61 archived hourly NASA POWER-derived wind-field files and hourly eye centers used for the reported reconstruction. |
| `melissa/reference/checkpoints/` | Three frozen final hurricane models (seeds 1234, 2345, 3456), with SHA-256 records. |
| `melissa/scripts/` | Data processing, evaluation, independent horizontal-fit recomputation and figures. |
| `docs/` | Reproduction instructions, provenance limitations and publication checklist. |

The trained checkpoints are small enough to be tracked directly in Git. Derived data under `outputs/` and `melissa/data/processed/` are normally ignored. The processed reference, checkpoints, and results associated with the validation-selected 70/15/15 protocol are explicitly tracked in Git.

## Quick start on Linux / Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r melissa/requirements.txt
bash scripts/check_sources.sh
```

### Synthetic space-time experiment (no retraining)

```bash
bash synthetic/time_dependent/final/scripts/07_verify_history.sh
bash synthetic/time_dependent/final/scripts/03_evaluate_virgin.sh
bash synthetic/time_dependent/final/scripts/04_figures.sh
```

The frozen models reproduce the **50,000-point independent synthetic evaluation**. The time-dependent synthetic benchmark has an analytic reference solution for all `(u,v,w)` components.

### Hurricane Melissa: historical in-sample experiment (no retraining)

```bash
bash melissa/scripts/06_verify_history.sh
bash melissa/scripts/07_recompute_historical.sh
bash melissa/scripts/09_recompute_uv_historical.sh
bash melissa/scripts/08_figures_historical.sh
```

The 61 hourly source fields are preprocessed to 721 five-minute instants (40 × 40 × 15 points per instant), producing 17,304,000 space-time rows. This processing step can require significant disk space and RAM. The UV script calculates fitting errors at the **61 assimilated observation times**, not an independent temporal holdout. The diagnostic reconstruction of `w` is assessed for kinematic consistency and stability across three seeds; **no independently measured vertical velocity dataset was available for validation**. Interpolated times do not constitute new independent observations.

## Reproduced numerical results

| Evaluation | Mean ± sample standard deviation, 3 models |
|---|---|
| Synthetic time-dependent independent test: relative L2 `(u,v,w)` | `(4.927 ± 0.366) × 10⁻³` |
| Synthetic time-dependent independent test: RMS divergence | `(2.117 ± 0.141) × 10⁻³` |
| Melissa in-sample hourly horizontal relative L2 `(u,v)` | `5.77781 ± 1.11904 %` |
| Melissa in-sample hourly horizontal RMSE `(u,v)` | `0.78147 ± 0.15135 m/s` |

For the hurricane, the mean observed-hour horizontal errors above were independently recomputed from the archived weights and the processed input fields; their numerical agreement with the historical training summaries was confirmed. The synthetic independent evaluation was also rerun from the archived weights. Details of what is independently tested versus reused from training are in [the reproducibility guide](docs/REPRODUCIBILITY.md).

## Scientific scope and limitations

The hurricane case uses 61 hourly inputs; five-minute intermediate fields result from temporal interpolation (PCHIP). Horizontal fit at the assimilated hours is **in-sample**. Consistency between the reconstructed vertical velocity and velocity inferred from horizontal divergence is **not validation against measured `w`**. A leave-one-hour-out experiment was not performed. Original raw service HTTP responses and the primary source document for some hurricane-track reference points are not preserved; the numerical input files and reconstruction scripts are supplied.

NASA POWER data citation and provenance guidance are provided in [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md). For Spanish-language instructions see [README_ES.md](README_ES.md).

**Licensing:** No software redistribution license has yet been selected; do not interpret public availability as authorization to redistribute or relicense any code or third-party data. The maintainers should add a suitable license after reviewing ownership and attribution obligations.
