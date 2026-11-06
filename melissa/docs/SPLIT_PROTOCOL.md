# Hurricane Melissa: validation-selected reconstruction

This directory documents the Hurricane Melissa
experiments reported in the manuscript:

"A mass-consistent neural variational reconstruction
method for three-dimensional wind fields."

## 1. Meteorological input

The experiment uses 61 hourly NASA POWER wind fields
covering 26 October 2025, 09:00 UTC, through
28 October 2025, 21:00 UTC.

The original hourly inputs are archived in the
repository.

The native meteorological resolution is approximately
0.5 x 0.625 degrees. The processed reference therefore
represents spatially smoothed atmospheric circulation,
not independent measurements at every query point.

## 2. Reconstruction problem

The neural model reconstructs three velocity components:

    (u, v, w)

Only horizontal components (u, v) are provided as
observations.

The vertical component w is constrained indirectly
through:

- Spatial mass conservation.
- A lower-boundary condition w = 0.
- The continuous neural representation.

The method performs reconstruction, not forecasting.

## 3. Experimental protocol

A spatially and temporally densified horizontal reference
is constructed from the 61 original hourly fields.

The processed space-time records are divided into
three disjoint subsets:

- Training:   70%
- Validation: 15%
- Test:       15%

The split does not represent independent meteorological
hours or an independent observing system.

The three network initialization seeds are:

    1234
    2345
    3456

For each seed, the training checkpoint is selected using
horizontal validation error only.

Vertical velocity is never supplied as a training or
checkpoint-selection target.

## 4. Held-out test results

Results are mean +/- sample standard deviation
over the three validation-selected models.

Horizontal relative L2 error:
    6.48 +/- 1.93 %

Horizontal RMSE:
    0.876 +/- 0.261 m/s

Normalized horizontal MSE per component:
    (7.97 +/- 4.29) x 10^-4

Divergence RMS:
    (5.39 +/- 1.25) x 10^-6 1/s

Lower-boundary vertical velocity RMS:
    (9.29 +/- 2.10) x 10^-5 m/s

These results refer to held-out records of the
processed horizontal reference.

## 5. Vertical-velocity diagnostics

Vertical velocity is not independently observed.

The reconstructed vertical fields are compared across
the three training seeds.

The relative RMS ensemble spread is 18.2%.

A complementary continuity-based diagnostic compares
reconstructed vertical-velocity increments against
increments derived from the processed horizontal
reference.

Correlation:
    0.920 +/- 0.033

Increment RMSE:
    (6.95 +/- 1.40) x 10^-3 m/s

These diagnostics assess stability and kinematic
compatibility; they do not establish accuracy against
independent atmospheric vertical-velocity measurements.

## 6. Archived artifacts

Source scripts:
    melissa/src/

Processed horizontal reference:
    melissa/data/processed/train_hourly_processed.npz

Processing metadata:
    melissa/data/processed/manifest.json

Validation-selected checkpoints:
    melissa/outputs/split_protocol/models/seed_1234/
    melissa/outputs/split_protocol/models/seed_2345/
    melissa/outputs/split_protocol/models/seed_3456/

Three-seed summaries:
    melissa/outputs/split_protocol/models/three_seed_results.csv
    melissa/outputs/split_protocol/models/three_seed_summary.json

## 7. Reproducibility notes

The archived checkpoints permit evaluation without
retraining the networks.

The acquisition, processing, training, and evaluation
scripts document the computational workflow.

Reproducing the full experiment requires the archived
meteorological inputs, the prescribed data-processing
configuration, and a compatible Python environment.

Results may vary when new NASA POWER data are
retrieved rather than using the archived input files.

## 8. Optional MATLAB visualization

The three-dimensional streamribbon visualization was generated
using MATLAB.

The source code is provided in:

    melissa/src/ribbon_melissa_split_2345.m

The corresponding publication figures are archived as PDF
and PNG files in:

    melissa/outputs/split_protocol/figures/revised_2345/

Running the MATLAB script requires a compatible MATLAB
installation and license.

MATLAB is not required for training, evaluating the neural
models, or reproducing the quantitative results.

An alternative three-dimensional visualization is available
through the Python script:

    melissa/src/make_melissa_fidelity_ribbons_split.py

## Regenerating the training arrays

The original 61 hourly NASA POWER files are available in
`data/raw/`, together with `hourly_eye_centers.csv`.

The preprocessing script generates the arrays required
for training the neural networks.

From the `melissa/` directory, run:

    python3 src/prepare_melissa_hourly.py \
        --input_dir data/raw \
        --centers data/raw/hourly_eye_centers.csv \
        --outdir data/processed/rebuild_check \
        --nx 40 \
        --ny 40 \
        --dz 20 \
        --dt_minutes 5 \
        --wahba_lambda 0.001 \
        --holdout_gap_minutes 0 \
        --max_height 290

The parameters correspond to the archived preprocessing
manifest.

The generated files include:

- train_X.npy
- train_Yuv.npy
- train_times.csv
- eye_path.npz
- train_hourly_processed.npz
- manifest.json

The generated training arrays contain approximately
17.3 million space-time records and require several
hundred megabytes of storage.

These large intermediate arrays are not tracked in Git.

The original meteorological data, processing scripts,
validation-selected checkpoints and numerical summaries
are archived in the repository.

The preprocessing command is documented for reproducibility,
but regeneration from a clean clone has not been
independently verified.

MATLAB is optional and is not required for training,
model evaluation or the Python visualizations.
