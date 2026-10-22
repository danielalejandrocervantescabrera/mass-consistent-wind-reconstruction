# Hurricane input provenance and attribution

The 61 hourly meteorological snapshot files in `melissa/data/raw/` were obtained through the NASA POWER workflow for the October 26–28, 2025 Hurricane Melissa reconstruction. Their archived gridded numerical content is included for reproducibility. Horizontal wind was constructed from POWER speed (WSC) and WD10M direction (direction from 10 m reused vertically); thus the eight-height vector fields are **not independent direct vector-wind measurements**.

**POWER citation:** NASA POWER asks publications to cite both the NASA POWER project and the particular hourly data product, including its service/version and access date. See https://power.larc.nasa.gov/docs/referencing/ . The exact source service version and retrieval date should be recovered/confirmed before the final academic release rather than guessed.

The archived `hourly_eye_centers.csv` documents the reconstruction eye trajectory used in processing. The original external source document for certain eye anchors and raw HTTP responses was not retained. The reconstruction uses linear vertical interpolation and PCHIP over time; intermediate five-minute fields are derived from the observed hourly snapshots.

Please verify attribution and rights for each bundled scientific input before making the GitHub repository public. Do not publish credentials, unpublished confidential data or patient-related information.
