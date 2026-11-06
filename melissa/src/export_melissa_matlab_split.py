#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export Hurricane Melissa neural reconstruction to MATLAB.

Model:
    Seed: 2345
    Validation-selected epoch: 2500

Snapshot:
    28 October 2025, 07:00 UTC

Run from the melissa/ directory:

    python src/export_melissa_matlab_split.py

Output:
    outputs/split_protocol/figures/revised_2345/
        melissa_reconstructed_for_matlab.mat

The reconstructed velocities are exported in physical
units (m/s), without visual exaggeration.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.io import savemat

from make_melissa_fidelity_ribbons_split import (
    argument_parser,
    read_inputs,
    load_and_predict,
)


def main():

    # --------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------

    parser = argument_parser()
    args = parser.parse_args()

    print("=" * 65)
    print("HURRICANE MELISSA - MATLAB FIELD EXPORT")
    print("=" * 65)

    print(f"Snapshot: {args.snapshot}")
    print(f"Network seed: {args.seed}")

    # --------------------------------------------------
    # 2. Read processed meteorological data
    # --------------------------------------------------

    print("\nReading processed reference...")

    data = read_inputs(args)

    print(f"Time: {data['hour']:.2f} h")
    print(f"Horizontal velocity scale: {data['S_t']:.6f} m/s")
    print(f"Vertical velocity scale: {data['Sw']:.6f} m/s")

    # --------------------------------------------------
    # 3. Load checkpoint and reconstruct velocity
    # --------------------------------------------------

    print("\nLoading trained neural network...")

    U, V, W = load_and_predict(data, args)

    print("Neural reconstruction completed.")

    print(f"U range: [{U.min():.6f}, {U.max():.6f}] m/s")
    print(f"V range: [{V.min():.6f}, {V.max():.6f}] m/s")
    print(f"W range: [{W.min():.6f}, {W.max():.6f}] m/s")

    # --------------------------------------------------
    # 4. Convert arrays to MATLAB convention
    # --------------------------------------------------

    # Python fields: (z, eta, xi)
    #
    # MATLAB meshgrid(xi, eta, z):
    #                (eta, xi, z)

    def to_matlab(array):

        result = np.transpose(array, (1, 2, 0))

        return np.ascontiguousarray(
            result,
            dtype=np.float64
        )

    U_mat = to_matlab(U)
    V_mat = to_matlab(V)
    W_mat = to_matlab(W)

    expected_shape = (
        len(data["eta"]),
        len(data["xi"]),
        len(data["z"]),
    )

    assert U_mat.shape == expected_shape
    assert V_mat.shape == expected_shape
    assert W_mat.shape == expected_shape

    print("\nMATLAB array shape:", expected_shape)

    # --------------------------------------------------
    # 5. Prepare output directory
    # --------------------------------------------------

    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        outdir / "melissa_reconstructed_for_matlab.mat"
    )

    # --------------------------------------------------
    # 6. Export coordinates and velocities
    # --------------------------------------------------

    matlab_data = {

        "xi_m": np.asarray(
            data["xi"],
            dtype=np.float64
        ).reshape(1, -1),

        "eta_m": np.asarray(
            data["eta"],
            dtype=np.float64
        ).reshape(1, -1),

        "z_m": np.asarray(
            data["z"],
            dtype=np.float64
        ).reshape(1, -1),

        "U_mps": U_mat,

        "V_mps": V_mat,

        "W_mps": W_mat,

        "snapshot_key": args.snapshot,

        "seed": np.int32(args.seed),

        "best_epoch": np.int32(
            data["summary"]["training"]["best_epoch"]
        ),

        "S_t_mps": float(data["S_t"]),

        "Sw_mps": float(data["Sw"]),
    }

    savemat(
        output_file,
        matlab_data,
        do_compression=True
    )

    # --------------------------------------------------
    # 7. Reproducibility metadata
    # --------------------------------------------------

    checkpoint_path = data["weights_path"]

    checkpoint_hash = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    metadata = {

        "snapshot": args.snapshot,

        "seed": args.seed,

        "best_epoch":
            data["summary"]["training"]["best_epoch"],

        "checkpoint_sha256": checkpoint_hash,

        "units": "coordinates in meters; velocities in m/s",

        "matlab_array_order": "(eta, xi, z)",

        "shape": list(expected_shape),

        "output_file": str(output_file),
    }

    metadata_path = (
        outdir / "matlab_export_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    # --------------------------------------------------
    # 8. Summary
    # --------------------------------------------------

    print("\n" + "=" * 65)

    print("MATLAB EXPORT COMPLETED")

    print("=" * 65)

    print(f"Seed: {args.seed}")

    print(
        "Selected epoch:",
        data["summary"]["training"]["best_epoch"]
    )

    print("\nOutput file:")

    print(output_file)

    print("\nVariables exported:")

    print("  xi_m")
    print("  eta_m")
    print("  z_m")
    print("  U_mps")
    print("  V_mps")
    print("  W_mps")

    print("\nReady for MATLAB streamribbon visualization.")


if __name__ == "__main__":
    main()
