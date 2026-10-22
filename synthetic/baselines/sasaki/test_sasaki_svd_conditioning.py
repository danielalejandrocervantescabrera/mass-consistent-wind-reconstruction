#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import time
from pathlib import Path

import numpy as np


def load_module(path, module_name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. "
            "Este script debe estar junto a los codigos Sasaki corregidos."
        )

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def tsvd_solve(U, s, Vt, b, rcond):
    """
    Solucion TSVD de

        G c = b,

    donde

        G = U diag(s) V^T.

    Se descartan valores singulares

        s_i <= rcond * s_max.
    """

    cutoff = rcond * s[0]

    keep = s > cutoff

    rank = np.count_nonzero(keep)

    if rank == 0:
        raise RuntimeError("TSVD elimino todos los valores singulares.")

    y = U.T @ b

    z = np.zeros_like(y)

    z[keep] = y[keep] / s[keep]

    c = Vt.T @ z

    s_keep = s[keep]

    effective_cond = s_keep[0] / s_keep[-1]

    return c, rank, cutoff, effective_cond


def relative_residual(G, c, b):
    return (
        np.linalg.norm(G @ c - b)
        /
        (np.linalg.norm(b) + 1.0e-15)
    )


def raw_condition_number(s):
    if s[-1] == 0:
        return np.inf

    return s[0] / s[-1]


def evaluate_example1(
    mod,
    coeff,
    centers,
    ep,
    epsilon,
    n_test,
    n_bottom,
    seed,
    chunk,
):

    rng = np.random.RandomState(seed)

    X = rng.uniform(
        [-7.0, -7.0, 0.0],
        [ 7.0,  7.0, 7.0],
        size=(n_test, 3),
    )

    u_exact = mod.exact_field(X, epsilon)

    u0 = mod.background_field(X, epsilon)

    grad_lambda = mod.eval_grad(
        ep,
        X,
        centers,
        coeff,
        chunk,
    )

    u_sasaki = u0 + grad_lambda

    # En Example I:
    #
    # div(u0) = -epsilon*z
    #
    lap_lambda = mod.eval_lap(
        ep,
        X,
        centers,
        coeff,
        chunk,
    )

    divergence = (
        -epsilon * X[:, 2]
        +
        lap_lambda
    )

    rng_bottom = np.random.RandomState(seed + 1)

    Xb = np.column_stack(
        [
            rng_bottom.uniform(-7.0, 7.0, n_bottom),
            rng_bottom.uniform(-7.0, 7.0, n_bottom),
            np.zeros(n_bottom),
        ]
    )

    grad_bottom = mod.eval_grad(
        ep,
        Xb,
        centers,
        coeff,
        chunk,
    )

    # g_b = 0
    bottom_error = grad_bottom[:, 2]

    mse_uv = mod.mse(
        u_sasaki[:, :2],
        u_exact[:, :2],
    )

    mse_uvw = mod.mse(
        u_sasaki,
        u_exact,
    )

    rel_l2 = mod.rel_l2(
        u_sasaki,
        u_exact,
    )

    rms_div = mod.rms(divergence)

    rms_bottom = mod.rms(bottom_error)

    return {
        "mse_uv": mse_uv,
        "mse_uvw": mse_uvw,
        "rel_l2": rel_l2,
        "rms_div": rms_div,
        "rms_bottom": rms_bottom,
    }


def evaluate_example2(
    mod,
    coeff,
    centers,
    ep,
    n_test,
    n_bottom,
    seed,
    chunk,
):

    rng = np.random.RandomState(seed)

    X = rng.uniform(
        [-1.0, -1.0, -1.0],
        [ 1.0,  1.0,  1.0],
        size=(n_test, 3),
    )

    u_exact = mod.exact_field(X)

    u0 = mod.background_field(X)

    grad_lambda = mod.evaluate_gradient(
        ep,
        X,
        centers,
        coeff,
        chunk,
    )

    u_sasaki = u0 + grad_lambda

    # En Example II:
    #
    # div(u0) = 0
    #
    divergence = mod.evaluate_laplacian(
        ep,
        X,
        centers,
        coeff,
        chunk,
    )

    rng_bottom = np.random.RandomState(seed + 1)

    Xb = np.column_stack(
        [
            rng_bottom.uniform(-1.0, 1.0, n_bottom),
            rng_bottom.uniform(-1.0, 1.0, n_bottom),
            np.full(n_bottom, -1.0),
        ]
    )

    grad_bottom = mod.evaluate_gradient(
        ep,
        Xb,
        centers,
        coeff,
        chunk,
    )

    bottom_target = mod.bottom_target(Xb)

    bottom_error = (
        grad_bottom[:, 2]
        -
        bottom_target
    )

    mse_uv = mod.mse(
        u_sasaki[:, :2],
        u_exact[:, :2],
    )

    mse_uvw = mod.mse(
        u_sasaki,
        u_exact,
    )

    rel_l2 = mod.rel_l2(
        u_sasaki,
        u_exact,
    )

    rms_div = mod.rms(divergence)

    rms_bottom = mod.rms(bottom_error)

    return {
        "mse_uv": mse_uv,
        "mse_uvw": mse_uvw,
        "rel_l2": rel_l2,
        "rms_div": rms_div,
        "rms_bottom": rms_bottom,
    }


def run_example(
    example,
    mod,
    n_centers,
    ep,
    epsilon,
    rconds,
    n_test,
    n_bottom,
    seed,
    chunk,
):

    print()
    print("=" * 70)
    print(f"Example {example}")
    print("=" * 70)

    print(
        f"Construyendo sistema RBF "
        f"(nCenters={n_centers}, ep={ep}) ..."
    )

    t0 = time.time()

    if example == 1:

        G, b, centers, n_side = mod.build_system(
            n_centers,
            ep,
            epsilon,
        )

    else:

        G, b, centers, n_side = mod.build_system(
            n_centers,
            ep,
        )

    build_time = time.time() - t0

    print(f"grid side               = {n_side}")
    print(f"actual centers          = {len(centers)}")
    print(f"shape parameter ep      = {ep}")
    print(f"build time              = {build_time:.3f} s")

    print()
    print("Calculando SVD completa ...")

    t0 = time.time()

    U, s, Vt = np.linalg.svd(
        G,
        full_matrices=False,
    )

    svd_time = time.time() - t0

    cond_raw = raw_condition_number(s)

    print(f"SVD time                = {svd_time:.3f} s")
    print(f"sigma_max               = {s[0]:.6e}")
    print(f"sigma_min               = {s[-1]:.6e}")
    print(f"cond_2(G)               = {cond_raw:.6e}")

    rows = []

    #
    # ---------------------------------------------------------
    # 1. SOLVE DIRECTO
    # ---------------------------------------------------------
    #

    print()
    print("------------------------------------------------------------")
    print("Direct solve")
    print("------------------------------------------------------------")

    try:

        t0 = time.time()

        coeff = np.linalg.solve(
            G,
            b,
        )

        solve_time = time.time() - t0

        residual = relative_residual(
            G,
            coeff,
            b,
        )

        coeff_norm = np.linalg.norm(coeff)

        if example == 1:

            metrics = evaluate_example1(
                mod,
                coeff,
                centers,
                ep,
                epsilon,
                n_test,
                n_bottom,
                seed,
                chunk,
            )

        else:

            metrics = evaluate_example2(
                mod,
                coeff,
                centers,
                ep,
                n_test,
                n_bottom,
                seed,
                chunk,
            )

        print(f"system rel residual    = {residual:.6e}")
        print(f"||coeff||_2            = {coeff_norm:.6e}")
        print(f"MSE(u,v)               = {metrics['mse_uv']:.6e}")
        print(f"MSE(u,v,w)             = {metrics['mse_uvw']:.6e}")
        print(f"relL2(field)           = {metrics['rel_l2']:.6e}")
        print(f"RMS(div)               = {metrics['rms_div']:.6e}")
        print(f"RMS(bottom BC)         = {metrics['rms_bottom']:.6e}")

        rows.append(
            {
                "example": example,
                "method": "direct_solve",
                "rcond": "",
                "nCenters_requested": n_centers,
                "nCenters_actual": len(centers),
                "n_side": n_side,
                "ep": ep,
                "raw_condition_2": cond_raw,
                "retained_rank": len(s),
                "discarded_singular_values": 0,
                "svd_cutoff": 0.0,
                "effective_condition": cond_raw,
                "system_rel_residual": residual,
                "coefficient_l2_norm": coeff_norm,
                "test_mse_uv": metrics["mse_uv"],
                "test_mse_uvw": metrics["mse_uvw"],
                "test_relL2_field": metrics["rel_l2"],
                "test_rms_div": metrics["rms_div"],
                "test_rms_bottom_bc": metrics["rms_bottom"],
                "solve_seconds": solve_time,
            }
        )

    except np.linalg.LinAlgError as e:

        print("Direct solve FAILED:")
        print(e)

    #
    # ---------------------------------------------------------
    # 2. TSVD
    # ---------------------------------------------------------
    #

    for rcond in rconds:

        print()
        print("------------------------------------------------------------")
        print(f"TSVD rcond = {rcond:.0e}")
        print("------------------------------------------------------------")

        t0 = time.time()

        coeff, rank, cutoff, effective_cond = tsvd_solve(
            U,
            s,
            Vt,
            b,
            rcond,
        )

        solve_time = time.time() - t0

        residual = relative_residual(
            G,
            coeff,
            b,
        )

        coeff_norm = np.linalg.norm(coeff)

        if example == 1:

            metrics = evaluate_example1(
                mod,
                coeff,
                centers,
                ep,
                epsilon,
                n_test,
                n_bottom,
                seed,
                chunk,
            )

        else:

            metrics = evaluate_example2(
                mod,
                coeff,
                centers,
                ep,
                n_test,
                n_bottom,
                seed,
                chunk,
            )

        print(f"cutoff                  = {cutoff:.6e}")
        print(f"retained rank           = {rank}/{len(s)}")
        print(
            f"discarded singular vals = "
            f"{len(s) - rank}"
        )
        print(
            f"effective condition     = "
            f"{effective_cond:.6e}"
        )
        print(
            f"system rel residual     = "
            f"{residual:.6e}"
        )
        print(
            f"||coeff||_2             = "
            f"{coeff_norm:.6e}"
        )
        print(
            f"MSE(u,v)                = "
            f"{metrics['mse_uv']:.6e}"
        )
        print(
            f"MSE(u,v,w)              = "
            f"{metrics['mse_uvw']:.6e}"
        )
        print(
            f"relL2(field)            = "
            f"{metrics['rel_l2']:.6e}"
        )
        print(
            f"RMS(div)                = "
            f"{metrics['rms_div']:.6e}"
        )
        print(
            f"RMS(bottom BC)          = "
            f"{metrics['rms_bottom']:.6e}"
        )

        rows.append(
            {
                "example": example,
                "method": "tsvd",
                "rcond": rcond,
                "nCenters_requested": n_centers,
                "nCenters_actual": len(centers),
                "n_side": n_side,
                "ep": ep,
                "raw_condition_2": cond_raw,
                "retained_rank": rank,
                "discarded_singular_values": len(s) - rank,
                "svd_cutoff": cutoff,
                "effective_condition": effective_cond,
                "system_rel_residual": residual,
                "coefficient_l2_norm": coeff_norm,
                "test_mse_uv": metrics["mse_uv"],
                "test_mse_uvw": metrics["mse_uvw"],
                "test_relL2_field": metrics["rel_l2"],
                "test_rms_div": metrics["rms_div"],
                "test_rms_bottom_bc": metrics["rms_bottom"],
                "solve_seconds": solve_time,
            }
        )

    return rows


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--example",
        choices=["1", "2", "both"],
        default="both",
    )

    parser.add_argument(
        "--code-dir",
        default=".",
    )

    parser.add_argument(
        "--nCenters1",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--ep1",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--nCenters2",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--ep2",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--rconds",
        type=float,
        nargs="+",
        default=[
            1e-10,
            1e-12,
            1e-14,
        ],
    )

    parser.add_argument(
        "--nTest",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--nBottomEval",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--evalSeed",
        type=int,
        default=54321,
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=2500,
    )

    parser.add_argument(
        "--output",
        default="sasaki_svd_conditioning.csv",
    )

    args = parser.parse_args()

    code_dir = Path(args.code_dir)

    rows = []

    if args.example in ("1", "both"):

        mod1 = load_module(
            code_dir
            /
            "sasaki_example1_corrected.py",
            "sasaki_example1_corrected",
        )

        rows += run_example(
            example=1,
            mod=mod1,
            n_centers=args.nCenters1,
            ep=args.ep1,
            epsilon=args.epsilon,
            rconds=args.rconds,
            n_test=args.nTest,
            n_bottom=args.nBottomEval,
            seed=args.evalSeed,
            chunk=args.chunk,
        )

    if args.example in ("2", "both"):

        mod2 = load_module(
            code_dir
            /
            "sasaki_example2_corrected.py",
            "sasaki_example2_corrected",
        )

        rows += run_example(
            example=2,
            mod=mod2,
            n_centers=args.nCenters2,
            ep=args.ep2,
            epsilon=args.epsilon,
            rconds=args.rconds,
            n_test=args.nTest,
            n_bottom=args.nBottomEval,
            seed=args.evalSeed,
            chunk=args.chunk,
        )

    fieldnames = [
        "example",
        "method",
        "rcond",
        "nCenters_requested",
        "nCenters_actual",
        "n_side",
        "ep",
        "raw_condition_2",
        "retained_rank",
        "discarded_singular_values",
        "svd_cutoff",
        "effective_condition",
        "system_rel_residual",
        "coefficient_l2_norm",
        "test_mse_uv",
        "test_mse_uvw",
        "test_relL2_field",
        "test_rms_div",
        "test_rms_bottom_bc",
        "solve_seconds",
    ]

    with open(
        args.output,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    print()
    print("=" * 70)
    print(f"Resultados guardados en: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
