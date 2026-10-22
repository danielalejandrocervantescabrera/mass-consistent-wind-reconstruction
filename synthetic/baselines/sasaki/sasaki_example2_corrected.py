#!/usr/bin/env python3
"""Corrected, self-contained Sasaki/RBF baseline for DeepWind Example II.

Exact field on Omega=[-1,1]^3:
    u_exact(x,y,z) = (-y z^2, x z^2, (x^2+y^2)/2).

Classical background field:
    u0(x,y,z) = (-y z^2, x z^2, 0),
which is already divergence-free. The mass-consistent Sasaki correction is
represented as
    u_S = u0 + grad(lambda),
with lambda obtained from the same inverse-multiquadric RBF collocation idea
used in the historical vector_field_7.py implementation.

Corrections relative to the historical script:
  1. The z-z second derivative is actually dzz (the old script called dyy twice).
  2. RMS(div) replaces signed mean(div), avoiding cancellation.
  3. MSE(u,v) and MSE(u,v,w) use exactly the same componentwise mean-square
     definitions as the DeepWind benchmark.
  4. Test points use the same evalSeed/nTest convention as DeepWind.
  5. The lower-boundary error is reported as RMS(w-g_b), g_b=(x^2+y^2)/2.
  6. The RBF tensor-grid partition includes the complete top face; the old
     three_dim_uniform_nodes_v2 omitted top edges/corners.
  7. RBF matrices are vectorized NumPy implementations of the same IMQ
     derivative formulas, making the baseline much faster without changing
     the mathematical discretization.

Boundary information is intentionally kept as in the old Sasaki Example-II
benchmark: zero correction flux on the lateral faces and exact normal vertical
velocity on both bottom and top. Thus Sasaki is given at least as much boundary
information as DeepWind, which uses only the lower w condition.
"""

import argparse
import csv
import os
import time
import numpy as np


def exact_field(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out = np.empty((xyz.shape[0], 3), dtype=np.float64)
    out[:, 0] = -y * z**2
    out[:, 1] =  x * z**2
    out[:, 2] = 0.5 * (x**2 + y**2)
    return out


def background_field(xyz: np.ndarray) -> np.ndarray:
    out = exact_field(xyz)
    out[:, 2] = 0.0
    return out


def bottom_target(xyz: np.ndarray) -> np.ndarray:
    return 0.5 * (xyz[:, 0]**2 + xyz[:, 1]**2)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b)**2))


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a**2)))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-14))


# -----------------------------------------------------------------------------
# Uniform RBF centers. This follows the historical V2 partition, but fixes the
# missing top edges/corners by assigning the whole top face to the top group.
# Every tensor-grid point is assigned exactly once, so the collocation matrix
# remains square.
# -----------------------------------------------------------------------------
def uniform_regions(bounds, n_centers_target: int):
    a, b, c, d, e, f = bounds
    n = max(3, int(np.ceil(n_centers_target ** (1.0 / 3.0))))
    xs = np.linspace(a, b, n)
    ys = np.linspace(c, d, n)
    zs = np.linspace(e, f, n)

    interior = np.array([[x, y, z]
                         for z in zs[1:-1]
                         for y in ys[1:-1]
                         for x in xs[1:-1]], dtype=np.float64)

    bottom = np.array([[x, y, e] for y in ys for x in xs], dtype=np.float64)
    top = np.array([[x, y, f] for y in ys for x in xs], dtype=np.float64)

    # Lateral faces exclude z=e,f because bottom/top own those edges/corners.
    V1 = np.array([[a, y, z] for z in zs[1:-1] for y in ys], dtype=np.float64)
    V2 = np.array([[b, y, z] for z in zs[1:-1] for y in ys], dtype=np.float64)
    # y-faces exclude x=a,b because x-faces own those lateral edges.
    V3 = np.array([[x, c, z] for z in zs[1:-1] for x in xs[1:-1]], dtype=np.float64)
    V4 = np.array([[x, d, z] for z in zs[1:-1] for x in xs[1:-1]], dtype=np.float64)

    all_pts = np.vstack([interior, V1, V2, V3, V4, bottom, top])
    if all_pts.shape[0] != n**3:
        raise RuntimeError(f"center partition error: got {all_pts.shape[0]}, expected {n**3}")

    return {
        "n_side": n,
        "interior": interior,
        "V1": V1,
        "V2": V2,
        "V3": V3,
        "V4": V4,
        "bottom": bottom,
        "top": top,
        "all": all_pts,
    }


# -----------------------------------------------------------------------------
# Vectorized inverse-multiquadric derivatives.
# phi(r) = 1/sqrt(1 + (ep r)^2)
# These are the same formulas used in the original rbf_utils.py.
# -----------------------------------------------------------------------------
def deltas(dsites: np.ndarray, ctrs: np.ndarray):
    D = dsites[:, None, :] - ctrs[None, :, :]
    r2 = np.sum(D * D, axis=2)
    return D, r2


def grad_imq(ep: float, dsites: np.ndarray, ctrs: np.ndarray):
    D, r2 = deltas(dsites, ctrs)
    A = 1.0 + ep**2 * r2
    fac = -ep**2 / A**1.5
    return D * fac[:, :, None]


def second_imq(ep: float, dsites: np.ndarray, ctrs: np.ndarray):
    D, r2 = deltas(dsites, ctrs)
    A = 1.0 + ep**2 * r2
    common = ep**2 / A**2.5
    dxx = common * (3.0 * ep**2 * D[:, :, 0]**2 - A)
    dyy = common * (3.0 * ep**2 * D[:, :, 1]**2 - A)
    dzz = common * (3.0 * ep**2 * D[:, :, 2]**2 - A)
    return dxx, dyy, dzz


def elliptic_matrix(ep: float, dsites: np.ndarray, ctrs: np.ndarray):
    dxx, dyy, dzz = second_imq(ep, dsites, ctrs)
    return dxx + dyy + dzz


def neumann_matrix(ep: float, normal, dsites: np.ndarray, ctrs: np.ndarray):
    G = grad_imq(ep, dsites, ctrs)
    n = np.asarray(normal, dtype=np.float64)
    return G @ n


def build_system(n_centers: int, ep: float):
    bounds = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    R = uniform_regions(bounds, n_centers)
    ctrs = R["all"]

    E = elliptic_matrix(ep, R["interior"], ctrs)
    NL = neumann_matrix(ep, [-1, 0, 0], R["V1"], ctrs)
    NR = neumann_matrix(ep, [ 1, 0, 0], R["V2"], ctrs)
    NBa = neumann_matrix(ep, [0, -1, 0], R["V3"], ctrs)
    NF = neumann_matrix(ep, [0,  1, 0], R["V4"], ctrs)
    NB = neumann_matrix(ep, [0, 0, -1], R["bottom"], ctrs)
    NT = neumann_matrix(ep, [0, 0,  1], R["top"], ctrs)

    gram = np.vstack([E, NL, NR, NBa, NF, NB, NT])

    # -Delta(lambda)=div(u0)=0 -> Delta(lambda)=0 in this zero-source case.
    # Lateral correction normal flux is zero. Bottom/top prescribe exact w.
    rhs = np.concatenate([
        np.zeros(R["interior"].shape[0]),
        np.zeros(R["V1"].shape[0]),
        np.zeros(R["V2"].shape[0]),
        np.zeros(R["V3"].shape[0]),
        np.zeros(R["V4"].shape[0]),
        -bottom_target(R["bottom"]),  # n=(0,0,-1)
        +bottom_target(R["top"]),     # n=(0,0,+1)
    ])

    if gram.shape != (ctrs.shape[0], ctrs.shape[0]):
        raise RuntimeError(f"RBF system is not square: Gram={gram.shape}, centers={ctrs.shape[0]}")
    return gram, rhs, ctrs, R["n_side"]


def evaluate_gradient(ep, xyz, ctrs, coeff, chunk=5000):
    ans = np.empty((xyz.shape[0], 3), dtype=np.float64)
    for i in range(0, xyz.shape[0], chunk):
        q = xyz[i:i+chunk]
        G = grad_imq(ep, q, ctrs)
        ans[i:i+len(q), 0] = G[:, :, 0] @ coeff
        ans[i:i+len(q), 1] = G[:, :, 1] @ coeff
        ans[i:i+len(q), 2] = G[:, :, 2] @ coeff
    return ans


def evaluate_laplacian(ep, xyz, ctrs, coeff, chunk=5000):
    ans = np.empty(xyz.shape[0], dtype=np.float64)
    for i in range(0, xyz.shape[0], chunk):
        q = xyz[i:i+chunk]
        dxx, dyy, dzz = second_imq(ep, q, ctrs)
        ans[i:i+len(q)] = (dxx + dyy + dzz) @ coeff
    return ans


def main():
    ap = argparse.ArgumentParser(description="Corrected Sasaki/RBF baseline for polynomial Example II")
    ap.add_argument("--nCenters", type=int, default=512,
                    help="target number of tensor-grid RBF centers (rounded up to n_side^3)")
    ap.add_argument("--ep", type=float, default=1.0, help="IMQ RBF shape parameter")
    ap.add_argument("--nTest", type=int, default=10000)
    ap.add_argument("--nBottomEval", type=int, default=5000)
    ap.add_argument("--evalSeed", type=int, default=54321)
    ap.add_argument("--chunk", type=int, default=2500, help="evaluation chunk size")
    ap.add_argument("--outdir", type=str, default="out_sasaki_example2")
    ap.add_argument("--tag", type=str, default="sasaki_example2")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summary_dir = os.path.join(args.outdir, "summary")
    pred_dir = os.path.join(args.outdir, "predictions")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    t0 = time.time()
    gram, rhs, ctrs, n_side = build_system(args.nCenters, args.ep)
    build_seconds = time.time() - t0

    t0 = time.time()
    coeff = np.linalg.solve(gram, rhs)
    solve_seconds = time.time() - t0

    cond_gram = float(np.linalg.cond(gram))
    system_rel_residual = float(np.linalg.norm(gram @ coeff - rhs) / (np.linalg.norm(rhs) + 1e-14))

    # Same fixed test distribution used by DeepWind.
    rng_eval = np.random.RandomState(args.evalSeed)
    Xtest = rng_eval.uniform([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], size=(args.nTest, 3))
    Uexact = exact_field(Xtest)
    U0 = background_field(Xtest)

    t0 = time.time()
    grad = evaluate_gradient(args.ep, Xtest, ctrs, coeff, args.chunk)
    Usasaki = U0 + grad
    eval_grad_seconds = time.time() - t0

    test_mse_uv = mse(Usasaki[:, :2], Uexact[:, :2])
    test_mse_uvw = mse(Usasaki, Uexact)
    test_rel_l2 = rel_l2(Usasaki, Uexact)

    t0 = time.time()
    div = evaluate_laplacian(args.ep, Xtest, ctrs, coeff, args.chunk)
    test_rms_div = rms(div)  # div(u0)=0 analytically
    eval_div_seconds = time.time() - t0

    rng_bottom = np.random.RandomState(args.evalSeed + 1)
    Xbottom = np.column_stack([
        rng_bottom.uniform(-1.0, 1.0, args.nBottomEval),
        rng_bottom.uniform(-1.0, 1.0, args.nBottomEval),
        np.full(args.nBottomEval, -1.0),
    ])
    grad_bottom = evaluate_gradient(args.ep, Xbottom, ctrs, coeff, args.chunk)
    test_rms_bottom_bc = rms(grad_bottom[:, 2] - bottom_target(Xbottom))

    rms_speed_exact = rms(Uexact)
    rms_uv_exact = rms(Uexact[:, :2])
    rel_rms_div = test_rms_div / (rms_speed_exact + 1e-14)
    rel_rms_bottom_bc = test_rms_bottom_bc / (rms_speed_exact + 1e-14)
    rel_uv = float(np.sqrt(test_mse_uv / (rms_uv_exact**2 + 1e-14)))
    rel_uvw = float(np.sqrt(test_mse_uvw / (rms_speed_exact**2 + 1e-14)))

    pred_csv = os.path.join(pred_dir, f"{args.tag}_test_predictions.csv")
    np.savetxt(
        pred_csv,
        np.column_stack([Xtest, Uexact, Usasaki]),
        delimiter=",",
        header="x,y,z,u_exact,v_exact,w_exact,u_sasaki,v_sasaki,w_sasaki",
        comments="",
    )

    summary_csv = os.path.join(summary_dir, f"{args.tag}_eval_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "nCenters_requested", "nSide", "nCenters_actual", "ep", "evalSeed", "nTest",
            "test_mse_uv", "test_mse_uvw", "test_relL2_field", "test_rms_div",
            "test_rms_bottom_bc", "rel_rms_div", "rel_rms_bottom_bc", "rel_uv", "rel_uvw",
            "cond_gram", "system_rel_residual", "build_seconds", "solve_seconds",
            "eval_grad_seconds", "eval_div_seconds", "prediction_csv",
        ])
        w.writerow([
            "sasaki_rbf", args.nCenters, n_side, ctrs.shape[0], args.ep, args.evalSeed, args.nTest,
            test_mse_uv, test_mse_uvw, test_rel_l2, test_rms_div,
            test_rms_bottom_bc, rel_rms_div, rel_rms_bottom_bc, rel_uv, rel_uvw,
            cond_gram, system_rel_residual, build_seconds, solve_seconds,
            eval_grad_seconds, eval_div_seconds, pred_csv,
        ])

    print("\n=== Corrected Sasaki/RBF: Example II ===")
    print(f"grid side               = {n_side}")
    print(f"actual centers          = {ctrs.shape[0]}")
    print(f"shape parameter ep      = {args.ep:g}")
    print(f"cond(Gram)              = {cond_gram:.6e}")
    print(f"system relative residual= {system_rel_residual:.6e}")
    print(f"test MSE(u,v)           = {test_mse_uv:.6e}")
    print(f"test MSE(u,v,w)         = {test_mse_uvw:.6e}")
    print(f"test relL2(field)       = {test_rel_l2:.6e}")
    print(f"test RMS(div)           = {test_rms_div:.6e}")
    print(f"test RMS(bottom BC)     = {test_rms_bottom_bc:.6e}")
    print(f"summary                 = {summary_csv}")
    print(f"predictions             = {pred_csv}")


if __name__ == "__main__":
    main()
