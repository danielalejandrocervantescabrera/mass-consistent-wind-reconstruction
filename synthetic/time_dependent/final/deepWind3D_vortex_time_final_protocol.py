#!/usr/bin/env python3
"""DeepWind 3D vortex -- time-dependent benchmark with the final protocol.

This script updates the previous time-trajectory experiment so that it follows
exactly the methodological choices adopted for the final stationary synthetic
benchmark:

  * fixed sparse observations X_obs in space-time, generated once by LHS;
  * only the horizontal components (u,v) are observed;
  * observation points and physics collocation points are distinct;
  * interior and bottom collocation points are re-sampled every optimization batch;
  * normalized network inputs in [-1,1]^4;
  * spatial divergence u_x + v_y + w_z computed by PyTorch autograd and the
    physical-coordinate chain rule (no time derivative enters continuity);
  * normalized penalty objective
        J = 1/2 D_data + beta1 Q_div + beta2 B_bottom;
  * validation/checkpoint selection uses ONLY MSE_val(u,v);
  * best checkpoint is reloaded before test and time-slice evaluation;
  * fixed validation and test sets are independent of the observation layout and
    the network/collocation seed.

The time-dependent field is the same analytical vortex translated in the
horizontal plane along a prescribed trajectory.  Therefore this benchmark is a
controlled test of continuous space-time reconstruction of a moving flow
structure; it is not intended as a general model of atmospheric dynamics.

Default configuration reproduces the parabolic benchmark used in the paper,
but with the frozen stationary architecture/training protocol:

    x,y in [-25,25], z in [0,7], t in [-1,1]
    cx(t) = 20 t
    cy(t) = 30 t^2
    epsilon = 0.02

    width=60, depth=2
    N_obs=1024 fixed LHS observations
    N_Omega=512, N_b=64 per optimization batch
    beta1=beta2=1
    nEpoch=2000, nBatch=25
    Adam lr=1e-2, weight_decay=1e-6
    StepLR(step_size=200, gamma=0.5), stepped once per epoch

Example single run:

  python3 deepWind3D_vortex_time_final_protocol.py \
      --seed 0 --outdir out_time_final_protocol --tag timefinal_parabolic_s0

The companion shell script run_time_final_protocol.sh runs seeds 0,1,2 and
aggregates the results.
"""

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


# -----------------------------------------------------------------------------
# Global dtype / reproducibility
# -----------------------------------------------------------------------------
torch.set_default_dtype(torch.float64)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_latin_hypercube(n: int, lower, upper, seed: int) -> np.ndarray:
    """Dependency-free randomized LHS in a rectangular box."""
    if n < 1:
        raise ValueError("n must be >= 1")

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape")
    if np.any(upper <= lower):
        raise ValueError("each upper bound must be greater than its lower bound")

    rng = np.random.RandomState(seed)
    dim = lower.size
    unit = np.empty((n, dim), dtype=np.float64)
    for j in range(dim):
        strata = rng.permutation(n)
        jitter = rng.rand(n)
        unit[:, j] = (strata + jitter) / float(n)
    return lower + unit * (upper - lower)


# -----------------------------------------------------------------------------
# Space-time domain
# -----------------------------------------------------------------------------
@dataclass
class DomainCubeTime:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    t0: float
    t1: float
    innerBatchSize: int
    bdyBatchSize: int
    numQuad: int

    def sample_from_domain(self) -> np.ndarray:
        x = np.random.rand(self.innerBatchSize) * (self.b - self.a) + self.a
        y = np.random.rand(self.innerBatchSize) * (self.d - self.c) + self.c
        z = np.random.rand(self.innerBatchSize) * (self.f - self.e) + self.e
        t = np.random.rand(self.innerBatchSize) * (self.t1 - self.t0) + self.t0
        return np.stack([x, y, z, t], axis=1)

    def sample_from_bottom(self) -> np.ndarray:
        x = np.random.rand(self.bdyBatchSize) * (self.b - self.a) + self.a
        y = np.random.rand(self.bdyBatchSize) * (self.d - self.c) + self.c
        z = np.full(self.bdyBatchSize, self.e, dtype=np.float64)
        t = np.random.rand(self.bdyBatchSize) * (self.t1 - self.t0) + self.t0
        return np.stack([x, y, z, t], axis=1)

    def spatial_volume(self) -> float:
        return (self.b - self.a) * (self.d - self.c) * (self.f - self.e)

    def bottom_area(self) -> float:
        return (self.b - self.a) * (self.d - self.c)

    def spacetime_measure(self) -> float:
        return self.spatial_volume() * (self.t1 - self.t0)

    def bottom_spacetime_measure(self) -> float:
        return self.bottom_area() * (self.t1 - self.t0)

    def uniform_spatial_grid(self) -> np.ndarray:
        xs = np.linspace(self.a, self.b, self.numQuad)
        ys = np.linspace(self.c, self.d, self.numQuad)
        zs = np.linspace(self.e, self.f, self.numQuad)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def uniform_bottom_grid(self) -> np.ndarray:
        xs = np.linspace(self.a, self.b, self.numQuad)
        ys = np.linspace(self.c, self.d, self.numQuad)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        zz = np.full(xx.size, self.e, dtype=np.float64)
        return np.column_stack([xx.ravel(), yy.ravel(), zz])


# -----------------------------------------------------------------------------
# Residual network, with internal affine normalization to [-1,1]^4
# -----------------------------------------------------------------------------
class RitzNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        lower: Sequence[float],
        upper: Sequence[float],
    ):
        super().__init__()

        lower_t = torch.tensor(lower, dtype=torch.get_default_dtype())
        upper_t = torch.tensor(upper, dtype=torch.get_default_dtype())
        self.register_buffer("lower", lower_t)
        self.register_buffer("upper", upper_t)

        self.inp = nn.Linear(in_dim, width)
        nn.init.xavier_uniform_(self.inp.weight)

        blocks = []
        for _ in range(depth):
            l1 = nn.Linear(width, width)
            l2 = nn.Linear(width, width)
            nn.init.xavier_uniform_(l1.weight)
            nn.init.xavier_uniform_(l2.weight)
            blocks.extend([l1, l2])
        self.blocks = nn.ModuleList(blocks)

        self.outp = nn.Linear(width, out_dim)
        nn.init.xavier_uniform_(self.outp.weight)

    def normalize(self, x_phys: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x_phys - self.lower) / (self.upper - self.lower) - 1.0

    def forward_normalized(self, x_hat: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.inp(x_hat))
        for i in range(0, len(self.blocks), 2):
            y = torch.tanh(self.blocks[i](x))
            y = torch.tanh(self.blocks[i + 1](y))
            x = x + y
        return self.outp(x)

    def forward(self, x_phys: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(self.normalize(x_phys))


# -----------------------------------------------------------------------------
# Prescribed trajectory and exact field
# -----------------------------------------------------------------------------
def center_trajectory(t: np.ndarray, p) -> Tuple[np.ndarray, np.ndarray]:
    mode = p.traj

    if mode == "linear":
        return p.x0 + p.vx * t, p.y0 + p.vy * t

    if mode == "circular":
        phase = p.omega * t + p.phi
        return p.x0 + p.R * np.cos(phase), p.y0 + p.R * np.sin(phase)

    if mode == "parabolic":
        return (
            p.x0 + p.vx * t + p.ax * t * t,
            p.y0 + p.vy * t + p.ay * t * t,
        )

    raise ValueError(f"Unknown trajectory '{mode}'")


def uexact_time_np(xyzt: np.ndarray, p) -> np.ndarray:
    """Exact moving-vortex field, shape (N,4) -> (N,3)."""
    x = xyzt[:, 0]
    y = xyzt[:, 1]
    z = xyzt[:, 2]
    t = xyzt[:, 3]

    cx, cy = center_trajectory(t, p)
    X = x - cx
    Y = y - cy

    ga = np.exp(-(X * X + Y * Y + z * z) / 49.0)

    out = np.zeros((xyzt.shape[0], 3), dtype=np.float64)
    out[:, 0] = 2.0 * Y * ga - 0.5 * p.epsilon * X * z
    out[:, 1] = -2.0 * X * ga - 0.5 * p.epsilon * Y * z
    out[:, 2] = 0.5 * p.epsilon * z * z
    return out


# -----------------------------------------------------------------------------
# Metrics / derivatives
# -----------------------------------------------------------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(d * d))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-14))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def physical_divergence_autograd(
    model: RitzNet,
    x_phys: torch.Tensor,
    create_graph: bool,
):
    """Return network output and physical spatial divergence.

    Network coordinates are (x,y,z,t), normalized to [-1,1]^4.  Continuity
    contains only spatial derivatives,

        div u = u_x + v_y + w_z,

    so no derivative with respect to t is included.
    """
    x_hat = model.normalize(x_phys).detach().requires_grad_(True)
    u = model.forward_normalized(x_hat)

    scales = 2.0 / (model.upper - model.lower)
    spatial_terms = []
    for comp in range(3):
        grad_comp = torch.autograd.grad(
            outputs=u[:, comp],
            inputs=x_hat,
            grad_outputs=torch.ones_like(u[:, comp]),
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0]
        spatial_terms.append(scales[comp] * grad_comp[:, comp])

    div_phys = spatial_terms[0] + spatial_terms[1] + spatial_terms[2]
    return u, div_phys


def divergence_rms(xyzt_t: torch.Tensor, model: RitzNet) -> float:
    was_training = model.training
    model.eval()
    with torch.enable_grad():
        _, div = physical_divergence_autograd(
            model=model,
            x_phys=xyzt_t,
            create_graph=False,
        )
        value = float(torch.sqrt(torch.mean(div.detach() ** 2)).cpu())
    if was_training:
        model.train()
    return value


@torch.no_grad()
def bottom_rms(bottom_xyzt_t: torch.Tensor, model: RitzNet) -> float:
    w = model(bottom_xyzt_t)[:, 2]
    return float(torch.sqrt(torch.mean(w * w)).cpu())


def safe_torch_load(path: str, device: torch.device):
    """Load checkpoints on both old and new PyTorch releases."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def time_tag(t: float) -> str:
    return f"{t:.6g}".replace("-", "m").replace(".", "p")


def parse_eval_times(spec: Sequence[float], t0: float, t1: float) -> List[float]:
    if not spec:
        spec = [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0]

    lo, hi = min(t0, t1), max(t0, t1)
    out = []
    for tt in spec:
        val = float(tt)
        if val < lo - 1e-12 or val > hi + 1e-12:
            raise ValueError(f"evaluation time {val} outside [{t0},{t1}]")
        out.append(val)
    return sorted(set(out))


def save_vectors_for_gnuplot(path: str, xyzt: np.ndarray, uvw: np.ndarray) -> None:
    xyz = xyzt[:, :3]
    np.savetxt(path, np.column_stack([xyz, uvw]), fmt="%.10e")


def maybe_plot_curves(loss_csv: str, plots_dir: str, tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(open(loss_csv, "r")))
        if not rows:
            return

        epochs = [int(r["epoch"]) for r in rows]
        fields = ["loss_total", "loss_data", "loss_div", "loss_bottom"]

        plt.figure()
        for field in fields:
            plt.plot(epochs, [float(r[field]) for r in rows], label=field)
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{tag}_loss.png"), dpi=180)
        plt.close()

        plt.figure()
        plt.plot(epochs, [float(r["val_mse_uv"]) for r in rows], label="val MSE(u,v)")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{tag}_val_mse_uv.png"), dpi=180)
        plt.close()
    except Exception as exc:
        print(f"[WARN] could not create plots: {exc}")


# -----------------------------------------------------------------------------
# Fixed evaluation sets
# -----------------------------------------------------------------------------
def sample_uniform_box(rng, n: int, lower, upper) -> np.ndarray:
    return rng.uniform(np.asarray(lower), np.asarray(upper), size=(n, len(lower)))


def sample_fixed_bottom(rng, n: int, p) -> np.ndarray:
    x = rng.uniform(p.a, p.b, size=n)
    y = rng.uniform(p.c, p.ddd, size=n)
    z = np.full(n, p.e, dtype=np.float64)
    t = rng.uniform(p.t0, p.t1, size=n)
    return np.column_stack([x, y, z, t])


# -----------------------------------------------------------------------------
# One training run
# -----------------------------------------------------------------------------
def train_one_run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    set_seeds(args.seed)

    dom = DomainCubeTime(
        a=args.a,
        b=args.b,
        c=args.c,
        d=args.ddd,
        e=args.e,
        f=args.f,
        t0=args.t0,
        t1=args.t1,
        innerBatchSize=args.innerBatchSize,
        bdyBatchSize=args.bdyBatchSize,
        numQuad=args.numQuad,
    )

    lower = np.array([args.a, args.c, args.e, args.t0], dtype=np.float64)
    upper = np.array([args.b, args.ddd, args.f, args.t1], dtype=np.float64)

    model = RitzNet(
        in_dim=4,
        out_dim=3,
        width=args.width,
        depth=args.depth,
        lower=lower,
        upper=upper,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # ------------------------------------------------------------------
    # Fixed observations, independent from physics collocation points.
    # Only u,v are used by the data term.
    # ------------------------------------------------------------------
    if args.obsSampling == "lhs":
        Xobs = sample_latin_hypercube(args.nObs, lower, upper, args.obsSeed)
    else:
        rng_obs = np.random.RandomState(args.obsSeed)
        Xobs = rng_obs.uniform(lower, upper, size=(args.nObs, 4))

    Uobs = uexact_time_np(Xobs, args)
    Xobs_t = torch.from_numpy(Xobs).to(device)
    Uobs_t = torch.from_numpy(Uobs[:, :2]).to(device)

    # Fixed validation and test sets.  Their exact w values are available only
    # for diagnostics; checkpoint selection uses val MSE(u,v) exclusively.
    rng_eval = np.random.RandomState(args.evalSeed)
    Xval = sample_uniform_box(rng_eval, args.nVal, lower, upper)
    Xtest = sample_uniform_box(rng_eval, args.nTest, lower, upper)
    Xval_bottom = sample_fixed_bottom(rng_eval, args.nValBottom, args)
    Xtest_bottom = sample_fixed_bottom(rng_eval, args.nTestBottom, args)

    Uval = uexact_time_np(Xval, args)
    Utest = uexact_time_np(Xtest, args)

    Xval_t = torch.from_numpy(Xval).to(device)
    Xtest_t = torch.from_numpy(Xtest).to(device)
    Xtest_bottom_t = torch.from_numpy(Xtest_bottom).to(device)

    # Output tree
    logs_dir = os.path.join(args.outdir, "logs")
    plots_dir = os.path.join(args.outdir, "plots")
    summary_dir = os.path.join(args.outdir, "summary")
    checkpoint_dir = os.path.join(args.outdir, "checkpoints")
    observations_dir = os.path.join(args.outdir, "observations")
    time_dir = os.path.join(args.outdir, "time_slices", args.tag)
    for d in [logs_dir, plots_dir, summary_dir, checkpoint_dir, observations_dir, time_dir]:
        os.makedirs(d, exist_ok=True)

    observation_file = os.path.join(observations_dir, f"{args.tag}_observations.csv")
    with open(observation_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "t", "u_obs", "v_obs"])
        for x, uv in zip(Xobs, Uobs[:, :2]):
            w.writerow([x[0], x[1], x[2], x[3], uv[0], uv[1]])

    best_model_path = os.path.join(checkpoint_dir, f"{args.tag}_best.pt")
    loss_csv = os.path.join(logs_dir, f"{args.tag}_loss.csv")
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "lr", "loss_total", "loss_data", "loss_div", "loss_bottom",
            "val_mse_uv", "val_mse_uvw_diag", "val_relL2_diag"
        ])

    best_val_mse_uv = float("inf")
    best_val_mse_uvw = float("nan")
    best_val_relL2 = float("nan")
    best_epoch = -1

    train_start = time.time()

    for epoch in range(args.nEpoch):
        model.train()
        accum = {"total": 0.0, "data": 0.0, "div": 0.0, "bottom": 0.0}

        for _ in range(args.nBatch):
            # Physics collocation sets are independent from Xobs and resampled
            # every optimization batch.
            x_col = torch.from_numpy(dom.sample_from_domain()).to(device)
            x_bdy = torch.from_numpy(dom.sample_from_bottom()).to(device)

            pred_obs = model(Xobs_t)
            loss_data = 0.5 * torch.mean(
                (pred_obs[:, 0] - Uobs_t[:, 0]) ** 2
                + (pred_obs[:, 1] - Uobs_t[:, 1]) ** 2
            )

            _, div_phys = physical_divergence_autograd(
                model=model,
                x_phys=x_col,
                create_graph=True,
            )
            loss_div = args.penalty * torch.mean(div_phys ** 2)

            w_bottom = model(x_bdy)[:, 2]
            loss_bottom = args.penalty2 * torch.mean(w_bottom ** 2)

            loss = loss_data + loss_div + loss_bottom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum["data"] += float(loss_data.detach().cpu())
            accum["div"] += float(loss_div.detach().cpu())
            accum["bottom"] += float(loss_bottom.detach().cpu())
            accum["total"] += float(loss.detach().cpu())

        # Important: one scheduler step per epoch, as in the final stationary protocol.
        scheduler.step()

        do_val = (epoch % args.valEvery == 0) or (epoch == args.nEpoch - 1)
        do_log = (epoch % args.logEvery == 0) or (epoch == args.nEpoch - 1)

        if do_val:
            model.eval()
            with torch.no_grad():
                Uval_pred = model(Xval_t).detach().cpu().numpy()

            val_mse_uv = mse(Uval_pred[:, :2], Uval[:, :2])
            val_mse_uvw = mse(Uval_pred, Uval)
            val_relL2 = rel_l2(Uval_pred, Uval)

            # Selection criterion is observable validation error only.
            if val_mse_uv < best_val_mse_uv:
                best_val_mse_uv = val_mse_uv
                best_val_mse_uvw = val_mse_uvw
                best_val_relL2 = val_relL2
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_mse_uv": float(val_mse_uv),
                        "val_mse_uvw_diag": float(val_mse_uvw),
                        "val_relL2_diag": float(val_relL2),
                    },
                    best_model_path,
                )

        if do_log:
            if not do_val:
                model.eval()
                with torch.no_grad():
                    Uval_log = model(Xval_t).detach().cpu().numpy()
                log_val_mse_uv = mse(Uval_log[:, :2], Uval[:, :2])
                log_val_mse_uvw = mse(Uval_log, Uval)
                log_val_relL2 = rel_l2(Uval_log, Uval)
            else:
                log_val_mse_uv = val_mse_uv
                log_val_mse_uvw = val_mse_uvw
                log_val_relL2 = val_relL2

            lr_now = optimizer.param_groups[0]["lr"]
            with open(loss_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    epoch,
                    lr_now,
                    accum["total"] / args.nBatch,
                    accum["data"] / args.nBatch,
                    accum["div"] / args.nBatch,
                    accum["bottom"] / args.nBatch,
                    log_val_mse_uv,
                    log_val_mse_uvw,
                    log_val_relL2,
                ])

            print(
                f"Epoch {epoch:4d} | loss={accum['total']/args.nBatch:.4e} "
                f"(data={accum['data']/args.nBatch:.2e}, "
                f"div={accum['div']/args.nBatch:.2e}, "
                f"bottom={accum['bottom']/args.nBatch:.2e}) | "
                f"val MSE(u,v)={log_val_mse_uv:.3e}, "
                f"val MSE(u,v,w)[diag]={log_val_mse_uvw:.3e}, "
                f"best epoch={best_epoch}"
            )

    train_seconds = time.time() - train_start

    # Reload validation-selected checkpoint before any final evaluation.
    checkpoint = safe_torch_load(best_model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if not args.noPlots:
        maybe_plot_curves(loss_csv, plots_dir, args.tag)

    # ------------------------------------------------------------------
    # Fixed random space-time test set
    # ------------------------------------------------------------------
    with torch.no_grad():
        Utest_pred = model(Xtest_t).detach().cpu().numpy()

    test_mse_uv = mse(Utest_pred[:, :2], Utest[:, :2])
    test_mse_uvw = mse(Utest_pred, Utest)
    test_relL2 = rel_l2(Utest_pred, Utest)
    test_rms_div = divergence_rms(Xtest_t, model)
    test_rms_bottom = bottom_rms(Xtest_bottom_t, model)

    rms_uv_exact = rms(Utest[:, :2])
    rms_uvw_exact = rms(Utest)
    rel_uv = math.sqrt(test_mse_uv / (rms_uv_exact ** 2 + 1e-14))
    rel_uvw = math.sqrt(test_mse_uvw / (rms_uvw_exact ** 2 + 1e-14))

    # ------------------------------------------------------------------
    # Deterministic time-slice diagnostics, evaluated only after reloading the
    # best checkpoint. These times lie inside the training time interval and
    # are visualization/diagnostic slices, not temporal extrapolation points.
    # ------------------------------------------------------------------
    eval_times = parse_eval_times(args.eval_times, args.t0, args.t1)
    spatial_xyz = dom.uniform_spatial_grid()
    bottom_xyz = dom.uniform_bottom_grid()
    time_metrics_csv = os.path.join(summary_dir, f"{args.tag}_time_metrics.csv")

    time_rows = []
    for t_eval in eval_times:
        xyzt = np.column_stack([spatial_xyz, np.full(spatial_xyz.shape[0], t_eval)])
        xyzt_bottom = np.column_stack([bottom_xyz, np.full(bottom_xyz.shape[0], t_eval)])

        xyzt_t = torch.from_numpy(xyzt).to(device)
        bottom_t = torch.from_numpy(xyzt_bottom).to(device)

        exact = uexact_time_np(xyzt, args)
        with torch.no_grad():
            approx = model(xyzt_t).detach().cpu().numpy()

        mse_uv_t = mse(approx[:, :2], exact[:, :2])
        mse_uvw_t = mse(approx, exact)
        rel_uv_t = rel_l2(approx[:, :2], exact[:, :2])
        rel_uvw_t = rel_l2(approx, exact)
        rms_div_t = divergence_rms(xyzt_t, model)
        rms_bottom_t = bottom_rms(bottom_t, model)

        tag_t = time_tag(t_eval)
        exact_file = os.path.join(time_dir, f"vector_exact_t{tag_t}.dat")
        approx_file = os.path.join(time_dir, f"vector_approx_t{tag_t}.dat")
        cmp_file = os.path.join(time_dir, f"vector_cmp_error_t{tag_t}.dat")

        save_vectors_for_gnuplot(exact_file, xyzt, exact)
        save_vectors_for_gnuplot(approx_file, xyzt, approx)

        err = np.sqrt(np.sum((approx - exact) ** 2, axis=1))
        cmp = np.column_stack([spatial_xyz, exact, approx, err])
        np.savetxt(
            cmp_file,
            cmp,
            fmt="%.10e",
            header="x y z u_exact v_exact w_exact u_app v_app w_app error",
        )

        row = {
            "seed": args.seed,
            "t": t_eval,
            "mse_uv": mse_uv_t,
            "mse_uvw": mse_uvw_t,
            "relL2_uv": rel_uv_t,
            "relL2_uvw": rel_uvw_t,
            "rms_div": rms_div_t,
            "rms_bottom": rms_bottom_t,
            "exact_file": exact_file,
            "approx_file": approx_file,
            "comparison_file": cmp_file,
        }
        time_rows.append(row)

        print(
            f"[TIME] t={t_eval: .2f} | MSEuv={mse_uv_t:.3e} "
            f"MSEuvw={mse_uvw_t:.3e} relL2uvw={rel_uvw_t:.3e} "
            f"RMSdiv={rms_div_t:.3e} RMSbottom={rms_bottom_t:.3e}"
        )

    with open(time_metrics_csv, "w", newline="") as f:
        fields = list(time_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(time_rows)

    # ------------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------------
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary_csv = os.path.join(summary_dir, f"{args.tag}_eval_summary.csv")
    summary = {
        "seed": args.seed,
        "traj": args.traj,
        "x0": args.x0,
        "y0": args.y0,
        "vx": args.vx,
        "vy": args.vy,
        "ax": args.ax,
        "ay": args.ay,
        "R": args.R,
        "omega": args.omega,
        "phi": args.phi,
        "a": args.a,
        "b": args.b,
        "c": args.c,
        "ddd": args.ddd,
        "e": args.e,
        "f": args.f,
        "t0": args.t0,
        "t1": args.t1,
        "width": args.width,
        "depth": args.depth,
        "n_params": n_params,
        "nEpoch": args.nEpoch,
        "nBatch": args.nBatch,
        "innerBatchSize": args.innerBatchSize,
        "bdyBatchSize": args.bdyBatchSize,
        "nObs": args.nObs,
        "obsSampling": args.obsSampling,
        "obsSeed": args.obsSeed,
        "evalSeed": args.evalSeed,
        "lr": args.lr,
        "decay": args.decay,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "beta1": args.penalty,
        "beta2": args.penalty2,
        "best_epoch": best_epoch,
        "best_val_mse_uv": best_val_mse_uv,
        "best_val_mse_uvw_diag": best_val_mse_uvw,
        "best_val_relL2_diag": best_val_relL2,
        "test_mse_uv": test_mse_uv,
        "test_mse_uvw": test_mse_uvw,
        "test_relL2": test_relL2,
        "test_rms_div": test_rms_div,
        "test_rms_bottom": test_rms_bottom,
        "test_rel_uv": rel_uv,
        "test_rel_uvw": rel_uvw,
        "train_seconds": train_seconds,
        "checkpoint_path": best_model_path,
        "observation_file": observation_file,
        "loss_csv": loss_csv,
        "time_metrics_csv": time_metrics_csv,
    }

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print("\n=== BEST VALIDATION CHECKPOINT ===")
    print(f"best epoch               = {best_epoch}")
    print(f"best val MSE(u,v)        = {best_val_mse_uv:.6e}  [selection criterion]")
    print(f"best val MSE(u,v,w)      = {best_val_mse_uvw:.6e}  [diagnostic only]")
    print(f"checkpoint               = {best_model_path}")
    print("\n=== FIXED SPACE-TIME TEST ===")
    print(f"test MSE(u,v)            = {test_mse_uv:.6e}")
    print(f"test MSE(u,v,w)          = {test_mse_uvw:.6e}")
    print(f"test relL2               = {test_relL2:.6e}")
    print(f"test RMS(div)            = {test_rms_div:.6e}")
    print(f"test RMS(bottom)         = {test_rms_bottom:.6e}")
    print(f"train time (s)           = {train_seconds:.2f}")
    print(f"trainable parameters     = {n_params}")
    print(f"summary CSV              = {summary_csv}")
    print(f"time metrics CSV         = {time_metrics_csv}")

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DeepWind moving-vortex benchmark with final fixed-observation/autograd protocol"
    )

    # Domain -- defaults are the parabolic benchmark used in the manuscript.
    ap.add_argument("--a", type=float, default=-25.0)
    ap.add_argument("--b", type=float, default=25.0)
    ap.add_argument("--c", type=float, default=-25.0)
    ap.add_argument("--ddd", type=float, default=25.0)
    ap.add_argument("--e", type=float, default=0.0)
    ap.add_argument("--f", type=float, default=7.0)
    ap.add_argument("--t0", type=float, default=-1.0)
    ap.add_argument("--t1", type=float, default=1.0)
    ap.add_argument("--numQuad", type=int, default=10)

    # Trajectory
    ap.add_argument("--traj", choices=["linear", "circular", "parabolic"], default="parabolic")
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--vx", type=float, default=20.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--ax", type=float, default=0.0)
    ap.add_argument("--ay", type=float, default=30.0)
    ap.add_argument("--R", type=float, default=2.0)
    ap.add_argument("--omega", type=float, default=2.0 * math.pi)
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--epsilon", type=float, default=0.02)

    # Frozen architecture/training protocol
    ap.add_argument("--width", type=int, default=60)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--nEpoch", type=int, default=2000)
    ap.add_argument("--nBatch", type=int, default=25)
    ap.add_argument("--innerBatchSize", type=int, default=512)
    ap.add_argument("--bdyBatchSize", type=int, default=64)

    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--decay", type=float, default=1e-6)
    ap.add_argument("--step_size", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.5)

    ap.add_argument("--penalty", type=float, default=1.0, help="normalized beta1")
    ap.add_argument("--penalty2", type=float, default=1.0, help="normalized beta2")

    # Fixed observations + evaluation
    ap.add_argument("--nObs", type=int, default=1024)
    ap.add_argument("--obsSampling", choices=["lhs", "iid"], default="lhs")
    ap.add_argument("--obsSeed", type=int, default=12345)
    ap.add_argument("--nVal", type=int, default=5000)
    ap.add_argument("--nTest", type=int, default=20000)
    ap.add_argument("--nValBottom", type=int, default=5000)
    ap.add_argument("--nTestBottom", type=int, default=10000)
    ap.add_argument("--evalSeed", type=int, default=54321)

    ap.add_argument(
        "--eval_times",
        nargs="+",
        type=float,
        default=None,
        help="diagnostic times inside [t0,t1]; defaults to -1 -0.8 -0.5 -0.2 0 0.2 0.5 0.8 1",
    )

    # Run controls / IO
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valEvery", type=int, default=10)
    ap.add_argument("--logEvery", type=int, default=30)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--outdir", type=str, default="out_time_final_protocol")
    ap.add_argument("--tag", type=str, default="timefinal_parabolic_s0")
    ap.add_argument("--noPlots", action="store_true")

    args = ap.parse_args()
    if args.nObs < 1 or args.innerBatchSize < 1 or args.bdyBatchSize < 1:
        ap.error("nObs, innerBatchSize and bdyBatchSize must be >= 1")
    if args.t1 <= args.t0:
        ap.error("t1 must be greater than t0")
    return args


def print_run_config(args: argparse.Namespace, device: torch.device) -> None:
    lower = np.array([args.a, args.c, args.e, args.t0], dtype=float)
    upper = np.array([args.b, args.ddd, args.f, args.t1], dtype=float)
    q_measure = np.prod(upper - lower)
    gamma_measure = (args.b - args.a) * (args.ddd - args.c) * (args.t1 - args.t0)

    # Parameter count for the current architecture.
    tmp = RitzNet(4, 3, args.width, args.depth, lower, upper)
    n_params = sum(p.numel() for p in tmp.parameters() if p.requires_grad)

    print("\n================ FINAL TIME PROTOCOL ================")
    print(f"device                 : {device}")
    print(f"dtype                  : {torch.get_default_dtype()}")
    print(f"domain                 : x=[{args.a},{args.b}], y=[{args.c},{args.ddd}], z=[{args.e},{args.f}]")
    print(f"time interval          : [{args.t0},{args.t1}]")
    print(f"trajectory             : {args.traj}")
    print(f"center parameters      : x0={args.x0}, y0={args.y0}, vx={args.vx}, vy={args.vy}, ax={args.ax}, ay={args.ay}")
    print(f"epsilon                : {args.epsilon}")
    print(f"network                : width={args.width}, depth={args.depth}, params={n_params}")
    print(f"fixed observations     : Nobs={args.nObs}, {args.obsSampling}, obsSeed={args.obsSeed}, only (u,v)")
    print(f"physics collocation    : N_Omega={args.innerBatchSize}, N_b={args.bdyBatchSize}, re-sampled each batch")
    print(f"training               : epochs={args.nEpoch}, batches/epoch={args.nBatch}")
    print(f"optimizer              : Adam lr={args.lr}, decay={args.decay}")
    print(f"scheduler              : StepLR step={args.step_size}, gamma={args.gamma}, once/epoch")
    print(f"beta1,beta2            : {args.penalty}, {args.penalty2}")
    print(f"space-time |Q|         : {q_measure:.6e}")
    print(f"bottom-time measure    : {gamma_measure:.6e}")
    print(f"equiv beta1_eff        : {2*args.penalty/q_measure:.6e}")
    print(f"equiv beta2_eff        : {2*args.penalty2/gamma_measure:.6e}")
    print("derivatives            : autograd in normalized coordinates + physical chain rule")
    print("checkpoint criterion   : MSE_val(u,v) ONLY")
    print(f"validation/test seed   : {args.evalSeed}")
    print(f"output                 : {args.outdir} / tag={args.tag}")
    print("=====================================================\n")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.outdir, exist_ok=True)
    print_run_config(args, device)
    train_one_run(args)


if __name__ == "__main__":
    main()
