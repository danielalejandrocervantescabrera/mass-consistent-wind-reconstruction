#!/usr/bin/env python3
"""deepWind3D_vortex_I.py

DeepWind 3D vortex (Example I) – single-run trainer.

Penalty approximation of the constrained variational reconstruction problem.

Normalized penalty objective:
  J(theta) = 1/2 * D_data
           + beta1 * Q_div
           + beta2 * B_bottom

where
  D_data   = mean[(u-u_obs)^2 + (v-v_obs)^2],
  Q_div    = mean[(div u_theta)^2],
  B_bottom = mean[w_theta^2 on Gamma_b].

Equivalently, this is implemented through the variational-integral form
  J(theta) = J_data
           + beta1_eff/2 * int_Omega (div u_theta)^2 dx
           + beta2_eff/2 * int_Gamma_b w_theta^2 dS,
with
  beta1_eff = 2*beta1/|Omega|,
  beta2_eff = 2*beta2/|Gamma_b|.

Therefore the command-line values --penalty 1 --penalty2 1 retain the same
normalized weighting of the revised functional; the effective
coefficients are computed internally from the corresponding domain measures.

Observation points and physics collocation points are treated as distinct sets.
A fixed observation set X_obs = {x_i}_{i=1}^{N_obs} is generated once at the
beginning of the run and is kept unchanged throughout training. By default,
the sensor locations are generated with randomized Latin Hypercube Sampling
(LHS), which provides stratified marginal coverage of the three-dimensional
domain while retaining irregular observation locations. IID uniform sampling
can also be selected with --obsSampling iid. Only the horizontal components
(u,v) are supplied at those locations. Independently, interior collocation
points are re-sampled every optimization batch to approximate the divergence
penalty, and bottom-boundary points are re-sampled to approximate the boundary
penalty.

The neural network receives normalized coordinates (xi,eta,zeta) in [-1,1]^3.
Spatial derivatives are computed exactly with PyTorch automatic differentiation
with respect to normalized coordinates and are converted to physical
derivatives by the chain rule.

This script is sweep-friendly:
  - writes per-epoch CSV: <outdir>/logs/<tag>_loss.csv
  - writes plots:         <outdir>/plots/<tag>_loss.png and <tag>_val_mse_uv.png
  - writes eval summary:  <outdir>/summary/<tag>_eval_summary.csv

Console output (every --logEvery epochs) matches the format:
  Epoch    0 | loss=... (data=..., div=..., bottom=...) | val MSE(u,v)=...

Notes:
  - Scheduler: StepLR is stepped ONCE per epoch.
  - Observation locations are fixed throughout training and controlled by --obsSeed.
  - Observation sampling is selected by --obsSampling {lhs,iid}; default: lhs.
  - Interior/bottom collocation points are re-sampled every optimization batch.
  - Network initialization/stochastic collocation are controlled by --seed.
  - Validation/test locations are fixed and controlled independently by --evalSeed.

Example:
  python3 deepWind3D_vortex_autograd_fixed_observations_lhs.py \
    --nObs 256 --obsSampling lhs --innerBatchSize 1024 --bdyBatchSize 64 \
    --penalty 1.0 --penalty2 1.0 --seed 0 --obsSeed 12345 --evalSeed 54321
"""

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

# ------------------------
#  Global dtype: float64
# ------------------------
torch.set_default_dtype(torch.float64)


# ------------------------
#  Reproducibility
# ------------------------
def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_latin_hypercube(n: int, lower, upper, seed: int) -> np.ndarray:
    """Randomized Latin Hypercube sample in a rectangular box.

    For each coordinate independently, the interval [0,1] is divided into n
    equiprobable strata. Exactly one point is placed in each stratum, and the
    strata are independently permuted across coordinates. The resulting sample
    is then mapped affinely to [lower, upper].

    This is a simple dependency-free LHS implementation. It guarantees
    stratification in each one-dimensional marginal; it is not a tensor grid.
    """
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


# ------------------------
#  Domain
# ------------------------
@dataclass
class DomainCube:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    innerBatchSize: int
    bdyBatchSize: int
    numQuad: int

    def sample_from_domain(self) -> np.ndarray:
        x = np.random.rand(self.innerBatchSize) * (self.b - self.a) + self.a
        y = np.random.rand(self.innerBatchSize) * (self.d - self.c) + self.c
        z = np.random.rand(self.innerBatchSize) * (self.f - self.e) + self.e
        return np.stack([x, y, z], axis=1)

    def sample_from_domain_fd(self, diff_hat: float) -> np.ndarray:
        """Sample physical points whose normalized coordinates can be shifted by diff_hat."""
        if diff_hat <= 0.0 or diff_hat >= 2.0:
            raise ValueError("diff_hat must satisfy 0 < diff_hat < 2")

        # If xi = 2(x-a)/(b-a)-1, then a normalized shift diff_hat
        # corresponds to a physical shift 0.5*(b-a)*diff_hat.
        hx = 0.5 * (self.b - self.a) * diff_hat
        hy = 0.5 * (self.d - self.c) * diff_hat
        hz = 0.5 * (self.f - self.e) * diff_hat

        x = np.random.rand(self.innerBatchSize) * (self.b - self.a - hx) + self.a
        y = np.random.rand(self.innerBatchSize) * (self.d - self.c - hy) + self.c
        z = np.random.rand(self.innerBatchSize) * (self.f - self.e - hz) + self.e
        return np.stack([x, y, z], axis=1)

    def sample_from_bottom(self) -> np.ndarray:
        x = np.random.rand(self.bdyBatchSize) * (self.b - self.a) + self.a
        y = np.random.rand(self.bdyBatchSize) * (self.d - self.c) + self.c
        z = np.full(self.bdyBatchSize, self.e)
        return np.stack([x, y, z], axis=1)

    def area_bottom(self) -> float:
        return (self.b - self.a) * (self.d - self.c)

    def volume(self) -> float:
        return (self.b - self.a) * (self.d - self.c) * (self.f - self.e)

    def uniform_eval_grid(self) -> np.ndarray:
        xs = np.linspace(self.a, self.b, self.numQuad)
        ys = np.linspace(self.c, self.d, self.numQuad)
        zs = np.linspace(self.e, self.f, self.numQuad)
        arr = np.zeros((self.numQuad ** 3, 3), dtype=np.float64)
        k = 0
        for i in range(self.numQuad):
            for j in range(self.numQuad):
                for m in range(self.numQuad):
                    arr[k] = [xs[i], ys[j], zs[m]]
                    k += 1
        return arr


# ------------------------
#  Network
# ------------------------
class RitzNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        bounds,
    ):
        super().__init__()
        a, b, c, d, e, f = bounds

        lower = torch.tensor([a, c, e], dtype=torch.get_default_dtype())
        upper = torch.tensor([b, d, f], dtype=torch.get_default_dtype())
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)

        self.inp = nn.Linear(in_dim, width)
        nn.init.xavier_uniform_(self.inp.weight)

        blocks = []
        for _ in range(depth):
            l1 = nn.Linear(width, width)
            l2 = nn.Linear(width, width)
            nn.init.xavier_uniform_(l1.weight)
            nn.init.xavier_uniform_(l2.weight)
            blocks.append(l1)
            blocks.append(l2)
        self.blocks = nn.ModuleList(blocks)

        self.outp = nn.Linear(width, out_dim)
        nn.init.xavier_uniform_(self.outp.weight)

    def normalize(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Affine map from the physical domain to [-1,1]^3."""
        return 2.0 * (x_phys - self.lower) / (self.upper - self.lower) - 1.0

    def forward_normalized(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Evaluate the network with normalized coordinates."""
        x = torch.tanh(self.inp(x_hat))
        for i in range(0, len(self.blocks), 2):
            y = torch.tanh(self.blocks[i](x))
            y = torch.tanh(self.blocks[i + 1](y))
            x = x + y
        return self.outp(x)

    def forward(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Evaluate from physical coordinates; normalization is internal."""
        return self.forward_normalized(self.normalize(x_phys))


# ------------------------
#  Exact field (Example I)
# ------------------------
def uexact_np(xyz: np.ndarray, epsilon: float) -> np.ndarray:
    """Return exact u(x,y,z) as numpy (N,3)."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    ga = np.exp(-(x * x + y * y + z * z) / 49.0)

    out = np.zeros_like(xyz, dtype=np.float64)
    out[:, 0] = 2.0 * y * ga - 0.5 * epsilon * x * z
    out[:, 1] = -2.0 * x * ga - 0.5 * epsilon * y * z
    out[:, 2] = 0.5 * z * z * epsilon
    return out


# ------------------------
#  Metrics
# ------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(d * d))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b)) + 1e-14
    return num / den


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def physical_divergence_autograd(
    model: nn.Module,
    x_phys: torch.Tensor,
    create_graph: bool,
):
    """Return model output and physical divergence using autograd.

    The network is evaluated in normalized coordinates
        x_hat_i = 2 (x_i-a_i)/(b_i-a_i) - 1.

    By the chain rule,
        d/dx_i = 2/(b_i-a_i) d/dx_hat_i.

    During training create_graph=True is required because the divergence
    penalty must be differentiated with respect to the network parameters.
    """
    # x_hat is the differentiation variable.  It does not need to remain
    # connected to x_phys because the physical coordinates are collocation
    # locations, not trainable quantities.
    x_hat = model.normalize(x_phys).detach().requires_grad_(True)
    u = model.forward_normalized(x_hat)

    # Chain-rule factors: d/dx = scale_x d/dxi, etc.
    scales = 2.0 / (model.upper - model.lower)

    grads = []
    for comp in range(3):
        grad_comp = torch.autograd.grad(
            outputs=u[:, comp],
            inputs=x_hat,
            grad_outputs=torch.ones_like(u[:, comp]),
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads.append(grad_comp[:, comp])

    div_phys = (
        scales[0] * grads[0]
        + scales[1] * grads[1]
        + scales[2] * grads[2]
    )
    return u, div_phys


def divergence_rms(
    xyz_t: torch.Tensor,
    model: nn.Module,
) -> float:
    """RMS of the physical divergence computed by automatic differentiation."""
    was_training = model.training
    model.eval()

    # We need gradients with respect to coordinates even during evaluation.
    with torch.enable_grad():
        _, div = physical_divergence_autograd(
            model=model,
            x_phys=xyz_t,
            create_graph=False,
        )
        value = float(torch.sqrt(torch.mean(div.detach() ** 2)).cpu())

    if was_training:
        model.train()
    return value


@torch.no_grad()
def bottom_u3_rms(bottom_xyz_t: torch.Tensor, model: nn.Module) -> float:
    u = model(bottom_xyz_t)[:, 2]
    return float(torch.sqrt(torch.mean(u * u)).cpu())


# ------------------------
#  Plotting (optional)
# ------------------------
def maybe_plot_curves(loss_csv: str, out_dir: str, tag: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = []
        loss_total = []
        loss_data = []
        loss_div = []
        loss_bottom = []
        val_mse_uv = []
        val_mse_uvw = []

        with open(loss_csv, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                epochs.append(int(row["epoch"]))
                loss_total.append(float(row["loss_total"]))
                loss_data.append(float(row["loss_data"]))
                loss_div.append(float(row["loss_div"]))
                loss_bottom.append(float(row["loss_bottom"]))
                val_mse_uv.append(float(row["val_mse_uv"]))
                val_mse_uvw.append(float(row["val_mse_uvw"]))

        os.makedirs(out_dir, exist_ok=True)

        plt.figure()
        plt.plot(epochs, loss_total, label="total")
        plt.plot(epochs, loss_data, label="data")
        plt.plot(epochs, loss_div, label="div")
        plt.plot(epochs, loss_bottom, label="bottom")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss (log)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_loss.png"), dpi=180)
        plt.close()

        plt.figure()
        plt.plot(epochs, val_mse_uv, label="val MSE(u,v)")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("MSE (log)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_val_mse_uv.png"), dpi=180)
        plt.close()

        plt.figure()
        plt.plot(epochs, val_mse_uvw, label="val MSE(u,v,w)")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("MSE (log)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_val_mse_uvw.png"), dpi=180)
        plt.close()


    except Exception as e:
        print(f"[WARN] Could not plot curves: {e}")



def relL2(u_pred, u_true, eps=1e-12):
    num = np.linalg.norm(u_pred - u_true)
    den = np.linalg.norm(u_true) + eps
    return num / den
        
# ------------------------
#  Train
# ------------------------
def train_one_run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    set_seeds(args.seed)


    val_mse_uv = float("nan")
    val_mse_uvw = float("nan")
    val_relL2 = float("nan")
    
    
    dom = DomainCube(
        a=args.a,
        b=args.b,
        c=args.c,
        d=args.ddd,
        e=args.e,
        f=args.f,
        innerBatchSize=args.innerBatchSize,
        bdyBatchSize=args.bdyBatchSize,
        numQuad=args.numQuad,
    )

    bounds = (args.a, args.b, args.c, args.ddd, args.e, args.f)
    model = RitzNet(
        in_dim=3,
        out_dim=3,
        width=args.width,
        depth=args.depth,
        bounds=bounds,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    sch = StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    # ------------------------------------------------------------------
    # Fixed observation set, independent of the physics collocation set.
    # Only (u,v) are used as observations; w is never supplied to training.
    # obsSeed is independent of the network/collocation seed so that multiple
    # network seeds can be compared using exactly the same sensor locations.
    # ------------------------------------------------------------------
    if args.nObs < 1:
        raise ValueError("--nObs must be >= 1")

    obs_lower = np.array([args.a, args.c, args.e], dtype=np.float64)
    obs_upper = np.array([args.b, args.ddd, args.f], dtype=np.float64)

    if args.obsSampling == "lhs":
        Xobs = sample_latin_hypercube(
            n=args.nObs,
            lower=obs_lower,
            upper=obs_upper,
            seed=args.obsSeed,
        )
    elif args.obsSampling == "iid":
        rng_obs = np.random.RandomState(args.obsSeed)
        Xobs = rng_obs.uniform(obs_lower, obs_upper, size=(args.nObs, 3))
    else:
        raise ValueError(f"Unsupported observation sampling: {args.obsSampling}")
    Uobs = uexact_np(Xobs, args.epsilon)
    Xobs_t = torch.from_numpy(Xobs).to(device)
    Uobs_t = torch.from_numpy(Uobs).to(device)

    # Fixed validation/test sets, independent of both network seed and obsSeed.
    rng_eval = np.random.RandomState(args.evalSeed)
    Xval = rng_eval.uniform(
        [args.a, args.c, args.e],
        [args.b, args.ddd, args.f],
        size=(args.nVal, 3),
    )
    Xtest = rng_eval.uniform(
        [args.a, args.c, args.e],
        [args.b, args.ddd, args.f],
        size=(args.nTest, 3),
    )
    Uval = uexact_np(Xval, args.epsilon)
    Utest = uexact_np(Xtest, args.epsilon)

    Xval_t = torch.from_numpy(Xval).to(device)
    xtest_t = torch.from_numpy(Xtest).to(device)

    # Output folders
    logs_dir = os.path.join(args.outdir, "logs")
    plots_dir = os.path.join(args.outdir, "plots")
    summary_dir = os.path.join(args.outdir, "summary")
    checkpoint_dir = os.path.join(args.outdir, "checkpoints")
    observations_dir = os.path.join(args.outdir, "observations")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(observations_dir, exist_ok=True)

    # Store only the actually observed quantities.  The exact w component is
    # intentionally not written to this file because it is not training data.
    observation_file = os.path.join(observations_dir, f"{args.tag}_observations.csv")
    with open(observation_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "u_obs", "v_obs"])
        for x, uv in zip(Xobs, Uobs[:, :2]):
            writer.writerow([x[0], x[1], x[2], uv[0], uv[1]])

    best_model_path = os.path.join(checkpoint_dir, f"{args.tag}_best.pt")
    best_val_mse_uv = float("inf")
    best_val_mse_uvw = float("nan")
    best_val_relL2 = float("nan")
    best_epoch = -1

    loss_csv = os.path.join(logs_dir, f"{args.tag}_loss.csv")
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "lr",
            "loss_total",
            "loss_data",
            "loss_div",
            "loss_bottom",
            "val_mse_uv",
            "val_mse_uvw",
            "val_relL2",
        ])

    # Training loop
    t0 = time.time()
    for epoch in range(args.nEpoch):
        model.train()

        loss_data_epoch   = 0.0
        loss_div_epoch    = 0.0
        loss_bottom_epoch = 0.0
        loss_total_epoch  = 0.0

        for _ in range(args.nBatch):
            # Physics collocation points are independent from observations and
            # are re-sampled every batch (Monte Carlo quadrature).
            x_col = torch.from_numpy(dom.sample_from_domain()).to(device)
            x_bdy = torch.from_numpy(dom.sample_from_bottom()).to(device)

            # Data term: always evaluated on the same fixed sensor locations.
            # Only horizontal components are supervised.
            pred_obs = model(Xobs_t)
            loss_data = 0.5 * torch.mean(
                (Uobs_t[:, 0] - pred_obs[:, 0]) ** 2
                + (Uobs_t[:, 1] - pred_obs[:, 1]) ** 2
            )

            # Network output and physical divergence from automatic
            # differentiation in normalized coordinates + chain rule.
            # For the default domain [-7,7]x[-7,7]x[0,7]:
            #   d/dx = (1/7)d/dxi,
            #   d/dy = (1/7)d/deta,
            #   d/dz = (2/7)d/dzeta.
            _, div_phys = physical_divergence_autograd(
                model=model,
                x_phys=x_col,
                create_graph=True,
            )

            # User-facing beta1,beta2 follow the normalization used in the
            # original sensitivity study.  Internally, the integral-form
            # coefficients are rescaled by 2/|Omega|:
            #   beta1_eff = 2 beta1 / |Omega|, beta2_eff = 2 beta2 / |Gamma_b|.
            # Hence
            #   (beta1_eff/2)|Omega| mean(div^2) = beta1 mean(div^2),
            # and
            #   (beta2_eff/2)|Gamma_b| mean(w^2) = beta2 mean(w^2).
            beta1_eff = 2.0 * args.penalty / dom.volume()
            beta2_eff = 2.0 * args.penalty2 / dom.area_bottom()

            loss_div = (
                0.5
                * beta1_eff
                * dom.volume()
                * torch.mean(div_phys ** 2)
            )

            # bottom w
            ob = model(x_bdy)
            loss_bottom = (
                0.5
                * beta2_eff
                * dom.area_bottom()
                * torch.mean(ob[:, 2] ** 2)
            )

            loss = loss_data + loss_div + loss_bottom

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_data_epoch   += float(loss_data.detach().cpu())
            loss_div_epoch    += float(loss_div.detach().cpu())
            loss_bottom_epoch += float(loss_bottom.detach().cpu())
            loss_total_epoch  += float(loss.detach().cpu())


        sch.step()

        # Validation is independent of console logging.  The best checkpoint is
        # selected ONLY with the observable horizontal validation error MSE(u,v).
        do_val = (epoch % args.valEvery == 0) or (epoch == args.nEpoch - 1)
        do_log = (epoch % args.logEvery == 0) or (epoch == args.nEpoch - 1)

        if do_val:
            model.eval()
            with torch.no_grad():
                Upv = model(Xval_t).detach().cpu().numpy()

            val_mse_uv  = mse(Upv[:, :2], Uval[:, :2])
            val_mse_uvw = mse(Upv[:, :3], Uval[:, :3])
            val_relL2   = relL2(Upv[:, :3], Uval[:, :3])

            if val_mse_uv < best_val_mse_uv:
                best_val_mse_uv = val_mse_uv
                best_val_mse_uvw = val_mse_uvw
                best_val_relL2 = val_relL2
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sch.state_dict(),
                    "val_mse_uv": float(val_mse_uv),
                    "val_mse_uvw": float(val_mse_uvw),
                    "val_relL2": float(val_relL2),
                }, best_model_path)

        if do_log:
            # If logEvery and valEvery are not aligned, evaluate here for a
            # self-contained log row without affecting checkpoint selection.
            if not do_val:
                model.eval()
                with torch.no_grad():
                    Upv_log = model(Xval_t).detach().cpu().numpy()
                log_val_mse_uv = mse(Upv_log[:, :2], Uval[:, :2])
                log_val_mse_uvw = mse(Upv_log[:, :3], Uval[:, :3])
                log_val_relL2 = relL2(Upv_log[:, :3], Uval[:, :3])
            else:
                log_val_mse_uv = val_mse_uv
                log_val_mse_uvw = val_mse_uvw
                log_val_relL2 = val_relL2

            lr_now = opt.param_groups[0]["lr"]
            with open(loss_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    epoch, lr_now,
                    loss_total_epoch / args.nBatch,
                    loss_data_epoch / args.nBatch,
                    loss_div_epoch / args.nBatch,
                    loss_bottom_epoch / args.nBatch,
                    log_val_mse_uv, log_val_mse_uvw, log_val_relL2,
                ])

            print(
                f"Epoch {epoch:4d} | loss={loss_total_epoch/args.nBatch:.4e} "
                f"(data={loss_data_epoch/args.nBatch:.2e}, div={loss_div_epoch/args.nBatch:.2e}, bottom={loss_bottom_epoch/args.nBatch:.2e}) "
                f"| val MSE(u,v)={log_val_mse_uv:.3e}, val MSE(u,v,w)={log_val_mse_uvw:.3e}, "
                f"val relL2={log_val_relL2:.3e} | best epoch={best_epoch}"
            )

    train_seconds = time.time() - t0

    # Restore the checkpoint selected by validation MSE(u,v) before testing.
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[INFO] Reloaded best checkpoint: epoch={best_epoch}, "
          f"val MSE(u,v)={best_val_mse_uv:.6e}")

    # Plots
    if not args.noPlots:
        maybe_plot_curves(loss_csv, plots_dir, args.tag)

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        Upt = model(xtest_t).detach().cpu().numpy()

    test_mse_uv       = mse(Upt[:, :2], Utest[:, :2])
    test_mse_uvw      = mse(Upt[:, :3], Utest[:, :3])
    test_rel_l2_field = rel_l2(Upt, Utest)

    # physics metrics
    test_rms_div = divergence_rms(
        xtest_t,
        model,
    )
    bottom_eval        = torch.from_numpy(dom.sample_from_bottom()).to(device)
    test_rms_u3_bottom = bottom_u3_rms(bottom_eval, model)

    # normalization baselines (for "relative" versions)
    rms_speed_exact = rms(Utest)
    rms_uv_exact    = rms(Utest[:, :2])
    rms_uvw_exact   = rms(Utest[:, :3])

    # print(test_mse_uv)
    # print(test_mse_uvw)
    # print(rms_uv_exact)
    # print(rms_uvw_exact)

    rel_rms_div       = test_rms_div / (rms_speed_exact + 1e-14)
    rel_rms_u3_bottom = test_rms_u3_bottom / (rms_speed_exact + 1e-14)
    rel_uv            = math.sqrt(test_mse_uv / (rms_uv_exact ** 2 + 1e-14))
    rel_uvw           = math.sqrt(test_mse_uvw / (rms_uvw_exact ** 2 + 1e-14))

    # print('rel_uv',rel_uv)
    # print('rel_uvw',rel_uvw)

    # Summary CSV
    summary_csv = os.path.join(summary_dir, f"{args.tag}_eval_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "seed",
            "width",
            "depth",
            "nEpoch",
            "nBatch",
            "innerBatchSize",
            "bdyBatchSize",
            "nObs",
            "obsSampling",
            "obsSeed",
            "evalSeed",
            "lr",
            "decay",
            "step_size",
            "gamma",
            "epsilon",
            "beta1",
            "beta2",
            "test_mse_uv",
            "test_mse_uvw",
            "test_relL2_field",
            "test_rms_div",
            "test_rms_u3_bottom",
            "rms_speed_exact",
            "rms_uv_exact",
            "rms_uvw_exact",
            "rel_rms_div",
            "rel_rms_u3_bottom",
            "rel_uv",
            "rel_uvw",
            "best_epoch",
            "best_val_mse_uv",
            "best_val_mse_uvw",
            "best_val_relL2",
            "loss_csv",
            "checkpoint_path",
            "observation_file",
            "train_seconds",
        ])
        w.writerow([
            args.seed,
            args.width,
            args.depth,
            args.nEpoch,
            args.nBatch,
            args.innerBatchSize,
            args.bdyBatchSize,
            args.nObs,
            args.obsSampling,
            args.obsSeed,
            args.evalSeed,
            args.lr,
            args.decay,
            args.step_size,
            args.gamma,
            args.epsilon,
            args.penalty,         # beta1
            args.penalty2,        # beta2
            test_mse_uv,
            test_mse_uvw,
            test_rel_l2_field,
            test_rms_div,
            test_rms_u3_bottom,
            rms_speed_exact,
            rms_uv_exact,
            rms_uvw_exact,
            rel_rms_div,
            rel_rms_u3_bottom,
            rel_uv,
            rel_uvw,
            best_epoch,
            best_val_mse_uv,
            best_val_mse_uvw,
            best_val_relL2,
            loss_csv,
            best_model_path,
            observation_file,
            train_seconds,
        ])
        
    # Print final metrics (requested)
    print("\n=== Best validation checkpoint ===")
    print(f"best epoch              = {best_epoch}")
    print(f"best val MSE(u,v)       = {best_val_mse_uv:.6e}")
    print(f"best val MSE(u,v,w)     = {best_val_mse_uvw:.6e}")
    print(f"checkpoint              = {best_model_path}")
    print(f"fixed observations      = {observation_file}")
    print("\n=== Test metrics from best checkpoint ===")
    print(f"test MSE(u,v)           = {test_mse_uv:.6e}  (rel_uv={rel_uv:.6e})")
    print(f"test MSE(u,v,w)         = {test_mse_uvw:.6e}  (rel_uvw={rel_uvw:.6e})")
    print(f"test relL2(field)       = {test_rel_l2_field:.6e}")
    print(f"test RMS(div)           = {test_rms_div:.6e}  (rel_rms_div={rel_rms_div:.6e})")
    print(f"test RMS(u3)(bottom)    = {test_rms_u3_bottom:.6e}  (rel_rms_u3_bottom={rel_rms_u3_bottom:.6e})")
    print(f"wrote loss CSV          = {loss_csv}")
    print(f"wrote eval summary      = {summary_csv}")
    
    return {
        "loss_csv": loss_csv,
        "summary_csv": summary_csv,
        "checkpoint_path": best_model_path,
        "observation_file": observation_file,
        "nObs": args.nObs,
        "obsSampling": args.obsSampling,
        "best_epoch": best_epoch,
        "best_val_mse_uv": best_val_mse_uv,
        "test_mse_uv": test_mse_uv,
        "test_relL2_field": test_rel_l2_field,
        "test_rms_div": test_rms_div,
        "test_rms_u3_bottom": test_rms_u3_bottom,
        "rel_rms_div": rel_rms_div,
        "rel_rms_u3_bottom": rel_rms_u3_bottom,
        "rel_uv": rel_uv,
        "rel_uvw": rel_uvw,        
    }


# ------------------------
#  Pretty print
# ------------------------
def print_run_config(args: argparse.Namespace, device: torch.device) -> None:
    print("\n============== Run configuration ==============")
    print(f"Device: {device}, default dtype: {torch.get_default_dtype()}")
    print("Domain:")
    print(f"  x ∈ [{args.a}, {args.b}]")
    print(f"  y ∈ [{args.c}, {args.ddd}]")
    print(f"  z ∈ [{args.e}, {args.f}]")
    print(f"Uniform eval grid: numQuad = {args.numQuad}  → {args.numQuad}^3 points")
    print("\nNetwork:")
    print("  input dim = 3, output dim = 3")
    print(f"  width = {args.width}, depth = {args.depth}")
    print("\nTraining:")
    print(f"  nEpoch       = {args.nEpoch}")
    print(f"  nBatch       = {args.nBatch}")
    print(f"  innerBatch   = {args.innerBatchSize}")
    print(f"  bdyBatch     = {args.bdyBatchSize}")
    print(f"  valEvery     = {args.valEvery}")
    print("\nOptimization:")
    print(f"  lr           = {args.lr}")
    print(f"  weight_decay = {args.decay}")
    print(f"  StepLR: step_size = {args.step_size}, gamma = {args.gamma}")
    print("\nPhysics / penalties:")
    print("  derivatives = PyTorch autograd")
    print("  network input scaled to [-1,1]^3")
    print("  physical divergence uses explicit chain-rule factors")
    volume = (args.b - args.a) * (args.ddd - args.c) * (args.f - args.e)
    area_bottom = (args.b - args.a) * (args.ddd - args.c)
    beta1_eff = 2.0 * args.penalty / volume
    beta2_eff = 2.0 * args.penalty2 / area_bottom
    print(f"  beta1 (normalized/user) = {args.penalty}")
    print(f"  beta2 (normalized/user) = {args.penalty2}")
    print(f"  |Omega|                 = {volume:.10e}")
    print(f"  |Gamma_b|               = {area_bottom:.10e}")
    print(f"  beta1_eff = 2 beta1/|Omega|   = {beta1_eff:.10e}")
    print(f"  beta2_eff = 2 beta2/|Gamma_b| = {beta2_eff:.10e}")
    print(f"  epsilon  = {args.epsilon}")
    print("\nSampling / datasets:")
    print(f"  fixed observations N_obs = {args.nObs}")
    print(f"  observation sampling      = {args.obsSampling}")
    print(f"  observation seed          = {args.obsSeed}")
    print(f"  interior collocation      = {args.innerBatchSize} per batch (re-sampled)")
    print(f"  bottom collocation        = {args.bdyBatchSize} per batch (re-sampled)")
    print(f"  nVal                      = {args.nVal}")
    print(f"  nTest                     = {args.nTest}")
    print(f"  evaluation seed           = {args.evalSeed}")
    print("\nOutput:")
    print(f"  outdir = '{args.outdir}'")
    print(f"  tag    = '{args.tag}'")
    print("===============================================\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DeepWind 3D vortex with fixed LHS/IID observations separated from stochastic physics collocation")

    # Domain/grid
    ap.add_argument("--a", type=float, default=-7.0)
    ap.add_argument("--b", type=float, default=7.0)
    ap.add_argument("--c", type=float, default=-7.0)
    ap.add_argument("--ddd", type=float, default=7.0)
    ap.add_argument("--e", type=float, default=0.0)
    ap.add_argument("--f", type=float, default=7.0)
    ap.add_argument("--numQuad", type=int, default=10)

    # Network/training
    ap.add_argument("--width", type=int, default=60)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--nEpoch", type=int, default=1200)
    ap.add_argument("--nBatch", type=int, default=25)
    ap.add_argument("--innerBatchSize", type=int, default=1024)
    ap.add_argument("--bdyBatchSize", type=int, default=64)

    # Optim
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--decay", type=float, default=1e-6)
    ap.add_argument("--step_size", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.5)

    # Physics/penalties
    ap.add_argument("--penalty", type=float, default=1.0, help="normalized beta1; internal integral coefficient is 2*beta1/|Omega|")
    ap.add_argument("--penalty2", type=float, default=1.0, help="normalized beta2; internal integral coefficient is 2*beta2/|Gamma_b|")
    ap.add_argument("--epsilon", type=float, default=0.1)

    # Fixed observations + validation/test
    ap.add_argument("--nObs", type=int, default=256,
                    help="number of fixed observation locations; only u,v are observed")
    ap.add_argument("--obsSampling", type=str, default="lhs", choices=["lhs", "iid"],
                    help="fixed observation layout: randomized Latin hypercube (lhs) or IID uniform (iid)")
    ap.add_argument("--obsSeed", type=int, default=12345,
                    help="seed for the fixed observation/sensor locations")
    ap.add_argument("--nVal", type=int, default=5000)
    ap.add_argument("--nTest", type=int, default=10000)
    ap.add_argument("--evalSeed", type=int, default=54321,
                    help="seed for fixed validation/test locations")

    # IO
    ap.add_argument("--outdir", type=str, default="out")
    ap.add_argument("--tag", type=str, default="fixedobs_lhs_n256_w60_d5_i1024_b64")
    ap.add_argument("--seed", type=int, default=0)

    # Misc
    ap.add_argument("--logEvery", type=int, default=30)
    ap.add_argument("--valEvery", type=int, default=10, help="validation/checkpoint cadence in epochs")
    ap.add_argument("--device", type=str, default="", help="e.g. cuda:0 or cpu")
    ap.add_argument("--noPlots", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.outdir, exist_ok=True)
    print_run_config(args, device)

    t0 = time.time()
    train_one_run(args)
    print(f"[INFO] Total wall time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
