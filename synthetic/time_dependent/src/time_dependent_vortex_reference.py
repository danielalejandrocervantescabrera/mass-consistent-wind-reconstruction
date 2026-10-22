#!/usr/bin/env python3
# DeepWind 3D vortex: time-dependent center trajectory selectable by CLI.
# Trajectory modes:
#   1) linear    : cx = x0 + vx*t,           cy = y0 + vy*t
#   2) circular  : cx = x0 + R*cos(ω t + φ), cy = y0 + R*sin(ω t + φ)
#   3) parabolic : cx = x0 + vx*t + ax*t^2,  cy = y0 + vy*t + ay*t^2
#
# Notes:
# - Field is the same vortex in a moving frame (X,Y)=(x-cx(t), y-cy(t)).
# - w = 0.5*epsilon*z^2 => w=0 at the base z=0 for all times.
#
# Example:
#   python3 deepWind3D_vortex_time_traj.py --traj circular --R 3.0 --omega 6.283185 \
#     --nEpoch 1200 --nBatch 25 --width 80 --depth 5


"""
 python3 deepWind3D_vortex_time_traj_evaltimes_fixed2.py \
   --traj parabolic \
   --x0 0 --y0 0 \
   --vx 20 --vy 0 \
   --ax 0 --ay 30 \
   --a -25 --b 25 \
   --c -25 --ddd 25 \
   --e 0 --f 7 \
   --t0 -1 --t1 1 \
   --epsilon 0.02 \
   --eval_times -1 -0.8 -0.5 -0.2 0 0.2 0.5 0.8 1 \
   --outdir out_time_bigXY_parabolaB_lr1e3 \
   --nEpoch 500 --nBatch 256 \
   --lr 0.001
"""

# python deepWind3D_vortex_time_traj_evaltimes_fixed2.py \
#   --traj parabolic \
#   --vx 2 --vy 0 \
#   --ax 0 --ay 5 \
#   --nEpoch 1000



import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

# ------------------------
#  Domain in space-time
# ------------------------
class DomainCubeTime(object):
    """
    Spatial domain:
        x in [a,b], y in [c,d], z in [e,f]
    Time domain:
        t in [t0, t1]
    """
    def __init__(self, params):
        self.a, self.b = params["a"],   params["b"]
        self.c, self.d = params["c"],   params["ddd"]
        self.e, self.f = params["e"],   params["f"]
        self.t0, self.t1 = params["t0"], params["t1"]

        self.nEval          = params["numQuad"]
        self.innerBatchSize = params["innerBatchSize"]
        self.bdyBatchSize   = params["bdyBatchSize"]

    def sampleFromDomain(self):
        """Uniform random sampling in space-time interior."""
        x = np.random.rand(self.innerBatchSize)*(self.b-self.a) + self.a
        y = np.random.rand(self.innerBatchSize)*(self.d-self.c) + self.c
        z = np.random.rand(self.innerBatchSize)*(self.f-self.e) + self.e
        t = np.random.rand(self.innerBatchSize)*(self.t1-self.t0) + self.t0
        return np.stack([x, y, z, t], axis=1)

    def sampleFromBottom(self):
        """
        Bottom boundary (z = e) in space-time.
        Enforce w(x,y,z=e,t) = 0 for all t.
        """
        x = np.random.rand(self.bdyBatchSize)*(self.b-self.a) + self.a
        y = np.random.rand(self.bdyBatchSize)*(self.d-self.c) + self.c
        z = np.full(self.bdyBatchSize, self.e)
        t = np.random.rand(self.bdyBatchSize)*(self.t1-self.t0) + self.t0
        return np.stack([x, y, z, t], axis=1)

    def areaBottom(self):
        return (self.b-self.a)*(self.d-self.c)*(self.t1-self.t0)

    def volumen(self):
        return (self.b-self.a)*(self.d-self.c)*(self.f-self.e)*(self.t1-self.t0)

    def uniformSpatialGrid(self):
        xs = np.linspace(self.a, self.b, self.nEval)
        ys = np.linspace(self.c, self.d, self.nEval)
        zs = np.linspace(self.e, self.f, self.nEval)
        arr = np.zeros((self.nEval**3, 3))
        k = 0
        for i in range(self.nEval):
            for j in range(self.nEval):
                for m in range(self.nEval):
                    arr[k] = [xs[i], ys[j], zs[m]]
                    k += 1
        return arr

# ------------------------
#  Network
# ------------------------
class RitzNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.p = params
        self.inp = nn.Linear(self.p["d"], self.p["width"])
        nn.init.xavier_uniform_(self.inp.weight)

        self.blocks = nn.ModuleList()
        for _ in range(self.p["depth"]):
            l1 = nn.Linear(self.p["width"], self.p["width"])
            l2 = nn.Linear(self.p["width"], self.p["width"])
            nn.init.xavier_uniform_(l1.weight)
            nn.init.xavier_uniform_(l2.weight)
            self.blocks += [l1, l2]

        self.outp = nn.Linear(self.p["width"], self.p["dd"])
        nn.init.xavier_uniform_(self.outp.weight)

    def forward(self, x):
        x = torch.tanh(self.inp(x))
        for i in range(0, len(self.blocks), 2):
            y = torch.tanh(self.blocks[i](x))
            y = torch.tanh(self.blocks[i+1](y))
            x = x + y
        return self.outp(x)

# ------------------------
#  Trajectory helper
# ------------------------
def center_trajectory(t, p):
    """
    t: numpy array (N,)
    returns (cx, cy) numpy arrays (N,)
    """
    mode = p.get("traj", "linear")

    x0 = p.get("x0", 0.0)
    y0 = p.get("y0", 0.0)
    vx = p.get("vx", 1.0)
    vy = p.get("vy", 0.0)

    if mode == "linear":
        cx = x0 + vx * t
        cy = y0 + vy * t
        return cx, cy

    if mode == "circular":
        R     = p.get("R", 2.0)
        omega = p.get("omega", 2*np.pi)
        phi   = p.get("phi", 0.0)
        cx = x0 + R * np.cos(omega * t + phi)
        cy = y0 + R * np.sin(omega * t + phi)
        return cx, cy

    if mode == "parabolic":
        ax = p.get("ax", 0.0)
        ay = p.get("ay", 0.0)
        cx = x0 + vx * t + ax * (t**2)
        cy = y0 + vy * t + ay * (t**2)
        return cx, cy

    raise ValueError(f"Unknown traj mode '{mode}'. Use linear|circular|parabolic.")

# ------------------------
#  Time-dependent exact field with selectable trajectory
# ------------------------
def uexact_time(data, params):
    """
    data: numpy array (N,4) with columns (x,y,z,t).
    returns: torch tensor (N,3).
    """
    out = np.zeros((data.shape[0], 3), dtype=np.float64)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    t = data[:,3]

    eps = params["epsilon"]

    cx, cy = center_trajectory(t, params)

    # Shifted coordinates (moving vortex frame)
    X = x - cx
    Y = y - cy

    # Gaussian core (same shape as static vortex but in X,Y)
    ga = np.exp(-(X*X + Y*Y + z*z)/49.0)

    # Vortex-like structure
    out[:,0] =  2.0*Y*ga - 0.5*eps*X*z
    out[:,1] = -2.0*X*ga - 0.5*eps*Y*z
    out[:,2] =  0.5*z*z*eps  # ensures w=0 at z=0 for all t

    return torch.from_numpy(out)

# ------------------------
#  Metrics / helpers
# ------------------------
def errorFun(output, target):
    err = target - output
    err = torch.sqrt(torch.mean(err*err))
    ref = torch.sqrt(torch.mean(target*target))
    return (err/ref).item()

@torch.no_grad()
def compute_divergence(data, model, params, device):
    diff = params["diff"]
    sx = torch.tensor([diff,0.0,0.0,0.0], device=device).float()
    sy = torch.tensor([0.0,diff,0.0,0.0], device=device).float()
    sz = torch.tensor([0.0,0.0,diff,0.0], device=device).float()

    u   = model(data)
    ux  = model(data+sx)[:,0]
    vy  = model(data+sy)[:,1]
    wz  = model(data+sz)[:,2]

    du  = (ux - u[:,0]) / diff
    dv  = (vy - u[:,1]) / diff
    dw  = (wz - u[:,2]) / diff
    div = du + dv + dw
    return div.mean().item()

def save_vectors_for_gnuplot(path, xyz_t, uvw_t):
    xyz_t = xyz_t.detach().cpu().numpy() if torch.is_tensor(xyz_t) else np.asarray(xyz_t)
    uvw   = uvw_t.detach().cpu().numpy() if torch.is_tensor(uvw_t) else np.asarray(uvw_t)
    xyz = xyz_t[:, :3]  # drop t
    arr = np.column_stack([xyz, uvw])
    np.savetxt(path, arr, fmt='%.8f')



# ------------------------
#  Eval-times utilities
# ------------------------
def parse_eval_times(spec, t0: float, t1: float):
    """Return a sorted unique list of evaluation times within [t0, t1].

    - If spec is None: returns [t0, (t0+t1)/2, t1]
    - If spec is a list/tuple of floats: validates and returns sorted unique times
    """
    if spec is None:
        return [t0, 0.5*(t0+t1), t1]

    # Accept a single string as a convenience (comma-separated), in case older calls remain.
    if isinstance(spec, str):
        spec = spec.strip()
        if spec == '':
            return [t0, 0.5*(t0+t1), t1]
        try:
            spec = [float(x) for x in spec.split(',') if x.strip() != '']
        except ValueError as e:
            raise ValueError(
                f"Invalid --eval_times string '{spec}'. Use floats, e.g. --eval_times 0 0.5 1"
            ) from e

    times = list(spec)
    lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
    out = []
    for tt in times:
        t = float(tt)
        if t < lo - 1e-12 or t > hi + 1e-12:
            raise ValueError(f"Evaluation time {t} is outside [t0,t1]=[{t0},{t1}].")
        out.append(t)

    return sorted(set(out))
def time_tag(t: float):
    '''Safe filename tag from float time.'''
    s = f"{t:.6g}"
    s = s.replace('-', 'm').replace('.', 'p')
    return s


def rel_error_uv(approx, exact):
    num = torch.sqrt(torch.mean((approx[:,0:2] - exact[:,0:2])**2))
    den = torch.sqrt(torch.mean(exact[:,0:2]**2))
    return (num / (den + 1e-12)).item()

def rel_error_uvw(approx, exact):
    num = torch.sqrt(torch.mean((approx - exact)**2))
    den = torch.sqrt(torch.mean(exact**2))
    return (num / (den + 1e-12)).item()

def div_rms_at_time(domain, model, t_eval, p, device):
    sx = torch.tensor([p["diff"],0,0,0], device=device).float()
    sy = torch.tensor([0,p["diff"],0,0], device=device).float()
    sz = torch.tensor([0,0,p["diff"],0], device=device).float()

    X = torch.from_numpy(domain.sampleFromDomain()).float().to(device)
    X[:,3] = float(t_eval)

    u  = model(X)
    ux = model(X+sx)[:,0]
    vy = model(X+sy)[:,1]
    wz = model(X+sz)[:,2]

    div = (ux-u[:,0])/p["diff"] + (vy-u[:,1])/p["diff"] + (wz-u[:,2])/p["diff"]
    return torch.sqrt(torch.mean(div**2)).item()

def bdy_rms_at_time(domain, model, t_eval, device):
    Xb = torch.from_numpy(domain.sampleFromBottom()).float().to(device)
    Xb[:,3] = float(t_eval)
    w = model(Xb)[:,2]
    return torch.sqrt(torch.mean(w**2)).item()

def compute_metrics_fixed(model, X_interior, X_bottom, p, domain, device):
    sx = torch.tensor([p["diff"], 0.0, 0.0, 0.0], device=device).float()
    sy = torch.tensor([0.0, p["diff"], 0.0, 0.0], device=device).float()
    sz = torch.tensor([0.0, 0.0, p["diff"], 0.0], device=device).float()

    with torch.no_grad():
        model.eval()

        tgt = uexact_time(X_interior.detach().cpu().numpy(), p).to(device)
        out = model(X_interior)

        mse_uvw = torch.mean((out - tgt)**2).item()

        num = torch.sqrt(torch.mean((out - tgt)**2))
        den = torch.sqrt(torch.mean(tgt**2))
        relL2_uvw = (num / (den + 1e-12)).item()

        outx = model(X_interior + sx)
        outy = model(X_interior + sy)
        outz = model(X_interior + sz)

        du = (outx[:, 0] - out[:, 0]) / p["diff"]
        dv = (outy[:, 1] - out[:, 1]) / p["diff"]
        dw = (outz[:, 2] - out[:, 2]) / p["diff"]
        rms_div = torch.sqrt(torch.mean((du + dv + dw)**2)).item()

        w = model(X_bottom)[:, 2]
        rms_bdy = torch.sqrt(torch.mean(w**2)).item()

        model.train()

    return mse_uvw, relL2_uvw, rms_div, rms_bdy

# ------------------------
#  Pretty print run configuration
# ------------------------
def print_run_config(args, device):
    print("\n============== Run configuration ==============")
    print(f"Device: {device}")
    print("\nSpatial domain:")
    print(f"  x ∈ [{args.a}, {args.b}]")
    print(f"  y ∈ [{args.c}, {args.ddd}]")
    print(f"  z ∈ [{args.e}, {args.f}]")
    print("\nTime domain:")
    print(f"  t ∈ [{args.t0}, {args.t1}]")

    print("\nTrajectory:")
    print(f"  traj = '{args.traj}'")
    print(f"  (x0, y0) = ({args.x0}, {args.y0})")
    print(f"  (vx, vy) = ({args.vx}, {args.vy})")
    if args.traj == "circular":
        print(f"  R     = {args.R}")
        print(f"  omega = {args.omega}")
        print(f"  phi   = {args.phi}")
    if args.traj == "parabolic":
        print(f"  ax = {args.ax}")
        print(f"  ay = {args.ay}")

    print("\nNetwork:")
    print(f"  input dim = 4 (x,y,z,t)")
    print(f"  output dim = 3 (u,v,w)")
    print(f"  width = {args.width}, depth = {args.depth}")

    print("\nTraining:")
    print(f"  nEpoch       = {args.nEpoch}")
    print(f"  nBatch       = {args.nBatch}")
    print(f"  innerBatch   = {args.innerBatchSize}")
    print(f"  bdyBatch     = {args.bdyBatchSize}")

    print("\nOptimization:")
    print(f"  lr           = {args.lr}")
    print(f"  weight_decay = {args.decay}")
    print(f"  StepLR: step_size = {args.step_size}, gamma = {args.gamma}")

    print("\nPhysics / penalties:")
    print(f"  diff     = {args.diff}")
    print(f"  penalty  = {args.penalty}")
    print(f"  penalty2 = {args.penalty2}")
    print(f"  epsilon  = {args.epsilon}")

    print("\nEvaluation:")
    print(f"  numQuad (spatial grid per axis) = {args.numQuad}")

    print("\nOutput:")
    print(f"  outdir = '{args.outdir}'")
    print(f"  tag    = '{args.tag}'")
    print("===============================================\n")

# ------------------------
#  Train loop (VAL metrics on fixed sets)
# ------------------------
def train(domain, model, device, p, optim, sched, Xval, Xval_bdy):
    model.train()

    sx = torch.tensor([p["diff"],0.0,0.0,0.0], device=device).float()
    sy = torch.tensor([0.0,p["diff"],0.0,0.0], device=device).float()
    sz = torch.tensor([0.0,0.0,p["diff"],0.0], device=device).float()

    for epoch in range(p["nEpoch"]):
        epoch_cost = 0.0

        for _ in range(p["nBatch"]):
            data1 = torch.from_numpy(domain.sampleFromDomain()).float().to(device)
            data2 = torch.from_numpy(domain.sampleFromBottom()).float().to(device)

            tgt = uexact_time(data1.cpu().numpy(), p).to(device)
            o1  = model(data1)

            loss1 = 0.5*torch.mean(
                (tgt[:,0] - o1[:,0])**2 +
                (tgt[:,1] - o1[:,1])**2
            ) * domain.volumen()

            o1x = model(data1+sx)
            o1y = model(data1+sy)
            o1z = model(data1+sz)

            du  = (o1x[:,0]-o1[:,0])/p["diff"]
            dv  = (o1y[:,1]-o1[:,1])/p["diff"]
            dw  = (o1z[:,2]-o1[:,2])/p["diff"]
            loss2 = torch.mean((du+dv+dw)**2) * p["penalty"] * domain.volumen()

            o2    = model(data2)
            loss3 = torch.mean(o2[:,2]**2) * p["penalty2"] * domain.areaBottom()

            loss = loss1 + loss2 + loss3

            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                epoch_cost += errorFun(o1[:,0:2], tgt[:,0:2])

            sched.step()

        if epoch % 30 == 0:
            mse_uvw, relL2_uvw, rms_div, rms_bdy = compute_metrics_fixed(
                model=model,
                X_interior=Xval,
                X_bottom=Xval_bdy,
                p=p,
                domain=domain,
                device=device
            )
            print(
                f"Epoch {epoch:5d} | "
                f"VAL mse_uvw={mse_uvw:.3e} | "
                f"VAL relL2_uvw={relL2_uvw:.3e} | "
                f"cost(u,v)≈{(epoch_cost/p['nBatch']):.3e} | "
                f"VAL rms_div={rms_div:.3e} | "
                f"VAL rms_bdy={rms_bdy:.3e}"
            )

# ------------------------
#  Main
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="DeepWind 3D vortex – time-dependent trajectory (linear/circular/parabolic)")

    # Spatial domain
    ap.add_argument('--a', type=float, default=-7.0)
    ap.add_argument('--b', type=float, default= 7.0)
    ap.add_argument('--c', type=float, default=-7.0)
    ap.add_argument('--ddd', type=float, default= 7.0)
    ap.add_argument('--e', type=float, default= 0.0)
    ap.add_argument('--f', type=float, default= 7.0)
    ap.add_argument('--numQuad', type=int, default=10, help='grid points per axis for uniform eval')

    # Time domain
    ap.add_argument('--t0', type=float, default=0.0, help='initial time')
    ap.add_argument('--t1', type=float, default=1.0, help='final time')

    ap.add_argument(
        '--eval_times',
        nargs='+',
        type=float,
        default=None,
        help="Evaluation times as a list of numbers. Example: --eval_times 0 0.25 0.5 1. "
             "If omitted, uses t0, (t0+t1)/2, t1."
    )


    # Trajectory selector
    ap.add_argument('--traj', type=str, default="linear", choices=["linear","circular","parabolic"],
                    help="center trajectory type")

    # Common center params
    ap.add_argument('--x0', type=float, default=0.0, help='base x-center')
    ap.add_argument('--y0', type=float, default=0.0, help='base y-center')
    ap.add_argument('--vx', type=float, default=1.0, help='linear velocity in x (used in linear/parabolic)')
    ap.add_argument('--vy', type=float, default=0.0, help='linear velocity in y (used in linear/parabolic)')

    # Circular params
    ap.add_argument('--R',     type=float, default=2.0, help='radius (circular)')
    ap.add_argument('--omega', type=float, default=2*np.pi, help='angular velocity (circular)')
    ap.add_argument('--phi',   type=float, default=0.0, help='phase shift in radians (circular)')

    # Parabolic params
    ap.add_argument('--ax', type=float, default=0.0, help='quadratic accel coef in x (parabolic)')
    ap.add_argument('--ay', type=float, default=0.0, help='quadratic accel coef in y (parabolic)')

    # Network/training
    ap.add_argument('--width',          type=int, default=80)
    ap.add_argument('--depth',          type=int, default=5)
    ap.add_argument('--nEpoch',         type=int, default=1000)
    ap.add_argument('--nBatch',         type=int, default=25)
    ap.add_argument('--innerBatchSize', type=int, default=512)
    ap.add_argument('--bdyBatchSize',   type=int, default=256)
    ap.add_argument('--lr',             type=float, default=1e-2)
    ap.add_argument('--decay',          type=float, default=1e-6)
    ap.add_argument('--step_size',      type=int, default=5000)
    ap.add_argument('--gamma',          type=float, default=0.7)

    # Physics/penalties
    ap.add_argument('--diff',           type=float, default=1e-3)
    ap.add_argument('--penalty',        type=float, default=1.0)
    ap.add_argument('--penalty2',       type=float, default=1.0)
    ap.add_argument('--epsilon',        type=float, default=0.1)

    # IO
    ap.add_argument('--outdir', type=str, default='out_time')
    ap.add_argument('--tag', type=str, default='run_time_traj')

    args = ap.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    p = vars(args).copy()
    p.update({"d":4, "dd":3})

    os.makedirs(args.outdir, exist_ok=True)

    print_run_config(args, device)

    model = RitzNet(p).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    sch   = StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    dom   = DomainCubeTime(p)

    # Fixed VAL / TEST sets
    N_val  = 5000
    N_test = 20000
    rng = np.random.default_rng(1234)

    def sample_fixed_interior(N):
        x = rng.random(N)*(args.b-args.a) + args.a
        y = rng.random(N)*(args.ddd-args.c) + args.c
        z = rng.random(N)*(args.f-args.e) + args.e
        t = rng.random(N)*(args.t1-args.t0) + args.t0
        return np.stack([x, y, z, t], axis=1)

    def sample_fixed_bottom(N):
        x = rng.random(N)*(args.b-args.a) + args.a
        y = rng.random(N)*(args.ddd-args.c) + args.c
        z = np.full(N, args.e)
        t = rng.random(N)*(args.t1-args.t0) + args.t0
        return np.stack([x, y, z, t], axis=1)

    Xval     = torch.from_numpy(sample_fixed_interior(N_val)).float().to(device)
    Xval_bdy = torch.from_numpy(sample_fixed_bottom(N_val)).float().to(device)

    Xtest     = torch.from_numpy(sample_fixed_interior(N_test)).float().to(device)
    Xtest_bdy = torch.from_numpy(sample_fixed_bottom(N_test)).float().to(device)

    tstart = time.time()
    train(dom, model, device, p, opt, sch, Xval, Xval_bdy)
    print(f"Training took {time.time()-tstart:.2f}s")

    # Evaluation at several times
    spatial_xyz = dom.uniformSpatialGrid()
    print(f"Spatial eval grid shape: {spatial_xyz.shape}")

    # Parse evaluation times (supports negative times if within [t0,t1])
    try:
        eval_times = parse_eval_times(args.eval_times, args.t0, args.t1)
    except ValueError as e:
        ap.error(str(e))

    for t_eval in eval_times:
        t_eval = float(t_eval)
        t_vec = np.full((spatial_xyz.shape[0], 1), t_eval, dtype=np.float64)
        xyz_t = np.concatenate([spatial_xyz, t_vec], axis=1)

        eval_xyz_t = torch.from_numpy(xyz_t).float().to(device)

        with torch.no_grad():
            out_approx = model(eval_xyz_t)
            out_exact  = uexact_time(xyz_t, p).to(device)

        tagt = time_tag(t_eval)
        file_exact  = os.path.join(args.outdir, f"vector_exact_t{tagt}.dat")
        file_approx = os.path.join(args.outdir, f"vector_approx_t{tagt}.dat")
        save_vectors_for_gnuplot(file_exact,  eval_xyz_t, out_exact)
        save_vectors_for_gnuplot(file_approx, eval_xyz_t, out_approx)
        print(f"[t={t_eval:.2f}] Saved exact:   {file_exact}")
        print(f"[t={t_eval:.2f}] Saved approx:  {file_approx}")

        # Save comparison (exact vs approx) + pointwise error magnitude
        xyz_np    = eval_xyz_t.detach().cpu().numpy()[:, :3]
        exact_np  = out_exact.detach().cpu().numpy()
        approx_np = out_approx.detach().cpu().numpy()
        err       = np.sqrt(np.sum((exact_np - approx_np)**2, axis=1))
        data_cmp  = np.column_stack([xyz_np, exact_np, approx_np, err])
        header    = "x y z u_exact v_exact w_exact u_app v_app w_app error"
        file_error = os.path.join(args.outdir, f"vector_cmp_error_t{tagt}.dat")
        np.savetxt(file_error, data_cmp, fmt="%.8f", header=header)
        mse = np.mean((exact_np - approx_np)**2)
        print(f"[t={t_eval:.2f}] MSE = {mse:.6e}, error file: {file_error}")

    with torch.no_grad():
        data_for_div = torch.from_numpy(dom.sampleFromDomain()).float().to(device)
        div_mean = compute_divergence(data_for_div, model, p, device)
        print(f"Mean divergence on random space-time sample ≈ {div_mean:.5e}")


    # Final TEST metrics at eval_times
    print("\n========== FINAL TEST METRICS ==========")

    mse_all   = []
    relL2_all = []
    Euv_all   = []
    Euvw_all  = []
    div_all   = []
    bdy_all   = []

    for t_eval in eval_times:
        t_eval = float(t_eval)
        t_vec = np.full((spatial_xyz.shape[0], 1), t_eval)
        xyz_t = np.concatenate([spatial_xyz, t_vec], axis=1)
        X = torch.from_numpy(xyz_t).float().to(device)

        with torch.no_grad():
            approx = model(X)
            exact  = uexact_time(xyz_t, p).to(device)

        mse = torch.mean((approx - exact)**2).item()

        num = torch.sqrt(torch.mean((approx - exact)**2))
        den = torch.sqrt(torch.mean(exact**2))
        relL2 = (num / (den + 1e-12)).item()

        Euv  = rel_error_uv(approx, exact)
        Euvw = rel_error_uvw(approx, exact)
        divr = div_rms_at_time(dom, model, t_eval, p, device)
        bdyr = bdy_rms_at_time(dom, model, t_eval, device)

        mse_all.append(mse)
        relL2_all.append(relL2)
        Euv_all.append(Euv)
        Euvw_all.append(Euvw)
        div_all.append(divr)
        bdy_all.append(bdyr)

        print(
            f"[TEST] t={t_eval:.2f} | "
            f"MSE_uvw={mse:.3e} | "
            f"relL2={relL2:.3e} | "
            f"Erel_uv={Euv:.3e} | "
            f"Erel_uvw={Euvw:.3e} | "
            f"div_rms={divr:.3e} | "
            f"bdy_rms={bdyr:.3e}"
        )

    print("----------------------------------------")
    print(
        f"[TEST-AVG] "
        f"MSE_uvw={np.mean(mse_all):.3e} | "
        f"relL2={np.mean(relL2_all):.3e} | "
        f"Erel_uv={np.mean(Euv_all):.3e} | "
        f"Erel_uvw={np.mean(Euvw_all):.3e} | "
        f"div_rms={np.mean(div_all):.3e} | "
        f"bdy_rms={np.mean(bdy_all):.3e}"
    )
    print("========================================\n")
        
    # # Final TEST metrics at eval_times
    # print("\n========== FINAL TEST METRICS ==========")

    # mse_all   = []
    # relL2_all = []
    # Euv_all   = []
    # Euvw_all  = []
    # div_all   = []
    # bdy_all   = []

    # for t_eval in eval_times:
    #     t_eval = float(t_eval)
    #     t_vec = np.full((spatial_xyz.shape[0], 1), t_eval)
    #     xyz_t = np.concatenate([spatial_xyz, t_vec], axis=1)
    #     X = torch.from_numpy(xyz_t).float().to(device)

    #     with torch.no_grad():
    #         approx = model(X)
    #         exact  = uexact_time(xyz_t, p).to(device)

    #     mse = torch.mean((approx - exact)**2).item()

    #     num = torch.sqrt(torch.mean((approx - exact)**2))
    #     den = torch.sqrt(torch.mean(exact**2))
    #     relL2 = (num / (den + 1e-12)).item()

    #     Euv  = rel_error_uv(approx, exact)
    #     Euvw = rel_error_uvw(approx, exact)
    #     divr = div_rms_at_time(dom, model, t_eval, p, device)
    #     bdyr = bdy_rms_at_time(dom, model, t_eval, device)

    #     mse_all.append(mse)
    #     relL2_all.append(relL2)
    #     Euv_all.append(Euv)
    #     Euvw_all.append(Euvw)
    #     div_all.append(divr)
    #     bdy_all.append(bdyr)

    #     print(
    #         f"[TEST] t={t_eval:.2f} | "
    #         f"MSE_uvw={mse:.3e} | "
    #         f"relL2={relL2:.3e} | "
    #         f"Erel_uv={Euv:.3e} | "
    #         f"Erel_uvw={Euvw:.3e} | "
    #         f"div_rms={divr:.3e} | "
    #         f"bdy_rms={bdyr:.3e}"
    #     )

    # print("----------------------------------------")
    # print(
    #     f"[TEST-AVG] "
    #     f"MSE_uvw={np.mean(mse_all):.3e} | "
    #     f"relL2={np.mean(relL2_all):.3e} | "
    #     f"Erel_uv={np.mean(Euv_all):.3e} | "
    #     f"Erel_uvw={np.mean(Euvw_all):.3e} | "
    #     f"div_rms={np.mean(div_all):.3e} | "
    #     f"bdy_rms={np.mean(bdy_all):.3e}"
    # )
    # print("========================================\n")
        
    # # Final TEST metrics at eval_times
    # print("\n========== FINAL TEST METRICS ==========")

    # Euv_all   = []
    # Euvw_all  = []
    # div_all   = []
    # bdy_all   = []

    # for t_eval in eval_times:
    #     t_eval = float(t_eval)
    #     t_vec = np.full((spatial_xyz.shape[0], 1), t_eval)
    #     xyz_t = np.concatenate([spatial_xyz, t_vec], axis=1)
    #     X = torch.from_numpy(xyz_t).float().to(device)

    #     with torch.no_grad():
    #         approx = model(X)
    #         exact  = uexact_time(xyz_t, p).to(device)

    #     Euv  = rel_error_uv(approx, exact)
    #     Euvw = rel_error_uvw(approx, exact)
    #     divr = div_rms_at_time(dom, model, t_eval, p, device)
    #     bdyr = bdy_rms_at_time(dom, model, t_eval, device)

    #     Euv_all.append(Euv)
    #     Euvw_all.append(Euvw)
    #     div_all.append(divr)
    #     bdy_all.append(bdyr)

    #     print(
    #         f"[TEST] t={t_eval:.2f} | "
    #         f"Erel_uv={Euv:.3e} "
    #         f"Erel_uvw={Euvw:.3e} | "
    #         f"div_rms={divr:.3e} "
    #         f"bdy_rms={bdyr:.3e}"
    #     )

    # print("----------------------------------------")
    # print(
    #     f"[TEST-AVG] "
    #     f"Erel_uv={np.mean(Euv_all):.3e} | "
    #     f"Erel_uvw={np.mean(Euvw_all):.3e} | "
    #     f"div_rms={np.mean(div_all):.3e} | "
    #     f"bdy_rms={np.mean(bdy_all):.3e}"
    # )
    # print("========================================\n")

    # Fixed random TEST metrics (space-time)
    mse_uvw_fix, relL2_uvw_fix, rms_div_fix, rms_bdy_fix = compute_metrics_fixed(
        model=model,
        X_interior=Xtest,
        X_bottom=Xtest_bdy,
        p=p,
        domain=dom,
        device=device
    )

    print("========== FIXED RANDOM TEST (space-time) ==========")
    print(
        f"[TEST-FIXED] "
        f"mse_uvw={mse_uvw_fix:.3e} | "
        f"relL2_uvw={relL2_uvw_fix:.3e} | "
        f"rms_div={rms_div_fix:.3e} | "
        f"rms_bdy={rms_bdy_fix:.3e}"
    )
    print("====================================================\n")

if __name__ == '__main__':
    main()
