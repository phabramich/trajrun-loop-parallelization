"""CPU (numpy/fp64) vs Metal (MLX/fp32) on the same parallel executor.

The GPU column only wins where a sweep is real work: wide states and
Newton's T x d x d scan matmuls. Scalar cases stay CPU-faster -- one
kernel launch per sweep costs more than the whole fp64 vector op.
"""

import time

import mlx.core as mx
import numpy as np

import parallel_run_mlx as m
from parallel_run import jacobi as cpu_jacobi
from parallel_run import newton as cpu_newton
from parallel_run import sequential

rng = np.random.default_rng(0)


def make_tanh(rho, d, seed=1):
    W = np.random.default_rng(seed).standard_normal((d, d))
    W *= rho / np.abs(np.linalg.eigvals(W)).max()
    Wm, Wmt = mx.array(W.astype(np.float32)), mx.array(W.T.astype(np.float32))
    f_np = lambda s, x: np.tanh(s @ W + x)
    f_mx = lambda s, x: mx.tanh(s @ Wm + x)
    j_np = lambda s, x: (1 - f_np(s, x)**2)[..., None] * W.T
    j_mx = lambda s, x: (1 - f_mx(s, x)**2)[..., None] * Wmt
    return (f_np, j_np), (f_mx, j_mx)


def make_linear(d, lam, seed=2):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    A = (lam * Q).astype(np.float64)
    Am = mx.array(A.astype(np.float32))
    f_np = lambda s, x: s @ A + x
    f_mx = lambda s, x: s @ Am + x
    # column-state Jacobian of s -> s @ A is A.T
    At, Atm = np.ascontiguousarray(A.T), mx.array(A.T.astype(np.float32))
    j_np = lambda s, x: np.broadcast_to(At, (len(s), d, d))
    j_mx = lambda s, x: mx.broadcast_to(Atm, (s.shape[0], d, d))
    return (f_np, j_np), (f_mx, j_mx)


def timed(fn, *a, warm=True, **kw):
    def once():
        out = fn(*a, **kw)
        if isinstance(out[0], mx.array):  # lazy: force GPU execution
            mx.eval(out[0])
        return out
    if warm:  # compile kernels outside the timed region
        once()
    t = time.perf_counter()
    out = once()
    return out, time.perf_counter() - t


rows = []

def bench(name, fpair, s0, xs, tol, do_newton=True):
    (f_np, j_np), (f_mx, j_mx) = fpair
    ref = sequential(f_np, s0, xs)[1:]
    for dev, run_j, run_n, f_, j_ in (
            ("cpu64", cpu_jacobi, cpu_newton, f_np, j_np),
            ("metal32", m.jacobi, m.newton, f_mx, j_mx)):
        def emit(row):
            rows.append(row)
            print(f"{name:<14} {dev:<8} {row[2]:<7} {row[3]:>6} "
                  f"{row[4]:>8.3f} {row[5]:>10.2e}  {row[6]}", flush=True)
        (s2, k2, ok2), tj = timed(run_j, f_, s0, xs, tol=tol)
        err = np.abs(np.asarray(s2) - ref).max() if ok2 else np.nan
        emit((dev, "", "jacobi", k2, tj, err, "" if ok2 else "NC"))
        if do_newton:
            (sn, kn, okn), tn = timed(run_n, f_, s0, xs, jac=j_, tol=tol)
            errn = np.abs(np.asarray(sn) - ref).max() if okn else np.nan
            emit((dev, "", "newton", kn, tn, errn, "" if okn else "NC"))


print(f"{'case':<14} {'dev':<8} {'method':<7} {'sweeps':>6} {'wall s':>8} "
      f"{'max|Δ|':>10}  note", flush=True)
TOL = 1e-4
T = 4096
bench("tanh d=8", make_tanh(0.8, 8), np.zeros(8),
      rng.standard_normal((T, 8)) * .3, TOL)

Tw = 2048
bench("tanh d=128", make_tanh(0.8, 128), np.zeros(128),
      (rng.standard_normal((Tw, 128)) * .3), TOL)
bench("linear d=96", make_linear(96, 0.9), np.zeros(96),
      rng.standard_normal((1024, 96)) * .5, TOL)
