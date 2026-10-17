"""MLX/Metal port of parallel_run — same API, f written in mlx ops.

Same math as the numpy version; the differences are honest ones:
- Metal has no fp64: everything is fp32, so `tol` floors around ~1e-5 and
  "bitwise equal to the serial run" becomes "equal within fp32 noise".
- No uint64: the discrete cases (hash chain, CA) stay on CPU.
- Each Jacobi sweep / scan pass is one or two GPU kernel launches --
  this is the real deal, not a simulation of parallelism.
"""

import mlx.core as mx
import numpy as np


def _mx(x):
    return mx.array(np.asarray(x))


def jacobi(f, s0, xs, tol=0.0, init=None, max_iter=None):
    s0, xs = _mx(s0), _mx(xs)
    T = xs.shape[0]
    prev = _mx(init) if init is not None else mx.broadcast_to(
        s0, (T,) + s0.shape)
    lim = max_iter or T
    for k in range(1, lim + 1):
        s = f(mx.concatenate([s0[None], prev[:-1]]), xs)
        if bool(mx.allclose(s, prev, atol=tol, rtol=0).item()):
            return s, k, True
        prev = s
    return prev, lim, lim >= T


def _scan_linear(A, b):
    """Kogge-Stone over the time axis. Each pass is batched matmuls."""
    T = A.shape[0]
    step = 1
    while step < T:
        upd_b = (A[step:] @ b[:-step, :, None])[..., 0]
        upd_A = A[step:] @ A[:-step]
        b = mx.concatenate([b[:step], b[step:] + upd_b])
        A = mx.concatenate([A[:step], upd_A])
        step *= 2
    return b


def newton(f, s0, xs, jac=None, h=1e-3, max_iter=64, tol=1e-5):
    """Newton + parallel scan, damped, with Jacobi warm-start ladder."""
    s0v = _mx(np.atleast_1d(np.asarray(s0, dtype=np.float32)))
    xs = _mx(xs)
    T, d = xs.shape[0], s0v.size
    spent = 0
    for w in (0, 8, 32):
        if w:
            prev, _, _ = jacobi(f, s0v, xs, max_iter=w)
            spent += w
        else:
            prev = mx.broadcast_to(s0v, (T, d))
        for k in range(1, max_iter + 1):
            s_in = mx.concatenate([s0v[None], prev[:-1]])
            r = prev - f(s_in, xs)
            if bool(mx.allclose(r, mx.zeros_like(r), atol=tol, rtol=0)
                    .item()):
                return prev, spent + k - 1, True
            if jac is not None:
                J = jac(s_in, xs)
            else:
                base = f(s_in, xs)
                Jl = []
                for i in range(d):
                    e = np.zeros((1, d), np.float32)
                    e[0, i] = 1.0
                    dp = s_in + h * _mx(e)
                    Jl.append(((f(dp, xs) - base) / h)[:, :, None])
                J = mx.concatenate(Jl, axis=2)
            delta = _scan_linear(J, -r)
            ok = bool(mx.isfinite(delta).all().item())
            if ok:  # damped step
                scale = 1.0
                rnorm = float(mx.abs(r).max().item())
                for _ in range(30):
                    cand = prev + scale * delta
                    s_in_c = mx.concatenate([s0v[None], cand[:-1]])
                    rc = cand - f(s_in_c, xs)
                    if bool(mx.isfinite(cand).all().item()) and \
                            float(mx.abs(rc).max().item()) < rnorm:
                        prev = cand
                        break
                    scale *= 0.5
                else:
                    ok = False
            if not ok:
                spent += k
                break
        else:
            spent += max_iter
    return prev, spent, False


def to_np(x):
    return np.asarray(x)
