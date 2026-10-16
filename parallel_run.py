"""Universal parallel executor for recurrences  s_{t+1} = f(s_t, x_t).

Any computation is a trajectory of states. Instead of stepping through it,
treat the whole trajectory as unknown and solve all equations
s_{t+1} = f(s_t, x_t) simultaneously:

- jacobi : iterate the entire guessed trajectory in parallel until fixpoint.
           Position t is guaranteed exact after t sweeps, so worst case = T
           sweeps = plain sequential execution. Never wrong: a stable
           trajectory IS a certificate.
- newton : Newton (DEER-style) on the trajectory residual. Each sweep solves
           the linearized recurrence with a parallel associative scan
           (Kogge-Stone, depth O(log T)). Quadratic convergence when the
           dynamics contracts; diverges on chaos -- same law.

All f are batched: f(s_batch, x_batch) -> next_state_batch.
"""

import numpy as np


def sequential(f, s0, xs):
    """Serial reference. Returns (T+1, ...) states including s0."""
    s0 = np.asarray(s0)
    T = len(xs)
    s = np.empty((T + 1,) + s0.shape, dtype=s0.dtype)
    s[0] = s0
    for t in range(T):
        s[t + 1] = f(s[t : t + 1], xs[t : t + 1])[0]
    return s


def jacobi(f, s0, xs, tol=0.0, init=None, max_iter=None):
    """Parallel fixed-point sweeps.

    Each sweep recomputes s_{t+1}=f(s_t,x_t) for all t at once from the
    previous guess. Converged => the trajectory is a bitwise fixpoint =>
    identical to sequential execution (tol=0) or within tol.

    Returns (states[T] = s_1..s_T, sweeps, converged).
    """
    s0 = np.asarray(s0)
    T = len(xs)
    prev = np.empty((T,) + s0.shape, dtype=s0.dtype)
    prev[:] = s0 if init is None else init
    lim = max_iter or T
    for k in range(1, lim + 1):
        s = f(np.concatenate([s0[None], prev[:-1]]), xs)
        done = np.allclose(s, prev, atol=tol, rtol=0)
        prev = s
        if done:
            return s, k, True
    return prev, lim, lim >= T  # sweep T always gives the exact trajectory


def _scan_linear(A, b):
    """Inclusive scan for s_t = A_t @ s_{t-1} + b_t with s_{-1} = 0.

    Kogge-Stone doubling: log2(T) passes, each a batched matmul over time.
    A: (T,d,d), b: (T,d) -> cumulative states (T,d).
    """
    A, b = A.copy(), b.copy()
    step = 1
    T = len(A)
    while step < T:
        b[step:] += np.einsum("tij,tj->ti", A[step:], b[:-step])
        A[step:] = np.einsum("tij,tjk->tik", A[step:], A[:-step])
        step *= 2
    return b


def newton(f, s0, xs, jac=None, h=1e-6, max_iter=64, tol=1e-13):
    """Newton on the trajectory residual r_t = s_t - f(s_{t-1}, x_t).

    Each sweep: linearize -> solve delta_t = J_t delta_{t-1} - r_t by the
    parallel scan -> update. 1 sweep for linear f (the scan IS the parallel
    form), few sweeps for contracting nonlinear f, divergence on chaos.

    Newton is local: on divergence the guess is re-warmed with Jacobi
    sweeps (globally convergent on contracting dynamics -- the coarse
    propagator of this framework). Escalates warmups (0, 8, 32).

    jac(s_batch, x_batch) -> (T,d,d) Jacobians; finite-diff if None.
    Returns (states[T], total parallel sweeps, converged).
    """
    s0v = np.atleast_1d(np.asarray(s0, dtype=float))
    T, d = len(xs), s0v.size
    spent = 0
    for w in (0, 8, 32):
        if w:
            prev, _, _ = jacobi(f, s0v, xs, max_iter=w)
            spent += w
        else:
            prev = np.broadcast_to(s0v, (T, d)).copy()
        for k in range(1, max_iter + 1):
            s_in = np.concatenate([s0v[None], prev[:-1]])
            r = prev - f(s_in, xs)
            if np.allclose(r, 0.0, atol=tol, rtol=0):
                return prev, spent + k - 1, True
            if jac is not None:
                J = jac(s_in, xs)
            else:
                J = np.empty((T, d, d))
                for i in range(d):
                    dp = s_in.copy()
                    dp[:, i] += h
                    J[:, :, i] = (f(dp, xs) - f(s_in, xs)) / h
            delta = _scan_linear(J, -r)
            ok = np.isfinite(delta).all()
            if ok:  # damped step: halve until the residual shrinks
                scale, rnorm = 1.0, np.abs(r).max()
                for _ in range(30):
                    cand = prev + scale * delta
                    s_in_c = np.concatenate([s0v[None], cand[:-1]])
                    if np.isfinite(cand).all() and \
                            np.abs(cand - f(s_in_c, xs)).max() < rnorm:
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
