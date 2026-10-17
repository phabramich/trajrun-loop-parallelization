"""Test battery: does iteration count track predictability?

Every case: exact sequential reference, then Jacobi and Newton sweeps.
`sweeps` = parallel depth (Jacobi: sweeps * span(f); Newton adds a
log2(T) scan per sweep). Asserts parallel output == serial output --
exactness is by construction, the only question is cost.
"""

import time

import numpy as np

from research.parallel_run import sequential, jacobi, newton

np.seterr(over="ignore", invalid="ignore")  # chaotic Jacobians legitimately overflow
rng = np.random.default_rng(0)

# --- dynamics ---------------------------------------------------------------

def f_linear(a):
    return lambda s, x: a * s + x

def f_logistic(r):
    return lambda s, x: r * s * (1.0 - s)

def make_tanh(rho, d=8, seed=1):
    W = np.random.default_rng(seed).standard_normal((d, d))
    W *= rho / np.abs(np.linalg.eigvals(W)).max()
    def f(s, x):
        return np.tanh(s @ W + x)
    def jac(s_in, x):
        y = f(s_in, x)
        return (1.0 - y**2)[..., None] * W.T[None]
    return f, jac

def f_ca(s, x):  # rule 30 on a 64-bit cyclic row
    L = (s >> np.uint64(1)) | (s << np.uint64(63))
    R = (s << np.uint64(1)) | (s >> np.uint64(63))
    return L ^ (s | R)

C1, C2, C3 = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9),
              np.uint64(0x94D049BB133111EB))
def f_hash(s, x):  # splitmix64-style avalanche chain
    z = s + x + C1
    z = (z ^ (z >> np.uint64(30))) * C2
    z = (z ^ (z >> np.uint64(27))) * C3
    return z ^ (z >> np.uint64(31))

def f_fsm(s, x):  # symbol 0 is a reset -> states merge
    nxt = (s * np.uint64(31) + x) % np.uint64(1 << 20)
    return np.where(x == 0, np.uint64(0), nxt)

# --- harness ----------------------------------------------------------------

rows = []

def check(got, ref, tol, discrete):
    if discrete:
        assert np.array_equal(got, ref), "mismatch vs sequential"
        return 0
    d = float(np.abs(got - ref).max())
    assert d <= max(10 * tol, 1e-12), f"mismatch vs sequential: {d}"
    return d

def run(name, f, s0, xs, tol=0.0, do_newton=True, jac=None):
    T = len(xs)
    discrete = np.asarray(s0).dtype.kind in "iu"
    t0 = time.perf_counter()
    ref = sequential(f, s0, xs)[1:]
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    s, k, ok = jacobi(f, s0, xs, tol=tol)
    t_j = time.perf_counter() - t0
    rows.append((name, T, "jacobi", k, k, t_seq, t_j,
                 check(s, ref, tol, discrete) if ok else np.nan,
                 "k=T: no gain" if ok and k >= T else "" if ok else "NO CONV"))

    if do_newton and not discrete:
        t0 = time.perf_counter()
        s, k, ok = newton(f, s0, xs, jac=jac)
        t_n = time.perf_counter() - t0
        depth = k * (int(np.log2(T)) + 2)
        rows.append((name, T, "newton+scan", k, depth, t_seq, t_n,
                     check(s, ref, tol or 1e-12, discrete) if ok else np.nan,
                     "" if ok else "diverged"))

# --- battery ----------------------------------------------------------------

T = 4096
x = (rng.standard_normal(T) * 0.5)[:, None]
for a in (0.5, 0.9, 0.99):
    run(f"linear a={a}", f_linear(a), np.zeros(1), x, tol=1e-13)

f, j = make_tanh(0.8); xr = rng.standard_normal((T, 8)) * 0.3
run("tanh-rnn rho=0.8", f, np.zeros(8), xr, tol=1e-13, jac=j)
f, j = make_tanh(1.4)
run("tanh-rnn rho=1.4", f, np.zeros(8), xr, tol=1e-13, jac=j)

for r in (2.8, 3.5, 3.8, 4.0):
    run(f"logistic r={r}", f_logistic(r), np.full(1, 0.3),
        np.zeros((2048, 1)), tol=1e-14)

Tc = 256
run("rule30 CA", f_ca, np.uint64(0x243F6A8885A308D3),
    np.zeros(Tc, dtype=np.uint64), do_newton=False)
run("hash chain", f_hash, np.uint64(42),
    rng.integers(0, 2**64, size=Tc, dtype=np.uint64), do_newton=False)

Tf = 8192
for p in (0.5, 0.01, 0.001):
    xs = rng.choice(4, size=Tf, p=[p] + [(1 - p) / 3] * 3).astype(np.uint64)
    run(f"fsm reset p={p}", f_fsm, np.uint64(7), xs, do_newton=False)

# --- report -----------------------------------------------------------------

print(f"{'case':<18} {'T':>5} {'method':<11} {'sweeps':>6} {'depth~':>6} "
      f"{'T/depth':>7} {'seq s':>7} {'par s':>7} {'max|Δ|':>9}  note")
for name, T, m, k, depth, ts, tp, err, note in rows:
    print(f"{name:<18} {T:>5} {m:<11} {k:>6} {depth:>6} {T/depth:>7.1f} "
          f"{ts:>7.3f} {tp:>7.3f} {err:>9.2e}  {note}")

# --- incremental recompute ---------------------------------------------------
# One input symbol edited; parallel re-solve warm-started from the old
# trajectory. "Healing" = how fast wrong states merge back onto the truth.

print("\none-symbol edit, init = old trajectory:")
print(f"{'case':<18} {'T':>5} {'edit@':>6} {'sweeps':>7}  note")

def rerun(name, f, s0, xs, t_edit, tol, bump):
    T = len(xs)
    ref = sequential(f, s0, xs)
    xs2 = xs.copy()
    xs2[t_edit] = bump(xs2[t_edit])
    s, k, ok = jacobi(f, s0, xs2, tol=tol, init=ref[1:])
    ref2 = sequential(f, s0, xs2)[1:]
    if np.asarray(s0).dtype.kind in "iu":
        assert np.array_equal(s, ref2)
    else:
        assert np.abs(s - ref2).max() <= max(10 * tol, 1e-12)
    print(f"{name:<18} {T:>5} {t_edit:>6} {k:>7}  "
          f"{'heals' if k < T - t_edit else 'never heals'}")

rerun("linear a=0.9", f_linear(0.9), np.zeros(1), x.copy(), 3000, 1e-13,
      lambda v: v + 1.0)
xs_fsm = rng.choice(4, size=8192, p=[.01, .33, .33, .33]).astype(np.uint64)
rerun("fsm reset p=.01", f_fsm, np.uint64(7), xs_fsm, 4096, 0.0,
      lambda v: (v + 1) % 4)
xs_hash = rng.integers(0, 2**64, size=8192, dtype=np.uint64)
rerun("hash chain", f_hash, np.uint64(42), xs_hash, 4096, 0.0,
      lambda v: v + np.uint64(1))
