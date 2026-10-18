# trajrun

One binary: `step.c` → parallel execution on multicore CPU or Apple Metal GPU.
Sequential trajectory `s[t+1] = f(s[t], x[t])`, executed speculatively —
**bit-exact, or it honestly says wall.**

```bash
cargo build --release
./target/release/trajrun examples/csv_bounds.c in.bin out.bin --cpu 8
./target/release/trajrun examples/fsm.c        in.bin out.bin --gpu
./target/release/trajrun --check examples/fm_rx.c in.bin
```

No toolchain needed for GPU (Metal JIT is built into macOS); the CPU path
compiles `step.c` to a dylib with the system `cc`.

## The contract

Write a scalar C step function over a small state struct:

```c
typedef struct { float mu, var; } S;            // 1..16 fields of f32/u32/i32/u8

float step(S *s, float x) {                     // Y step(S*, X)
    float dev = x - s->mu;
    s->mu  = s->mu + 0.02f * dev;
    s->var = s->var + 0.001f * (dev*dev - s->var);
    return dev / sqrtf(s->var);                 // y[t]
}
```

Supported scalar types: `float`, `uint32_t`, `int32_t`, `uint8_t`.
Input file is a flat array of `X` (u8 = bytes, others = little-endian words).
Output file gets one `Y` per input element.

## How it runs

- **CPU** (default, all cores): `step.c` is wrapped in a generated C++ shim
  exporting a *batched* `traj_run` — the step body is inlined into a compiled
  loop, so chunks run at native speed. Rust orchestrates speculative
  boundaries + serial merge fixup only.
- **GPU** (`--gpu`): the same source is embedded in generated Metal kernels
  (`S*` → `thread S*`, `sqrtf` → `sqrt`, FMA contraction disabled for
  bit-exactness) with checkpoint rounds, boundary reinjection, merge
  detection and output splicing.
- `--check`: compile + probe a merge — tells you speculative vs wall
  *before* you run.
- `--verify` (default on CPU): bit-exact comparison against serial.

Speculation works when chunk trajectories **merge** — a run started from a
wrong boundary state converges to the true state within the chunk
(synchronizing FSMs, contracting IIR/EWMA/PLL loops, resynchronizing codes).
A second, auto-detected path covers **Z/2^32-affine steps** (`f = A·s + b(x)`:
LCGs, integrators, counters) — they never merge, but chunk maps compose
exactly, so `trajrun` computes every true boundary state and runs two exact
parallel passes. Anything else → reported `wall`, not faked.

`--serve step.c --gpu` reads `(in.bin, out.bin)` job lines on stdin — the
dylib and Metal kernels are built once, so per-call JIT stops dominating.
That is where the GPU path is meant to be used.

## Python

`trajrun-py` (maturin/pyo3) exposes the same engine:

```python
import trajrun
out, status = trajrun.run(src, data, threads=8)   # -> (bytes, "speculative ...")
info = trajrun.check(src, data)                   # verdict without running
```

`pip install .` in `trajrun-py/`, see `../trajrun-py/README.md`.

## Measured (M1, wall-clock incl. file I/O)

| example | input | serial | CPU 8T | GPU | merge dist. |
|---|---|---|---|---|---|
| `fsm.c` lexer | 32MB | 76 ms | **24.7 ms** | 101 ms | ~1 step |
| `csv_bounds.c` RFC4180 | 200MB | 361 ms | **97 ms** | 484 ms | ~1 step |
| `utf8_check.c` | 100MB | 192 ms | **47 ms** | 217 ms | ~1 step |
| `fm_rx.c` IQ→FM+PLL | 256MB | 534 ms | **92 ms** | 381 ms | ~162 steps |
| `ewma_risk.c` RiskMetrics | 256MB | 645 ms | **131 ms** | 706 ms | ~23K steps |
| `audio_chain.c` IIR+env | 16M f32 | 172 ms | **37 ms** | 127 ms | slow |
| `ndjson.c` record bounds | 1GiB | 1570 ms | **574 ms** | 5072 ms | ~212 steps |
| `lcg.c` LCG PRNG | 64MB u8 | was `wall` | **227 ms** (affine) | 918 ms (2 rounds) | never merges |

GPU wall is dominated by ~50ms Metal source JIT + file I/O; it wins only on
repeated calls or GPU-resident data. CPU-8T is the default for a reason.

## Prior art (cite honestly)

Speculative chunk + merge + fixup is a mature technique on *enumerable*
states: Mytkowicz ASPLOS'14 (enumerate all starts, merge on convergence),
Zhao & Shen ASPLOS'14/FSE'15, GSpecPal IPDPS'22 (GPU framework),
ParPaRaw/cuDF-FST VLDB'20 (shipped transition-function scan), rapidgzip.
`trajrun` generalizes it to **opaque, non-enumerable state** (floats,
mixed structs) where merge can only be detected empirically — with
bit-exact commit and an honest wall verdict. See `../PRIOR_ART.md`.

## Limits (deliberate C subset)

- one `step` function, no imports/recursion/dynamic alloc
- state = flat struct of 32-bit scalars (i64/f64 rejected)
- monotonic accumulators in state break merges — emit them as outputs
  and reduce after (see `utf8_check.c` for the pattern)
- parser is a scanner, not a C front-end — exotic signatures may confuse it
