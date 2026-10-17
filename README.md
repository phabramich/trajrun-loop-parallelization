# Trajectory-parallel execution of sequential programs

**Write 15 lines of C describing one step `s[t+1] = f(s[t], x[t])` —
get a single binary that runs it in parallel on multicore CPU or Apple
Metal, bit-exact vs serial, or honestly reports a wall.**

## The product: `trajrun/`

```bash
cd trajrun && cargo build --release
./target/release/trajrun examples/csv_bounds.c big.csv bounds.bin --cpu 8
./target/release/trajrun --check examples/fm_rx.c iq.bin   # speculative or wall?
```

You write a scalar step function over a small state struct; `trajrun`
compiles it to a native dylib (CPU) or embeds it in generated Metal
kernels (GPU), then runs the trajectory speculatively: chunks execute in
parallel from guessed boundary states, and wherever a wrong guess
**merges** with the true trajectory the chunk is free. What never merges
(hash chains, non-contracting codecs) is reported as `wall`, not faked.

Measured on M1 (200–256MB inputs): CSV record bounds **3.7×**, UTF-8
validation **4.1×**, FM-receiver IQ chain **5.8×**, EWMA risk filter
**4.9×** on 8 CPU threads — all bit-exact. GPU wins on repeated calls or
GPU-resident data. See `trajrun/README.md`.

Prior art: this generalizes the FSM-speculation line (Mytkowicz'14,
GSpecPal'22, ParPaRaw/cuDF'20 — all enumerable-state) to **opaque state**
where merges can only be detected empirically. Map: `PRIOR_ART.md`.

---

## The research pipeline behind it

**Compiles a sequential Rust → WASM loop into a native Apple Metal kernel
and executes its trajectory `s[t+1] = f(s[t], x[t])` in parallel — bit-exact,
or it honestly reports a wall.**

The GPU cannot parallelize a sequential loop "as is" — each step needs the
previous state. This runtime asks a different question: *when is the
carried-state trajectory predictable enough that wrong guesses heal?*

- **Jacobi boundary iteration** — chunk the trajectory, roll out all chunks
  in parallel from guessed boundary states, re-inject true predecessors,
  repeat until a self-consistent fixpoint. Converges when dynamics
  **merge, contract or reset** (stable IIRs, saturating accumulators,
  FSMs with synchronizing sequences, escaped-quote lexers).
- **Automatic finite-state EnumScan** — when merges stall but observed
  boundary tuples fit a small domain, compose chunk transitions as a
  semidirect product (tuple + linear deltas) and resolve them by an
  associative scan. No manual hints.
- **Honest wall** — hash chains, VDF-like recurrences, non-contracting
  codecs (e.g. standard IMA ADPCM) are reported as `wall`, not faked.

Chunk boundaries always land on the loop header: `K` counts loop
*iterations* via back-edge detection, so variable-length bodies
(`br_table`, divergent `if/else`) don't break phase.

## Results (Apple Silicon, MLX/Metal, `uv run python wasm_metal.py`)

| workload | WASM steps | method | sweeps | vs 1-thread Metal | exact |
|---|---|---|---|---|---|
| `fsm_lex` CSV lexer, 256KB | 15.2M | Jacobi | 2 | ~10–24× | bit-exact |
| `fsm_lex` + escaped quotes (sync. word) | 15.2M | Jacobi | 5 | ~10–12× | bit-exact |
| `fsm_lex` quote-only Z₂ | 15.2M | **auto-EnumScan** | 1 rollout + scan | ~parity | bit-exact |
| `sat_acc` saturating i16 acc | 7.5M | Jacobi | 2 | ~16–19× | bit-exact |
| `adpcm_leaky` (λ=15/16) | 31.2M | Jacobi | 2 | ~20× | bit-exact |
| `adpcm_decode` IMA (λ=1) | 31.2M | — | — | **wall** | — |
| `audio_chain` HP→HP→peak-env→gain | 62.9M | Jacobi | 23 | **14.7×** | bit-exact |
| `float_leak` f32 IIR, `nibble_fsm` br_table, `call_test` inlined call | — | Jacobi | 2–4 | — | bit-exact |

`audio_chain` is a real DSP pipeline (2× highpass IIR + peak-hold envelope
+ gain) over 8M samples — the `ffmpeg -af highpass,acompressor` analog.

## Backends: honest baselines (M1, medians)

Three runtimes, same algorithm, different frontends — pick by where your
data lives and how much codegen you want:

| workload | Rust serial | Rust 8T (`trajpar`) | WASM→Metal | `gpu_step` |
|---|---|---|---|---|
| `fsm_lex` CSV 32MB | 70–100 ms | 11.7 ms | 21 ms | **7.4 ms** |
| `audio_chain` 16M f32 | 41–44 ms | **8.4 ms** | 25 ms | 14.2 ms |
| IMA ADPCM encoded audio | 83 ms | 83 ms (wall) | wall | — |

- **`trajrun/`** — single binary, zero deps at runtime:
  `trajrun step.c in.bin out.bin [--cpu N|--gpu]`. Your step is plain C:
  `typedef struct { float y1, x1, env; } S; float step(S *s, float x){…}`.
  CPU: compiled to a dylib (`cc`, batch loop → native speed) and run
  multicore; GPU: the same source is wrapped in the speculative Metal
  kernel — the Metal JIT ships with macOS, no toolchain needed.
  `trajrun --check step.c in.bin` predicts speculative vs wall upfront.
- **`trajpar/`** — native Rust crate, `std::thread` multicore, speculative
  chunks + merge detection + finite-domain scan, exact serial fallback.
  *The default choice*: no codegen, no GPU, 4–6× over serial on
  contracting workloads, `serial(wall)` on the rest.
- **`wasm_metal.py`** — the universal path above. Interpreting arbitrary
  WASM semantics per lane costs ~6× over a hand kernel; lands at CPU-8T
  parity, not ahead of it.
- **`gpu_step.py`** — the GPU niche that actually wins: write the per-step
  body in MSL (~10 lines), keep all speculative/fixup machinery. ~13× vs
  serial on 32MB CSV — beats 8 CPU threads. Narrow: one element in, one
  out, 32-bit state fields only.

GPU wins only when the state merges fast relative to input size and the
per-step body is trivial; CPU-8T wins on slower-merging f32 state; walls
are walls everywhere. For one-shot `trajrun` calls the ~40–60 ms Metal
source-JIT plus file I/O dominate — the GPU win is real for
repeated/GPU-resident workloads, less so for single files.

## Quickstart

```bash
uv sync                              # mlx + numpy, python >=3.13
uv run python wasm_metal.py          # full benchmark + bit-exact checks
```

Run your own function — convention `(in_ptr, n_bytes, out_ptr, extra u32...) -> usize`,
one loop:

```rust
#[no_mangle]
pub extern "C" fn my_alg(input: *const u8, n: usize, out: *mut u8) -> usize { ... }
```

```bash
cd wasm-progs && rustc --edition 2021 -O --crate-type cdylib \
  --target wasm32-unknown-unknown -o target/release/wasm_progs.wasm src/lib.rs
cd .. && uv run python run_parallel.py wasm-progs/target/release/wasm_progs.wasm:my_alg in.bin out.bin
```

```python
from run_parallel import run_wasm
out, info = run_wasm("p.wasm", "my_alg", data, extra_args=(97,))
# info: converged, method (jacobi / auto_enum_scan / wall), hist,
#       predict.merge_frac_1chunk, enum_info
```

## Repo map

```
trajrun/         the product: step.c -> multicore CPU or Metal GPU, one
                 binary, examples/ (csv_bounds, utf8_check, fm_rx,
                 ewma_risk, audio_chain, fsm)
wasm_load.py     WASM decoder -> flat pc-indexed program; `call` handled by
                 bytecode-level inlining (recursion/call_indirect rejected)
vm_ref.py        scalar reference interpreter — the correctness oracle
wasm_metal.py    WASM->MSL JIT kernel, Metal VM, Jacobi runner, probe3
                 (induction-var detection), merge predictor, auto EnumScan
run_parallel.py  thin CLI/API over the whole pipeline
gpu_step.py      GPU fast path: hand-written MSL step + speculative
                 checkpoint rounds — the config that beats multicore CPU
trajpar/         native Rust multicore backend (std::thread, no deps)
wasm-progs/      Rust test programs (fsm, adpcm, audio chain, f32, calls)
research/        earlier lineage: numpy Jacobi/Newton prototypes, spec
                 decode, per-instruction Metal VM (run: python -m research.X)
PRIOR_ART.md     research map: ASC/DEER/Parareal/FSM speculation/Parsynt
results.txt      experiment log incl. rejected ideas (Anderson accel,
                 threadgroup cache — measured, worse/needless)
README.ru.md     full Russian write-up
```

## Coverage & honest limits

**WASM subset:** structured control flow (`block/loop/if/else/br/br_if/
br_table`), all i32 ops, loads/stores, f32 arithmetic/comparisons/
conversions incl. `i32.trunc_sat_f32_*`, `select`, `call` via inlining.

**Not supported:** `call_indirect` (needs a static table), `i64`/`f64`
arithmetic, recursion, imports/syscalls, multiple loops, compaction-style
output indices that never contract.

- Speedup is *not* guaranteed — P-completeness is real. Payoff ≈
  predictability of the dynamics; `predict.merge_frac_1chunk` and `method`
  tell you upfront which side a workload falls on.
- Timings vary ±30% (Metal JIT, thermals); numbers are single-run, not
  medians.
- `merge_frac` is a 1-chunk probe: merges happening at distance ≥2 chunks
  are underestimated (false negatives), never overestimated.
- Everything is verified bit-exact against `vm_ref` and, where applicable,
  the sequential path of the same Metal kernel.

## License

MIT — see `LICENSE`.
