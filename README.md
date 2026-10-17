# Trajectory-parallel execution of sequential programs

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
wasm_load.py     WASM decoder -> flat pc-indexed program; `call` handled by
                 bytecode-level inlining (recursion/call_indirect rejected)
vm_ref.py        scalar reference interpreter — the correctness oracle
wasm_metal.py    WASM->MSL JIT kernel, Metal VM, Jacobi runner, probe3
                 (induction-var detection), merge predictor, auto EnumScan
run_parallel.py  thin CLI/API over the whole pipeline
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
