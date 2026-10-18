# trajrun (Python)

Python binding for [trajrun](../trajrun/README.md): speculative parallel
execution of a sequential trajectory described by a small C step function —
multicore CPU everywhere, Metal GPU on macOS, bit-exact.

## Install

```bash
pip install maturin
cd trajrun-py
maturin develop --release    # or: pip install .
```

## API

```python
import trajrun

src = open("step.c").read()          # same step.c contract as the CLI
data = bytes(...)                    # flat input: u8 or little-endian u32/f32 elements

out, status = trajrun.run(src, data)               # -> (bytes, str)
out, status = trajrun.run(src, data, threads=8)    # pin thread count
out, status = trajrun.run(src, data, gpu=True)     # Metal (macOS, falls back to CPU)

print(status)   # "speculative chunks=32 ..." | "affine ..." | "serial(wall)" | "gpu N rounds"
print(trajrun.check(src, data))       # verdict without running: merge distance / affine / wall
```

`status` tells you which method actually ran — a `wall` result is a serial run,
not a speedup; `affine` means the step was proven linear over Z/2^32 and was
executed as an exact scan (no speculation).

`run` raises `RuntimeError` on a malformed step.c or if the input length is not
a multiple of the element size.

## Example

```python
import trajrun

FSM = """
typedef struct { uint8_t q; } S;
uint8_t step(S *s, uint8_t x) {
    if (x == 'a') s->q = 1;
    else if (x == 'b') s->q = 0;
    return s->q;
}
"""

out, status = trajrun.run(FSM, open("input.bin", "rb").read())
```

## Test

```bash
python3 test_trajrun.py    # or: pytest test_trajrun.py
```

## Notes

- Compilation cost: `run` compiles the step once per call (~50-200 ms). For
  repeated calls on the same step, keep the CLI's `--serve` mode or cache at
  the call site — a persistent-process mode for Python is a future improvement.
- Output is a `bytes` copy; a `bytearray`/numpy zero-copy variant is possible
  but not implemented — input is read via the buffer protocol without a copy.
- GPU path requires macOS + Metal; `gpu=True` elsewhere silently uses CPU.
