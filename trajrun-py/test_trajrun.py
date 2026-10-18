"""smoke test: python3 test_trajrun.py  (or pytest)"""
import pathlib

import trajrun

EXAMPLES = pathlib.Path(__file__).parent.parent / "trajrun" / "examples"

FSM = """
typedef struct { uint8_t q; } S;
uint8_t step(S *s, uint8_t x) {
    if (x == 'a') s->q = 1;
    else if (x == 'b') s->q = 0;
    return s->q;
}
"""

def test_speculative_bit_exact():
    x = bytes((i * 2654435761) & 0xFF for i in range(1 << 20))
    out, status = trajrun.run(FSM, x, threads=4)
    assert "speculative" in status
    # serial reference in python
    q = 0
    ref = bytearray(len(x))
    for i, b in enumerate(x):
        if b == ord('a'):
            q = 1
        elif b == ord('b'):
            q = 0
        ref[i] = q
    assert bytes(out) == bytes(ref)

def test_affine_check():
    src = (EXAMPLES / "lcg.c").read_text()
    x = b"\x00" * 4096
    info = trajrun.check(src, x)
    assert "affine" in info
    out, status = trajrun.run(src, x, threads=4)
    assert "affine" in status
    # serial LCG reference (y is u32)
    s = 0   # traj_init zero-initializes S
    for i, b in enumerate(x):
        s = (1664525 * s + 1013904223 + b) & 0xFFFFFFFF
        assert s == int.from_bytes(out[i * 4 : i * 4 + 4], "little"), i

if __name__ == "__main__":
    test_speculative_bit_exact()
    test_affine_check()
    print("ok")
