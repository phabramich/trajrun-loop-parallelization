"""GPU fast path: the step written directly in MSL, `y[t] = step(s, x[t])`.

Same algorithm as trajpar (CPU), on Metal lanes:
  round 1   every lane runs its chunk from s0, writes outputs, records a
            state checkpoint every M steps;
  round r>1 lane c restarts from the previous round's end of lane c-1 and
            stops at the first checkpoint equal to its previous trajectory
            (or at once if its start did not change). Outputs before the
            merge point are spliced in by index — no extra output pass.
Data is laid out interleaved ([step][lane]) so neighbouring lanes read and
write neighbouring words. Bit-exact; rounds repeat until no boundary moves.

shortcut: one input/one output element per step, 32-bit state fields only;
          use wasm_metal for arbitrary programs.
"""
import mlx.core as mx

_SRC = """
uint c = thread_position_in_grid.x; uint C = threads_per_grid.x;
uint mode = params[0];
{decl}
if (mode == 0u) {{ {init} }} else {{ {load} }}
uint mq = L;  // first step NOT taken from this round (L = ran to the end)
if (mode != 0u && ({same})) mq = 0u;
for (uint j = 0; j < mq; ++j) {{
  {xt} x = xs[j * C + c];
  {yt} y;
  {step}
  ys[j * C + c] = y;
  if (((j + 1u) % M) == 0u) {{
    uint o = ((j + 1u) / M - 1u) * {nf}u * C + c;
    if (mode != 0u && {eq}) {{ mq = j + 1u; break; }}
    {rec}
  }}
}}
merge[c] = mq;
{store}
"""


class GpuStep:
    def __init__(self, name, state, step, x_dtype, y_dtype, M=64):
        """state: [(name, msl_type, init_value)], 32-bit fields;
        step: MSL statements reading `x`, updating state, assigning `y`."""
        self.state, self.M, self.ydt = state, M, y_dtype
        nf = len(state)
        bits = lambda v, t: f"as_type<uint>({v})" if t == "float" else f"(uint){v}"
        unb = lambda e, t: f"as_type<float>({e})" if t == "float" else f"({t}){e}"
        msl = {mx.float32: "float", mx.uint8: "uchar", mx.uint32: "uint",
               mx.int32: "int", mx.int16: "short"}
        src = _SRC.format(
            decl=" ".join(f"{t} {n};" for n, t, _ in state),
            init=" ".join(f"{n} = {v};" for n, t, v in state),
            load=" ".join(f"{n} = {unb(f's_in[{i}u * C + c]', t)};"
                          for i, (n, t, _) in enumerate(state)),
            same=" && ".join(f"s_in[{i}u * C + c] == s_prev[{i}u * C + c]"
                             for i in range(nf)),
            eq=" && ".join(f"ck_in[o + {i}u * C] == {bits(n, t)}"
                           for i, (n, t, _) in enumerate(state)),
            rec=" ".join(f"ck[o + {i}u * C] = {bits(n, t)};"
                         for i, (n, t, _) in enumerate(state)),
            store=" ".join(f"s_out[{i}u * C + c] = {bits(n, t)};"
                           for i, (n, t, _) in enumerate(state)),
            xt=msl[x_dtype], yt=msl[y_dtype], step=step, nf=nf)
        self.kern = mx.fast.metal_kernel(
            name=f"gpustep_{name}",
            input_names=["xs", "params", "s_in", "s_prev", "ck_in"],
            output_names=["ys", "s_out", "ck", "merge"], source=src)

    def _round(self, xs, C, L, mode, s_in, s_prev, ck_in):
        nf, M = len(self.state), self.M
        return self.kern(
            inputs=[xs, mx.array([mode], mx.uint32), s_in, s_prev, ck_in],
            template=[("L", L), ("M", M)], grid=(C, 1, 1),
            threadgroup=(min(256, C), 1, 1),
            output_shapes=[(L * C,), (nf * C,), ((L // M) * nf * C,), (C,)],
            output_dtypes=[self.ydt, mx.uint32, mx.uint32, mx.uint32],
            init_value=0)

    def run_interleaved(self, xs_il, C, max_rounds=64):
        """xs_il: [L*C] already interleaved (x[t] for lane c, step j at
        j*C + c, t = c*L + j). Returns (ys_il, rounds)."""
        L = xs_il.size // C
        assert L * C == xs_il.size and L % self.M == 0
        z = mx.zeros((1,), mx.uint32)
        ys, E, ck, _ = self._round(xs_il, C, L, 0, z, z, z)
        S = None
        j = mx.arange(L, dtype=mx.uint32)[:, None]
        q = mx.arange(L // self.M, dtype=mx.uint32)[:, None, None]
        for r in range(2, max_rounds + 1):
            nf = len(self.state)
            Er = E.reshape(nf, C)
            # lane c starts from the end of lane c-1; lane 0 keeps s0
            S_new = mx.concatenate(
                [self._s0_bits() if S is None else S.reshape(nf, C)[:, :1],
                 Er[:, :-1]], axis=1)
            S_prev = self._s0_all(C) if S is None else S
            y2, E2, ck2, mq = self._round(xs_il, C, L, 1, S_new.reshape(-1),
                                          S_prev.reshape(-1), ck)
            # merged/skipped lanes exited early: s_out is the state at the
            # exit point — the true end equals the old end (suffix identical)
            E2 = mx.where((mq == L)[None, :], E2.reshape(nf, C),
                          E.reshape(nf, C)).reshape(-1)
            moved = mx.any(E2 != E).item()
            ys = mx.where(j < mq[None, :], y2.reshape(L, C), ys.reshape(L, C)).reshape(-1)
            # merged at iter mq: lane wrote mq//M - 1 ckpts (the merge slot
            # itself is unchanged == old value); ran to end (mq==L): all NCK
            mqi = mq.astype(mx.int32)
            cut = mx.where(mqi == L, L // self.M,
                           mx.maximum(mqi // self.M - 1, 0))
            ck = mx.where(q < cut[None, None, :],
                          ck2.reshape(-1, nf, C), ck.reshape(-1, nf, C)).reshape(-1)
            E, S = E2, S_new.reshape(-1)
            if not moved:
                return ys, r
        raise RuntimeError("no convergence in max_rounds (wall?) — use serial")

    def _s0_bits(self):
        import struct
        b = [struct.unpack("<I", struct.pack("<f", float(v)))[0] if t == "float"
             else int(v) & 0xFFFFFFFF for _, t, v in self.state]
        return mx.array(b, mx.uint32)[:, None]

    def _s0_all(self, C):
        return mx.broadcast_to(self._s0_bits(), (len(self.state), C))

    def run(self, xs, C=4096):
        """Natural-order convenience wrapper (pays two transposes)."""
        L = -(-xs.size // C)
        L = -(-L // self.M) * self.M
        pad = L * C - xs.size
        xp = mx.concatenate([xs, mx.zeros((pad,), xs.dtype)]) if pad else xs
        ys, r = self.run_interleaved(xp.reshape(C, L).T.reshape(-1), C)
        return ys.reshape(L, C).T.reshape(-1)[:xs.size], r


FSM_LEX = dict(state=[("st", "uint", 0)], x_dtype=mx.uint8, y_dtype=mx.uint8, step="""
  bool q = x == 34, b = x == 92, nl = x == 10;
  st = nl ? 0u : (st == 0u ? (q ? 1u : 0u) : (st == 1u ? (b ? 2u : (q ? 0u : 1u)) : 1u));
  y = (uchar)st;""")

# FR freezes a product to rounded bits — blocks Metal's fma contraction so
# results match IEEE single-precision Rust/C exactly.
AUDIO_CHAIN = dict(state=[(n, "float", 0.0) for n in ("y1", "x1", "y2", "x2", "env")],
                   x_dtype=mx.float32, y_dtype=mx.float32, step="""
  float t1 = as_type<float>(as_type<uint>(0.9886f * y1));
  float a = x - x1; a = a + t1; x1 = x; y1 = a;
  float t2 = as_type<float>(as_type<uint>(0.9957f * y2));
  float b = a - x2; b = b + t2; x2 = a; y2 = b;
  float m = b < 0.0f ? -b : b;
  float d = as_type<float>(as_type<uint>(env * 0.995f));
  env = m > env ? m : d;
  float g = 0.5f / env; g = env > 0.5f ? g : 1.0f;
  y = b * g;""")


if __name__ == "__main__":
    import numpy as np

    n = 1 << 16
    d = np.frombuffer(b'abcdefg,,""\\\n', np.uint8)[
        np.random.RandomState(0).randint(0, 13, n)]
    st = np.zeros(n, np.uint8)
    for i, c in enumerate(d):  # scalar oracle: 0 plain / 1 quoted / 2 escaped
        st[i] = 0 if c == 10 else (1 if c == 34 else 0) if st[i - 1] == 0 else \
            (2 if c == 92 else (0 if c == 34 else 1)) if st[i - 1] == 1 else 1
    got, r = GpuStep("fsm", **FSM_LEX).run(mx.array(d), C=64)
    assert np.array_equal(np.asarray(got), st), "fsm mismatch"
    print(f"gpu_step self-check: {n} steps, {r} rounds, bit-exact")
