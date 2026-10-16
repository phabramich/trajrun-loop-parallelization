"""Batched wasm-VM на MLX + chunked-Jacobi исполнитель (parareal-стиль).

Состояние на позицию: (pc, locals[L], stack[D]) — u32-тензоры.
Один шаг = для каждого вида инструкций одно маскированное обновление
(divergence через select, не через ветки — FMAG-подход).
Память — shared read-only u8 (input + data-сегменты).
store — no-op для состояния, логируется отдельным проходом.

Исполнитель: неизвестные — состояния на границах каждые K шагов;
свип = K batched-шагов rollout'а всех границ + обновление Якоби.
Fixpoint (E[c] == S[c+1] для всех c) — сертификат точности.
"""

import numpy as np
try:
    import mlx.core as mx
except ImportError:
    from np_shim import np_shim as mx
from vm_ref import Program

U32 = np.uint32(0xFFFFFFFF)

# виды инструкций -> id
KINDS = ["skel", "lget", "lset", "ltee", "gget", "const", "un",
         "bin", "sel", "drop", "load", "store",
         "if", "else", "br", "brif", "halt"]

_BINOPS = ["i32.add", "i32.sub", "i32.mul", "i32.and", "i32.or",
           "i32.xor", "i32.shl", "i32.shr_u", "i32.shr_s", "i32.eq",
           "i32.ne", "i32.lt_s", "i32.lt_u", "i32.gt_s", "i32.gt_u",
           "i32.le_s", "i32.le_u", "i32.ge_s", "i32.ge_u", "i32.div_s",
           "i32.div_u", "i32.rem_s", "i32.rem_u", "i32.rotl", "i32.rotr",
           "i32.clz", "i32.ctz", "i32.popcnt"]
_UNOPS = ["i32.eqz"]


def _kind_of(nm):
    if nm in ("local.get",): return "lget"
    if nm in ("local.set",): return "lset"
    if nm in ("local.tee",): return "ltee"
    if nm in ("global.get",): return "gget"
    if nm in ("i32.const", "i64.const", "f32.const", "f64.const"): return "const"
    if nm in _UNOPS: return "un"
    if nm in _BINOPS: return "bin"
    if nm == "select": return "sel"
    if nm == "drop": return "drop"
    if nm.startswith("i32.load") or nm.startswith("i64.load"): return "load"
    if nm.startswith("i32.store") or nm.startswith("i64.store"): return "store"
    if nm == "if": return "if"
    if nm == "else": return "else"
    if nm == "br": return "br"
    if nm == "br_if": return "brif"
    if nm in ("halt", "return", "unreachable"): return "halt"
    return "skel"


class BatchVM:
    def __init__(self, prog: Program, mem: bytes, backend=None):
        if backend is not None:
            globals()["mx"] = backend
        self.mx = globals()["mx"]
        self.prog = prog
        n = prog.n_instr
        self.D = prog.maxdepth
        self.L = prog.nloc + prog.nparams
        self.mem = mx.array(np.frombuffer(mem, dtype=np.uint8))
        self.mem_size = len(mem)
        # таблицы программы, индексированные pc
        self.kind = np.array([KINDS.index(_kind_of(x)) for x in prog.names],
                             dtype=np.int32)
        self.op = np.zeros(n, np.int32)       # индекс бинарной/унарной операции
        for i, nm in enumerate(prog.names):
            if nm in _BINOPS:
                self.op[i] = _BINOPS.index(nm)
            elif nm in _UNOPS:
                self.op[i] = _UNOPS.index(nm)
        imm = np.zeros(n, np.int64)
        for i, (nm, v) in enumerate(zip(prog.names, prog.imms)):
            if isinstance(v, int):
                imm[i] = v
        self.imm = mx.array(imm.astype(np.uint32))
        self.dep = mx.array(np.array(prog.depth, np.int32))
        self.n_instr = n
        self.KIND = {k: i for i, k in enumerate(KINDS)}
        self.mkind = mx.array(self.kind)
        self.mop = mx.array(self.op)
        # load subkind: bytes count + signed flag per pc
        ldb = np.zeros(n, np.int32); lds = np.zeros(n, np.int32)
        stb = np.zeros(n, np.int32)
        for i, nm in enumerate(prog.names):
            if "load8" in nm: ldb[i] = 1
            elif "load16" in nm: ldb[i] = 2
            elif "load32" in nm or nm in ("i32.load", "f32.load"): ldb[i] = 4
            if "_s" in nm: lds[i] = 1
            if "store8" in nm: stb[i] = 1
            elif "store16" in nm: stb[i] = 2
            elif nm in ("i32.store", "f32.store", "i64.store"): stb[i] = 4
        self.ldb = mx.array(ldb); self.lds = mx.array(lds)
        self.stb = mx.array(stb)
        self.globals = mx.array(np.array(prog.globals, np.uint32))
        self.arange = None

    def init_state(self, B, args):
        """S0: pc=0, locals=args+0, stack=0."""
        loc = np.zeros((B, self.L), np.uint32)
        loc[:, :len(args)] = np.asarray(args, np.uint32)
        return (mx.zeros((B,), mx.int32), mx.array(loc),
                mx.zeros((B, self.D), mx.uint32))

    def _step(self, pc, loc, stk):
        B = pc.shape[0]
        if self.arange is None or self.arange.shape[0] != B:
            self.arange = mx.arange(B)
            self.oh_d = mx.arange(self.D)[None, :]
            self.oh_l = mx.arange(self.L)[None, :]
        ar = self.arange
        pc = mx.clip(pc, 0, self.n_instr - 1)  # защита от выхода за код
        kind = self.mkind[pc]
        dep = self.dep[pc]
        imm = self.imm[pc].astype(mx.int32)
        mK = lambda k: kind == k

        def rd(slot):
            return stk[ar, mx.clip(slot, 0, self.D - 1)]

        def wr_stk(slot, val, m):
            nonlocal stk
            oh = (self.oh_d == mx.clip(slot, 0, self.D - 1)[:, None])
            stk = mx.where(oh & m[:, None], val[:, None].astype(mx.uint32),
                           stk)

        def wr_loc(idx, val, m):
            nonlocal loc
            oh = (self.oh_l == mx.clip(idx, 0, self.L - 1)[:, None])
            loc = mx.where(oh & m[:, None], val[:, None].astype(mx.uint32),
                           loc)

        newpc = pc + 1

        # --- local/global/const ---
        m = mK(self.KIND["lget"])
        wr_stk(dep, loc[ar, mx.clip(imm, 0, self.L - 1)], m)
        m = mK(self.KIND["lset"])
        wr_loc(imm, rd(dep - 1), m)
        m = mK(self.KIND["ltee"])
        wr_loc(imm, rd(dep - 1), m)
        m = mK(self.KIND["gget"])
        wr_stk(dep, self.globals[mx.clip(imm, 0, len(self.prog.globals) - 1)], m)
        m = mK(self.KIND["const"])
        wr_stk(dep, imm.astype(mx.uint32), m)

        # --- unary ---
        m = mK(self.KIND["un"])
        v = rd(dep - 1)
        wr_stk(dep - 1, mx.where(v == 0, mx.ones_like(v), mx.zeros_like(v)), m)

        # --- binary: подвид из op[pc] ---
        m = mK(self.KIND["bin"])
        op = self.mop[pc]
        a, b = rd(dep - 2), rd(dep - 1)
        ai, bi = a.astype(mx.int32), b.astype(mx.int32)
        r = mx.zeros_like(a)
        for oid, fn in _bin_fns(ai, bi, a, b):
            r = mx.where(op == oid, fn, r)
        wr_stk(dep - 2, r, m)

        # --- select ---
        m = mK(self.KIND["sel"])
        c = rd(dep - 1)
        wr_stk(dep - 3, mx.where(c != 0, rd(dep - 3), rd(dep - 2)), m)

        # --- loads ---
        m = mK(self.KIND["load"])
        a = (rd(dep - 1).astype(mx.int64) + imm.astype(mx.int64)) \
            .astype(mx.int32)
        a = mx.clip(a, 0, self.mem_size - 1)
        nb, sgn = self.ldb[pc], self.lds[pc]
        v = self.mem[a].astype(mx.uint32)
        v = mx.where(nb > 1, v | (self.mem[mx.clip(a + 1, 0, self.mem_size - 1)]
                                  .astype(mx.uint32) << 8), v)
        v = mx.where(nb > 2,
                     v | (self.mem[mx.clip(a + 2, 0, self.mem_size - 1)]
                          .astype(mx.uint32) << 16)
                       | (self.mem[mx.clip(a + 3, 0, self.mem_size - 1)]
                          .astype(mx.uint32) << 24), v)
        v = mx.where((sgn == 1) & (nb == 1) & (v >= 0x80),
                     v | mx.full_like(v, 0xFFFFFF00), v)
        v = mx.where((sgn == 1) & (nb == 2) & (v >= 0x8000),
                     v | mx.full_like(v, 0xFFFF0000), v)
        wr_stk(dep - 1, v, m)

        # --- stores: логируем addr/val (-1 = не store) ---
        m = mK(self.KIND["store"])
        st_addr = (rd(dep - 2).astype(mx.int64) + imm.astype(mx.int64)) \
            .astype(mx.int32)
        st_val = rd(dep - 1)
        st_addr = mx.where(m, st_addr, -mx.ones_like(st_addr))

        # --- control ---
        m = mK(self.KIND["if"])
        newpc = mx.where(m, mx.where(rd(dep - 1) != 0, pc + 1, imm), newpc)
        m = mK(self.KIND["else"]) | mK(self.KIND["br"])
        newpc = mx.where(m, imm, newpc)
        m = mK(self.KIND["brif"])
        newpc = mx.where(m, mx.where(rd(dep - 1) != 0, imm, pc + 1), newpc)
        m = mK(self.KIND["halt"])
        newpc = mx.where(m, pc, newpc)  # заморозка на месте

        return mx.clip(newpc, 0, self.n_instr - 1), loc, stk, st_addr, st_val

    def step(self, S):
        pc, loc, stk = S
        npc, loc, stk, sa, sv = self._step(pc, loc, stk)
        return (npc, loc, stk), (sa, sv)

    def rollout(self, S, K, collect_stores=False, eval_every=32):
        pc, loc, stk = S
        log_a, log_v = [], []
        for j in range(K):
            npc, loc, stk, sa, sv = self._step(pc, loc, stk)
            if collect_stores:
                log_a.append(sa); log_v.append(sv)
            pc = npc
            if (j + 1) % eval_every == 0:
                mx.eval(pc, loc, stk)
        return (pc, loc, stk), (log_a, log_v)

def _bin_fns(ai, bi, a, b):
    """список (op_id, mlx-выражение) в порядке _BINOPS."""
    z = mx.zeros_like(a)
    c = lambda x: x.astype(mx.uint32)
    s = mx.clip(b, 0, 31)
    s0 = (32 - s) & 31
    f1 = mx.float32  # shortcut: div через f32 теряет точность при |x|>2^24;
    #   в наших программах div нет, точный путь — int64 на CPU
    bis = mx.where(bi == 0, mx.ones_like(bi), bi)  # защита от /0 в nan
    quot_s = mx.where((bi == 0) | ((ai == -2**31) & (bi == -1)), z,
                      (ai.astype(f1) / bis.astype(f1)).astype(mx.int32)
                      .astype(mx.uint32))
    rem_s_v = mx.where((bi == 0) | ((ai == -2**31) & (bi == -1)), ai,
                       (ai - (ai.astype(f1) / bis.astype(f1)).astype(mx.int32)
                        * bi).astype(mx.uint32))
    pc_ = a - ((a >> 1) & 0x55555555)
    pc_ = (pc_ & 0x33333333) + ((pc_ >> 2) & 0x33333333)
    pc_ = ((pc_ + (pc_ >> 4)) & 0x0F0F0F0F) * 0x01010101 >> 24

    def bitscan(x, is_clz):
        n = mx.zeros_like(x); y = x
        for w in (16, 8, 4, 2, 1):
            if is_clz:
                hi = y >> w
                n = mx.where(hi != 0, n + w, n)
                y = mx.where(hi != 0, hi, y)
            else:
                lo = y & ((1 << w) - 1)
                n = mx.where(lo != 0, n, n + w)
                y = mx.where(lo != 0, y >> w, y)
        r = 32 - n if is_clz else n
        return mx.where(x == 0, mx.full_like(x, 32), r)

    fns = [a + b, a - b, a * b, a & b, a | b, a ^ b,
           (a << s), (a >> s),
           (ai >> s.astype(mx.int32)).astype(mx.uint32),
           c(a == b), c(a != b),
           c(ai < bi), c(a < b), c(ai > bi), c(a > b),
           c(ai <= bi), c(a <= bi), c(ai >= bi), c(a >= b),
           quot_s, mx.where(b == 0, z, a // mx.maximum(b, 1)),
           rem_s_v, mx.where(b == 0, a, a % mx.maximum(b, 1)),
           (a << s) | (a >> s0), (a >> s) | (a << s0),
           bitscan(a, True), bitscan(a, False), pc_]
    return [(i, f) for i, f in enumerate(fns)]
