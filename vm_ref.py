"""Скалярный reference-интерпретатор плоского wasm-кода + ADPCM-энкодер.

Состояние: (pc, locals[L], stack[D]). Память — bytearray, read-only для
нас (store логируем, в память не пишем). i32-семантика: все значения
храним как беззнаковые u32, знаковые интерпретируем по месту.
"""

import math, random, struct
import numpy as np
from wasm_load import Module, decode, NAME_PP

U32 = 0xFFFFFFFF


def sx(v):          # u32 -> signed
    return v - 0x100000000 if v & 0x80000000 else v


def b2f(v):         # u32 bits -> f32 (как np.float32 — одинарное округление)
    return np.float32(struct.unpack("<f", struct.pack("<I", v))[0])


def f2b(f):         # f32 -> u32 bits
    return struct.unpack("<I", struct.pack("<f", float(np.float32(f))))[0]


class Program:
    def __init__(self, mod, name):
        (self.names, self.imms, self.pops, self.pushs, self.depth,
         self.nloc, self.nparams, self.end_pc) = decode(mod, name)
        self.globals = mod.globals
        self.n_instr = len(self.names)
        self.maxdepth = max(d for d in self.depth if d >= 0) + 1

    def run(self, mem, args, max_steps=10_000_000, log_stores=True,
            keep_every=1):
        """mem: bytearray. args -> начальные locals[0..nparams).
        Возвращает (states_log, stores, nsteps): states_log[i] = полное
        состояние после i-го шага. keep_every>1: хранит только границы."""
        pc = 0
        loc = [a & U32 for a in args] + [0] * self.nloc
        st = [0] * self.maxdepth
        stores = []
        states = [(0, tuple(loc), tuple(st))]  # states[i] = после iK шагов
        executed = 0
        for step in range(max_steps):
            if pc >= self.n_instr or self.names[pc] == "halt":
                break
            nm = self.names[pc]; imm = self.imms[pc]; d = self.depth[pc]
            npc = pc + 1
            if nm == "block" or nm == "loop" or nm == "end" or nm == "nop":
                pass
            elif nm == "if":
                npc = pc + 1 if st[d - 1] != 0 else imm
            elif nm == "else" or nm == "br":
                npc = imm
            elif nm == "br_if":
                npc = imm if st[d - 1] != 0 else pc + 1
            elif nm == "br_table":
                tgts, dflt = imm
                idx = st[d - 1]
                npc = tgts[idx] if idx < len(tgts) else dflt
            elif nm == "return":
                break
            elif nm == "local.get":
                st[d] = loc[imm]
            elif nm == "local.set":
                loc[imm] = st[d - 1]
            elif nm == "local.tee":
                loc[imm] = st[d - 1]
            elif nm == "global.get":
                st[d] = self.globals[imm]
            elif nm == "drop":
                pass
            elif nm == "select":
                st[d - 3] = st[d - 3] if st[d - 1] != 0 else st[d - 2]
            elif nm == "i32.const":
                st[d] = imm & U32
            elif nm == "f32.const":
                st[d] = f2b(imm)
            elif nm == "i64.const":
                st[d] = imm & 0xFFFFFFFFFFFFFFFF
            elif nm.startswith("i32.load8_u"):
                a = st[d - 1] + imm
                st[d - 1] = mem[a]
            elif nm.startswith("i32.load8_s"):
                a = st[d - 1] + imm
                v = mem[a]
                st[d - 1] = v | 0xFFFFFF00 if v & 0x80 else v
            elif nm.startswith("i32.load16_u"):
                a = st[d - 1] + imm
                st[d - 1] = mem[a] | (mem[a + 1] << 8)
            elif nm.startswith("i32.load16_s"):
                a = st[d - 1] + imm
                v = mem[a] | (mem[a + 1] << 8)
                st[d - 1] = v | 0xFFFF0000 if v & 0x8000 else v
            elif nm in ("i32.load", "f32.load"):
                a = st[d - 1] + imm
                st[d - 1] = (mem[a] | (mem[a + 1] << 8) |
                             (mem[a + 2] << 16) | (mem[a + 3] << 24))
            elif nm.startswith("i32.store") or nm in ("i64.store",
                                                      "f32.store"):
                a = st[d - 2] + imm
                if log_stores:
                    stores.append((a, st[d - 1], nm))
            elif nm == "i32.eqz":
                st[d - 1] = 1 if st[d - 1] == 0 else 0
            else:
                b = st[d - 1] if self.pops[pc] == 2 else None
                a = st[d - 2] if self.pops[pc] == 2 else st[d - 1]
                dst = d - 2 if self.pops[pc] == 2 else d - 1
                if nm == "i32.add": v = a + b
                elif nm == "i32.sub": v = a - b
                elif nm == "i32.mul": v = a * b
                elif nm == "i32.and": v = a & b
                elif nm == "i32.or": v = a | b
                elif nm == "i32.xor": v = a ^ b
                elif nm == "i32.shl": v = a << (b & 31)
                elif nm == "i32.shr_u": v = a >> (b & 31)
                elif nm == "i32.shr_s": v = sx(a) >> (b & 31)
                elif nm == "i32.eq": v = int(a == b)
                elif nm == "i32.ne": v = int(a != b)
                elif nm == "i32.lt_s": v = int(sx(a) < sx(b))
                elif nm == "i32.lt_u": v = int(a < b)
                elif nm == "i32.gt_s": v = int(sx(a) > sx(b))
                elif nm == "i32.gt_u": v = int(a > b)
                elif nm == "i32.le_s": v = int(sx(a) <= sx(b))
                elif nm == "i32.le_u": v = int(a <= b)
                elif nm == "i32.ge_s": v = int(sx(a) >= sx(b))
                elif nm == "i32.ge_u": v = int(a >= b)
                elif nm == "i32.div_s":
                    assert b != 0
                    v = int(sx(a) / sx(b))  # trunc to zero
                elif nm == "i32.div_u":
                    assert b != 0
                    v = a // b
                elif nm == "i32.rem_s":
                    assert b != 0
                    v = sx(a) - int(sx(a) / sx(b)) * sx(b)
                elif nm == "i32.rem_u":
                    assert b != 0
                    v = a % b
                elif nm == "i32.rotl":
                    s = b & 31; v = ((a << s) | (a >> (32 - s))) & U32
                elif nm == "i32.rotr":
                    s = b & 31; v = ((a >> s) | (a << (32 - s))) & U32
                elif nm == "i32.clz":
                    v = 32 - a.bit_length()
                elif nm == "i32.ctz":
                    v = 32 if a == 0 else (a & -a).bit_length() - 1
                elif nm == "i32.popcnt":
                    v = bin(a).count("1")
                elif nm in ("f32.add", "f32.sub", "f32.mul", "f32.div",
                            "f32.min", "f32.max", "f32.copysign"):
                    fa, fb = b2f(a), b2f(b)
                    if nm == "f32.add": r = fa + fb
                    elif nm == "f32.sub": r = fa - fb
                    elif nm == "f32.mul": r = fa * fb
                    elif nm == "f32.div": r = fa / fb
                    elif nm == "f32.min": r = np.minimum(fa, fb)
                    elif nm == "f32.max": r = np.maximum(fa, fb)
                    else: r = np.copysign(fa, fb)
                    v = f2b(r)
                elif nm in ("f32.eq", "f32.ne", "f32.lt", "f32.gt",
                            "f32.le", "f32.ge"):
                    fa, fb = b2f(a), b2f(b)
                    if nm == "f32.eq": v = int(fa == fb)
                    elif nm == "f32.ne": v = int(fa != fb)
                    elif nm == "f32.lt": v = int(fa < fb)
                    elif nm == "f32.gt": v = int(fa > fb)
                    elif nm == "f32.le": v = int(fa <= fb)
                    else: v = int(fa >= fb)
                elif nm in ("f32.abs", "f32.neg", "f32.sqrt", "f32.ceil",
                            "f32.floor", "f32.trunc", "f32.nearest"):
                    fa = b2f(a)
                    if nm == "f32.abs": r = np.abs(fa)
                    elif nm == "f32.neg": r = np.float32(-fa)
                    elif nm == "f32.sqrt": r = np.sqrt(fa)
                    elif nm == "f32.ceil": r = np.ceil(fa)
                    elif nm == "f32.floor": r = np.floor(fa)
                    elif nm == "f32.trunc": r = np.trunc(fa)
                    else: r = np.rint(fa)
                    v = f2b(r)
                elif nm == "i32.trunc_f32_s":
                    v = int(np.trunc(b2f(a))) & U32
                elif nm == "i32.trunc_f32_u" or nm == "i32.trunc_sat_f32_u":
                    t = float(np.trunc(b2f(a)))
                    v = int(min(max(t, 0.0), float(U32))) if not math.isnan(t) else 0
                elif nm == "i32.trunc_sat_f32_s":
                    t = float(np.trunc(b2f(a)))
                    v = int(min(max(t, -2147483648.0), 2147483647.0)) \
                        if not math.isnan(t) else 0
                elif nm == "f32.convert_i32_s":
                    v = f2b(np.float32(sx(a)))
                elif nm == "f32.convert_i32_u":
                    v = f2b(np.float32(a))
                elif nm in ("f32.reinterpret_i32", "i32.reinterpret_f32"):
                    v = a
                else:
                    raise NotImplementedError(nm)
                st[dst] = v & U32
            pc = npc
            executed += 1
            if executed % keep_every == 0:
                states.append((pc, tuple(loc), tuple(st)))
        if executed % keep_every != 0 or len(states) == 1:
            states.append((pc, tuple(loc), tuple(st)))
        return states, stores, executed


def make_memory(mod, regions):
    """regions: [(addr, bytes)]. Память = 1 стр. (64K) минимум + сегменты."""
    size = max(65536 * max(1, mod.mem_min),
               max(a + len(d) for a, d in regions + mod.data))
    mem = bytearray(size)
    for off, data in mod.data:
        mem[off:off + len(data)] = data
    for a, d in regions:
        mem[a:a + len(d)] = d
    return mem


STEP_TABLE = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28,
              31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107,
              118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
              337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876,
              963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
              2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894,
              6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
              15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
              32767]
IDX_TABLE = [-1, -1, -1, -1, 2, 4, 6, 8]


def adpcm_encode(samples):
    """IMA ADPCM encoder: i16 samples -> nibbles packed 2/byte."""
    pred, index = 0, 0
    nibs = []
    for s in samples:
        s = max(-32768, min(32767, int(s)))
        step = STEP_TABLE[index]
        diff = s - pred
        nib = 8 if diff < 0 else 0
        adiff = abs(diff)
        if adiff >= step: nib |= 4; adiff -= step
        if adiff >= step >> 1: nib |= 2; adiff -= step >> 1
        if adiff >= step >> 2: nib |= 1
        d = step >> 3
        if nib & 4: d += step
        if nib & 2: d += step >> 1
        if nib & 1: d += step >> 2
        pred = max(-32768, min(32767, pred - d if nib & 8 else pred + d))
        index = max(0, min(88, index + IDX_TABLE[nib & 7]))
        nibs.append(nib)
    out = bytearray(len(nibs) // 2)
    for i in range(len(out)):
        out[i] = nibs[2 * i] | (nibs[2 * i + 1] << 4)
    return bytes(out)


def adpcm_decode_py(data):
    """Эталон декодера на чистом Python (сверка wasm-интерпретатора)."""
    pred, index = 0, 0
    out = []
    for byte in data:
        for nib in (byte & 15, byte >> 4):
            step = STEP_TABLE[index]
            d = step >> 3
            if nib & 4: d += step
            if nib & 2: d += step >> 1
            if nib & 1: d += step >> 2
            pred = max(-32768, min(32767, pred - d if nib & 8 else pred + d))
            index = max(0, min(88, index + IDX_TABLE[nib & 7]))
            out.append(pred)
    return out


def adpcm_leaky_encode(samples):
    """Leaky ADPCM encoder (G.726/G.722 style, leak 15/16 on pred and index)."""
    pred, index = 0, 0
    nibs = []
    for s in samples:
        s = max(-32768, min(32767, int(s)))
        step = STEP_TABLE[index]
        diff = s - ((pred * 15) >> 4)
        nib = 8 if diff < 0 else 0
        adiff = abs(diff)
        if adiff >= step: nib |= 4; adiff -= step
        if adiff >= step >> 1: nib |= 2; adiff -= step >> 1
        if adiff >= step >> 2: nib |= 1
        index = max(0, min(88, ((index * 15) >> 4) + IDX_TABLE[nib & 7] + 2))
        step = STEP_TABLE[index]
        d = step >> 3
        if nib & 4: d += step
        if nib & 2: d += step >> 1
        if nib & 1: d += step >> 2
        pred = max(-32768, min(32767, ((pred * 15) >> 4) + (-d if nib & 8 else d)))
        nibs.append(nib)
    out = bytearray(len(nibs) // 2)
    for i in range(len(out)):
        out[i] = nibs[2 * i] | (nibs[2 * i + 1] << 4)
    return bytes(out)


def gen_input(n_samples=8192, seed=1):
    random.seed(seed)
    return [int(12000 * math.sin(i * 0.05) + random.randint(-1500, 1500))
            for i in range(n_samples)]
