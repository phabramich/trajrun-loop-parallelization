"""wasm-парсер: бинарь -> плоский список инструкций с разрешёнными ветвями.

Плоский формат: instrs[pc] = (name, imm); br/br_if несут АБСОЛЮТНЫЙ pc.
if -> else_target; else -> after_end; block/loop/end = скелет (nop).
Конец функции и return = "halt". br_table поддержан (список целей).
Глубина стека на каждом pc статична -> операнды = фиксированные слоты.
Подмножество: то, что реально эмитит rustc -O2 на наших программах.
"""

import struct

# opcode -> (name, imm_kind, pop, push)
OPS = {}
def _op(code, name, imm=None, pop=0, push=0):
    OPS[code] = (name, imm, pop, push)

for args in [
    (0x00, "unreachable"), (0x01, "nop"),
    (0x02, "block", "bt"), (0x03, "loop", "bt"),
    (0x04, "if", "bt", 1), (0x05, "else"),
    (0x0B, "end"), (0x0C, "br", "u32"), (0x0D, "br_if", "u32", 1),
    (0x0E, "br_table", "table", 1), (0x0F, "return"),
    (0x10, "call", "u32"), (0x11, "call_indirect", "u32"),
    (0x1A, "drop", None, 1), (0x1B, "select", None, 3, 1),
    (0x20, "local.get", "u32", 0, 1), (0x21, "local.set", "u32", 1),
    (0x22, "local.tee", "u32", 1, 1),
    (0x23, "global.get", "u32", 0, 1), (0x24, "global.set", "u32", 1),
    (0x41, "i32.const", "i32", 0, 1), (0x42, "i64.const", "i64", 0, 1),
    (0x43, "f32.const", "f32", 0, 1), (0x44, "f64.const", "f64", 0, 1),
    (0x3F, "memory.size", "u32", 0, 1), (0x40, "memory.grow", "u32", 1, 1),
]:
    _op(*args)

for i, nm in enumerate(["i32.load", "i64.load", "f32.load", "f64.load",
                        "i32.load8_s", "i32.load8_u", "i32.load16_s",
                        "i32.load16_u", "i64.load8_s", "i64.load8_u",
                        "i64.load16_s", "i64.load16_u", "i64.load32_s",
                        "i64.load32_u"]):
    _op(0x28 + i, nm, "mem", 1, 1)
for i, nm in enumerate(["i32.store", "i64.store", "f32.store", "f64.store",
                        "i32.store8", "i32.store16", "i64.store8",
                        "i64.store16", "i64.store32"]):
    _op(0x36 + i, nm, "mem", 2)

for i, nm in enumerate(["i32.eqz", "i32.eq", "i32.ne", "i32.lt_s",
                        "i32.lt_u", "i32.gt_s", "i32.gt_u", "i32.le_s",
                        "i32.le_u", "i32.ge_s", "i32.ge_u"]):
    _op(0x45 + i, nm, None, 1 if nm == "i32.eqz" else 2, 1)
for i, nm in enumerate(["i64.eqz", "i64.eq", "i64.ne", "i64.lt_s",
                        "i64.lt_u", "i64.gt_s", "i64.gt_u", "i64.le_s",
                        "i64.le_u", "i64.ge_s", "i64.ge_u"]):
    _op(0x50 + i, nm, None, 1 if nm == "i64.eqz" else 2, 1)
for i, nm in enumerate(["f32.eq", "f32.ne", "f32.lt", "f32.gt", "f32.le",
                        "f32.ge", "f64.eq", "f64.ne", "f64.lt", "f64.gt",
                        "f64.le", "f64.ge"]):
    _op(0x5B + i, nm, None, 2, 1)
for i, nm in enumerate(["i32.clz", "i32.ctz", "i32.popcnt", "i32.add",
                        "i32.sub", "i32.mul", "i32.div_s", "i32.div_u",
                        "i32.rem_s", "i32.rem_u", "i32.and", "i32.or",
                        "i32.xor", "i32.shl", "i32.shr_s", "i32.shr_u",
                        "i32.rotl", "i32.rotr"]):
    _op(0x67 + i, nm, None, 1 if i < 3 else 2, 1)
for i, nm in enumerate(["i64.clz", "i64.ctz", "i64.popcnt", "i64.add",
                        "i64.sub", "i64.mul", "i64.div_s", "i64.div_u",
                        "i64.rem_s", "i64.rem_u", "i64.and", "i64.or",
                        "i64.xor", "i64.shl", "i64.shr_s", "i64.shr_u",
                        "i64.rotl", "i64.rotr"]):
    _op(0x79 + i, nm, None, 1 if i < 3 else 2, 1)
for i, nm in enumerate(["f32.abs", "f32.neg", "f32.ceil", "f32.floor",
                        "f32.trunc", "f32.nearest", "f32.sqrt",
                        "f32.add", "f32.sub", "f32.mul", "f32.div",
                        "f32.min", "f32.max", "f32.copysign"]):
    _op(0x8B + i, nm, None, 1 if i < 7 else 2, 1)
for i, nm in enumerate(["f64.abs", "f64.neg", "f64.ceil", "f64.floor",
                        "f64.trunc", "f64.nearest", "f64.sqrt",
                        "f64.add", "f64.sub", "f64.mul", "f64.div",
                        "f64.min", "f64.max", "f64.copysign"]):
    _op(0x99 + i, nm, None, 1 if i < 7 else 2, 1)
# конверсии, которые любит rustc
for i, nm in enumerate(["i32.wrap_i64", "i32.trunc_f32_s", "i32.trunc_f32_u",
                        "i32.trunc_f64_s", "i32.trunc_f64_u", "i64.extend_i32_s",
                        "i64.extend_i32_u", "i64.trunc_f32_s",
                        "i64.trunc_f32_u", "i64.trunc_f64_s",
                        "i64.trunc_f64_u", "f32.convert_i32_s",
                        "f32.convert_i32_u", "f32.convert_i64_s",
                        "f32.convert_i64_u", "f32.demote_f64",
                        "f64.convert_i32_s", "f64.convert_i32_u",
                        "f64.convert_i64_s", "f64.convert_i64_u",
                        "f64.promote_f32", "i32.reinterpret_f32",
                        "i64.reinterpret_f64", "f32.reinterpret_i32",
                        "f64.reinterpret_i64"]):
    _op(0xA7 + i, nm, None, 1, 1)

# 0xFC-префикс: saturating trunc (rustc эмитит на `x as u8`-кастах)
FC_OPS = {0: ("i32.trunc_sat_f32_s", None, 1, 1),
          1: ("i32.trunc_sat_f32_u", None, 1, 1),
          2: ("i32.trunc_sat_f64_s", None, 1, 1),
          3: ("i32.trunc_sat_f64_u", None, 1, 1)}

# синтетическая инструкция — не имеет wasm-опкода
NAME_PP = {v[0]: (v[2], v[3])
           for v in list(OPS.values()) + list(FC_OPS.values())}
NAME_PP["halt"] = (0, 0)
NAME_PP["else"] = (0, 0)


def _uleb(b, p):
    r = s = 0
    while True:
        x = b[p]; p += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, p
        s += 7


def _sleb(b, p, bits=64):
    r = s = 0
    while True:
        x = b[p]; p += 1
        r |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            if s < bits and x & 0x40:
                r |= -1 << s
            return r, p


def _uleb_enc(v):
    out = bytearray()
    while True:
        x = v & 0x7F; v >>= 7
        out.append(x | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _func_locals(mod, fidx):
    """(nparams, n_declared_locals, offset_инструкций) для функции fidx."""
    nparams = len(mod.params[mod.funcs[fidx]])
    body = mod.code[fidx]
    nl, p = _uleb(body, 0)
    nloc = 0
    for _ in range(nl):
        cnt, p = _uleb(body, p)
        p += 1                      # type byte
        nloc += cnt
    return nparams, nloc, p


def _rewrite(mod, fidx, base, visiting, as_callee):
    """Тело fidx без locals-decls и без финального end, с инлайненными
    call'ами. Локалы смещены на +base. Возвращает (bytes, n_slots):
    n_slots = сколько слотов локалов занято начиная с base.
    as_callee=True: `return` -> `br k` на конец обёртки-инлайна."""
    if fidx in visiting:
        raise NotImplementedError("recursive call — unsupported")
    nparams, nloc, p = _func_locals(mod, fidx)
    body = mod.code[fidx]
    out = bytearray()
    depth = 0                        # вложенность блоков внутри тела
    next_free = base + nparams + nloc
    while p < len(body):
        op = body[p]; p += 1
        if op == 0xFC:               # префикс trunc_sat: копируем sub-uleb
            out.append(op)
            v, p = _uleb(body, p)
            out += _uleb_enc(v)
            continue
        if op == 0x10:               # call -> block + prologue + callee + end
            gidx, p = _uleb(body, p)
            np2 = len(mod.params[mod.funcs[gidx]])
            base2 = next_free
            inner, used = _rewrite(mod, gidx, base2,
                                   visiting | {fidx}, as_callee=True)
            next_free = base2 + used
            out += b"\x02\x40"       # block (void)
            for i in range(np2 - 1, -1, -1):
                out += b"\x21" + _uleb_enc(base2 + i)   # local.set arg
            out += inner
            out += b"\x0b"           # end
            continue
        if op == 0x11:
            raise NotImplementedError("call_indirect — unsupported")
        if op == 0x0F:               # return
            if as_callee:
                out += b"\x0c" + _uleb_enc(depth)   # br depth -> обёртка
            else:
                out.append(op)
            continue
        if op in (0x02, 0x03, 0x04):  # block/loop/if: depth++ , bt копируем
            out.append(op)
            out.append(body[p]); p += 1
            depth += 1
            continue
        if op == 0x0B:               # end
            if depth == 0:           # финальный end функции — стрипаем
                break
            depth -= 1
            out.append(op)
            continue
        if op in (0x20, 0x21, 0x22):  # local.get/set/tee: remap + base
            idx, p = _uleb(body, p)
            out.append(op)
            out += _uleb_enc(idx + base)
            continue
        # остальные: копируем op + immediates по типу
        nm, ik, _, _ = OPS[op]
        out.append(op)
        if ik == "u32" or ik in ("i32", "i64"):
            p0 = p
            _, p = _uleb(body, p)
            out += body[p0:p]
        elif ik == "f32":
            out += body[p:p + 4]; p += 4
        elif ik == "f64":
            out += body[p:p + 8]; p += 8
        elif ik == "bt":
            out.append(body[p]); p += 1
        elif ik == "mem":
            for _ in range(2):
                p0 = p
                _, p = _uleb(body, p)
                out += body[p0:p]
        elif ik == "table":
            nt, p2 = _uleb(body, p)
            out += _uleb_enc(nt)
            p = p2
            for _ in range(nt + 1):
                p0 = p
                _, p = _uleb(body, p)
                out += body[p0:p]
    return bytes(out), next_free - base


def inline_calls(mod, fidx):
    """Полное тело функции (decls + instrs + end) со всеми call
    проинлайненными рекурсивно."""
    instrs, used = _rewrite(mod, fidx, 0, frozenset(), as_callee=False)
    nparams, _, _ = _func_locals(mod, fidx)
    decls = _uleb_enc(1) + _uleb_enc(used - nparams) + b"\x7f"  # все i32
    return decls + instrs + b"\x0b"


class Module:
    def __init__(self, path):
        b = open(path, "rb").read()
        assert b[:4] == b"\0asm"
        self.funcs, self.exports, self.code = [], {}, []
        self.data, self.globals, self.mem_min = [], [], 0
        p = 8
        while p < len(b):
            sec = b[p]; size, p = _uleb(b, p + 1)
            body = b[p:p + size]; p += size
            getattr(self, f"_sec{sec}", lambda b: None)(body)

    def _sec1(self, b):
        n, p = _uleb(b, 0)
        self.params = []
        for _ in range(n):
            assert b[p] == 0x60; p += 1
            na, p = _uleb(b, p)
            args = list(b[p:p + na]); p += na
            nr, p = _uleb(b, p); p += nr
            self.params.append(args)

    def _sec3(self, b):
        n, p = _uleb(b, 0)
        self.funcs = list(b[p:p + n])

    def _sec5(self, b):
        n, p = _uleb(b, 0)
        for _ in range(n):
            fl = b[p]; p += 1
            self.mem_min, p = _uleb(b, p)
            if fl & 1:
                _, p = _uleb(b, p)

    def _sec6(self, b):
        n, p = _uleb(b, 0)
        for _ in range(n):
            p += 2
            assert b[p] == 0x41
            v, p = _sleb(b, p + 1, 32)
            assert b[p] == 0x0B; p += 1
            self.globals.append(v)

    def _sec7(self, b):
        n, p = _uleb(b, 0)
        for _ in range(n):
            ln, p = _uleb(b, p)
            name = b[p:p + ln].decode(); p += ln
            kind = b[p]; p += 1
            idx, p = _uleb(b, p)
            if kind == 0:
                self.exports[name] = idx

    def _sec10(self, b):
        n, p = _uleb(b, 0)
        for _ in range(n):
            sz, p = _uleb(b, p)
            self.code.append(b[p:p + sz]); p += sz

    def _sec11(self, b):
        n, p = _uleb(b, 0)
        for _ in range(n):
            fl, p = _uleb(b, p)
            assert fl == 0
            assert b[p] == 0x41
            off, p = _sleb(b, p + 1, 32)
            assert b[p] == 0x0B; p += 1
            ln, p = _uleb(b, p)
            self.data.append((off, b[p:p + ln])); p += ln


def decode(mod, name):
    """-> (names, imms, pops, pushs, depth, nloc, nparams)

    imms: br/br_if = абсолютный pc; br_table = список абс. pc;
    if = else-target; else = after-end; всё прочее = исходный imm.
    depth[pc] = статическая глубина стека ДО инструкции.
    """
    fidx = mod.exports[name]
    body = mod.code[fidx]
    if b"\x10" in body:                 # есть call-байт -> инлайним
        body = inline_calls(mod, fidx)
    nl, p = _uleb(body, 0)
    nloc = 0
    for _ in range(nl):
        cnt, p = _uleb(body, p)
        p += 1
        nloc += cnt
    nparams = len(mod.params[mod.funcs[mod.exports[name]]])

    instrs, cstack = [], []
    fixups = []          # (instr_idx, block_dict|list, slot)
    end_pc_of_func = None
    cstack.append(dict(kind="func", head=-1, else_pc=None))
    while p < len(body):
        pc = len(instrs)
        op = body[p]; p += 1
        if op == 0xFC:
            sub, p = _uleb(body, p)
            nm, ik, pop, push = FC_OPS[sub]
        else:
            nm, ik, pop, push = OPS[op]
        imm = 0
        if ik == "u32":
            imm, p = _uleb(body, p)
        elif ik == "i32":
            imm, p = _sleb(body, p, 32)
        elif ik == "i64":
            imm, p = _sleb(body, p, 64)
        elif ik == "f32":
            imm = struct.unpack("<f", body[p:p + 4])[0]; p += 4
        elif ik == "f64":
            imm = struct.unpack("<d", body[p:p + 8])[0]; p += 8
        elif ik == "bt":
            imm = body[p]; p += 1
        elif ik == "mem":
            _, p = _uleb(body, p)
            imm, p = _uleb(body, p)
        elif ik == "table":
            nt, p = _uleb(body, p)
            tgts = []
            for _ in range(nt):
                t, p = _uleb(body, p)
                tgts.append(t)
            dflt, p = _uleb(body, p)
            imm = (tgts, dflt)

        if nm in ("block", "loop", "if"):
            cstack.append(dict(kind=nm, head=pc, else_pc=None))
        elif nm == "else":
            cstack[-1]["else_pc"] = pc
        elif nm == "end":
            blk = cstack.pop()
            after = pc + 1
            if blk["kind"] == "if":
                tgt = blk["else_pc"] + 1 if blk["else_pc"] is not None else after
                instrs[blk["head"]][1] = tgt
                if blk["else_pc"] is not None:
                    instrs[blk["else_pc"]][1] = after
            blk["tgt"] = blk["head"] + 1 if blk["kind"] == "loop" else after
            blk["end_pc"] = pc
            if not cstack and blk["kind"] == "func":
                end_pc_of_func = pc
        elif nm == "br" or nm == "br_if":
            fixups.append((pc, cstack[-1 - imm], None))
        elif nm == "br_table":
            tgts, dflt = imm
            fixups.append((pc, cstack[-1 - dflt], None))
            for j, k in enumerate(tgts):
                fixups.append((pc, cstack[-1 - k], j))
        instrs.append([nm, imm])

    halt_pc = len(instrs)
    instrs.append(["halt", 0])  # br на func-блок / fall-through = halt

    # резолвим метки: imm <- абсолютный pc цели
    for pc, blk, slot in fixups:
        nm = instrs[pc][0]
        if nm == "br_table":
            cur = instrs[pc][1]
            if not isinstance(cur[0], list):
                instrs[pc][1] = ([None] * len(cur[0]), None)
            if slot is None:
                instrs[pc][1] = (instrs[pc][1][0], blk["tgt"])
            else:
                instrs[pc][1][0][slot] = blk["tgt"]
        else:
            instrs[pc][1] = blk["tgt"]

    # статическая глубина стека: обход графа, entry depth=0.
    # end функции = halt. return = halt.
    n = len(instrs)
    depth = [-1] * n
    depth[0] = 0
    wl = [0]
    while wl:
        pc = wl.pop()
        d = depth[pc]
        nm, imm = instrs[pc]
        pop, push = NAME_PP[nm]
        if nm in ("return",):
            continue
        if nm == "end" and pc == end_pc_of_func:
            continue
        succ = []
        if nm == "if":
            succ = [pc + 1, imm]
            d_out = d - 1
        elif nm == "else":
            succ = [imm]; d_out = d
        elif nm == "br":
            succ = [imm]; d_out = d
        elif nm == "br_if":
            succ = [pc + 1, imm]; d_out = d - 1
        elif nm == "br_table":
            succ = list(imm[0]) + [imm[1]]; d_out = d - 1
        elif nm in ("unreachable",):
            continue
        elif nm == "call" or nm == "call_indirect":
            raise NotImplementedError("call не поддержан — инлайнь код")
        else:
            succ = [pc + 1] if pc + 1 < n else []
            d_out = d - pop + push
        for s in succ:
            if s >= n:
                continue
            if depth[s] == -1:
                depth[s] = d_out
                wl.append(s)
            else:
                assert depth[s] == d_out, f"глубина стека не статична pc={s}"

    names = [i[0] for i in instrs]
    imms = [i[1] for i in instrs]
    pops = [NAME_PP[i[0]][0] for i in instrs]
    pushs = [NAME_PP[i[0]][1] for i in instrs]
    return names, imms, pops, pushs, depth, nloc, nparams, end_pc_of_func


if __name__ == "__main__":
    import sys, collections
    mod = Module(sys.argv[1] if len(sys.argv) > 1
                 else "wasm-progs/target/wasm32-unknown-unknown/release/wasm_progs.wasm")
    print("exports:", mod.exports)
    print("globals:", mod.globals, "mem_min:", mod.mem_min)
    print("data:", [(o, len(d)) for o, d in mod.data])
    for nm in sys.argv[2:] or mod.exports:
        names, imms, pops, pushs, depth, nloc, nparams, epc = decode(mod, nm)
        cnt = collections.Counter(names)
        print(f"\n== {nm}: {len(names)} instr, {nloc} locals+{nparams} params, "
              f"maxdepth={max(d for d in depth if d >= 0)}")
        print(" ops:", dict(cnt))
        for i, (a, b, dd) in enumerate(zip(names, imms, depth)):
            print(f"  {i:3d} d={dd:2d} {a:14s} {b}")
