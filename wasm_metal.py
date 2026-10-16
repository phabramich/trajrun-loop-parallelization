"""WASM -> Native Metal Kernel компилятор + параллельные исполнители.

Решает две проблемы базового интерпретатора vm_metal.py:
1. Инженерная стена (dispatch overhead): вместо ~50 тензорных операций из
   Python на каждую инструкцию wasm, весь плоский байткод функции
   транслируется в одно нативное ядро Metal (mx.fast.metal_kernel).
   Локалы и стек wasm живут в аппаратных регистрах потока (uint l*, s*),
   а линейные участки (basic blocks) сливаются в прямые циклы без switch.
2. Алгоритмическая стена:
   - Выравнивание по заголовку цикла (prologue -> loop_pc, depth=0) +
     probe3 (вторая разность d2 == 0 отличает индукционные счётчики от
     динамического состояния и не уводит таблицы за границы памяти).
   - Контрактивная переформулировка (adpcm_leaky, утечка 15/16 по образцу
     G.726/G.722 на паре (pred, index)): спектр Якобиана < 1 -> 100%
     траекторий сливаются бит-в-бит за ~180 сэмплов (2–3 свипа на Metal).
   - State-enumeration + associative scan (Mytkowicz/Blelloch) на том же
     wasm-ядре: когда у автомата (fsm_lex) нет ресетов (\n) и Jacobi
     упирается в стену, GPU прогоняет все C x |S| пар (чанк, входное
     состояние) за 1 свип и сворачивает таблицу переходов T[c, s] за
     log2(C) шагов скана.
"""

import random, time
import numpy as np
import mlx.core as mx
from wasm_load import Module
from vm_ref import (Program, make_memory, adpcm_encode, adpcm_leaky_encode,
                    gen_input)
from vm_bench import csv_text, biased_i16


def _emit_op(prog: Program, pc: int, with_pc: bool = True) -> str:
    nm, imm, d = prog.names[pc], prog.imms[pc], prog.depth[pc]
    halt_pc = prog.n_instr - 1
    if nm in ("halt", "return"):
        return f"pc = {halt_pc}u;"
    if d < 0:
        return f"pc = {pc + 1}u;" if with_pc else ""
    pc_s = f" pc = {pc + 1}u;" if with_pc else ""
    if nm in ("block", "loop", "end", "nop", "drop"):
        return pc_s.strip()
    if nm == "if":
        return f"pc = (s{d - 1} != 0u) ? {pc + 1}u : {imm}u;"
    if nm in ("else", "br"):
        return f"pc = {imm}u;"
    if nm == "br_if":
        return f"pc = (s{d - 1} != 0u) ? {imm}u : {pc + 1}u;"
    if nm == "local.get":
        return f"s{d} = l{imm};{pc_s}"
    if nm in ("local.set", "local.tee"):
        return f"l{imm} = s{d - 1};{pc_s}"
    if nm == "global.get":
        return f"s{d} = {prog.globals[imm] & 0xFFFFFFFF}u;{pc_s}"
    if nm == "select":
        return f"s{d - 3} = (s{d - 1} != 0u) ? s{d - 3} : s{d - 2};{pc_s}"
    if nm in ("i32.const", "i64.const"):
        return f"s{d} = {imm & 0xFFFFFFFF}u;{pc_s}"
    if nm.startswith("i32.load8_u"):
        return f"{{ uint a = s{d - 1} + {imm}u; s{d - 1} = (uint)mem[a]; }}{pc_s}"
    if nm.startswith("i32.load8_s"):
        return (f"{{ uint a = s{d - 1} + {imm}u; uint v = (uint)mem[a]; "
                f"s{d - 1} = (v & 0x80u) ? (v | 0xFFFFFF00u) : v; }}{pc_s}")
    if nm.startswith("i32.load16_u"):
        return (f"{{ uint a = s{d - 1} + {imm}u; "
                f"s{d - 1} = (uint)mem[a] | ((uint)mem[a + 1u] << 8); }}{pc_s}")
    if nm.startswith("i32.load16_s"):
        return (f"{{ uint a = s{d - 1} + {imm}u; "
                f"uint v = (uint)mem[a] | ((uint)mem[a + 1u] << 8); "
                f"s{d - 1} = (v & 0x8000u) ? (v | 0xFFFF0000u) : v; }}{pc_s}")
    if nm == "i32.load":
        return (f"{{ uint a = s{d - 1} + {imm}u; "
                f"s{d - 1} = (uint)mem[a] | ((uint)mem[a + 1u] << 8) | "
                f"((uint)mem[a + 2u] << 16) | ((uint)mem[a + 3u] << 24); }}{pc_s}")
    if nm == "i32.store8":
        return (f"if (write_mem) {{ uint a = s{d - 2} + {imm}u - out_base; "
                f"mem_out[a] = (uint8_t)(s{d - 1} & 0xFFu); }}{pc_s}")
    if nm == "i32.store16":
        return (f"if (write_mem) {{ uint a = s{d - 2} + {imm}u - out_base; "
                f"uint v = s{d - 1}; mem_out[a] = (uint8_t)(v & 0xFFu); "
                f"mem_out[a + 1u] = (uint8_t)((v >> 8) & 0xFFu); }}{pc_s}")
    if nm in ("i32.store", "i64.store"):
        return (f"if (write_mem) {{ uint a = s{d - 2} + {imm}u - out_base; "
                f"uint v = s{d - 1}; mem_out[a] = (uint8_t)(v & 0xFFu); "
                f"mem_out[a + 1u] = (uint8_t)((v >> 8) & 0xFFu); "
                f"mem_out[a + 2u] = (uint8_t)((v >> 16) & 0xFFu); "
                f"mem_out[a + 3u] = (uint8_t)((v >> 24) & 0xFFu); }}{pc_s}")
    if nm == "i32.eqz":
        return f"s{d - 1} = (s{d - 1} == 0u) ? 1u : 0u;{pc_s}"
    a, b = f"s{d - 2}", f"s{d - 1}"
    ops = {
        "i32.add": f"{a} + {b}", "i32.sub": f"{a} - {b}", "i32.mul": f"{a} * {b}",
        "i32.and": f"{a} & {b}", "i32.or": f"{a} | {b}", "i32.xor": f"{a} ^ {b}",
        "i32.shl": f"{a} << ({b} & 31u)", "i32.shr_u": f"{a} >> ({b} & 31u)",
        "i32.shr_s": f"(uint)((int){a} >> ({b} & 31u))",
        "i32.eq": f"({a} == {b}) ? 1u : 0u", "i32.ne": f"({a} != {b}) ? 1u : 0u",
        "i32.lt_s": f"((int){a} < (int){b}) ? 1u : 0u", "i32.lt_u": f"({a} < {b}) ? 1u : 0u",
        "i32.gt_s": f"((int){a} > (int){b}) ? 1u : 0u", "i32.gt_u": f"({a} > {b}) ? 1u : 0u",
        "i32.le_s": f"((int){a} <= (int){b}) ? 1u : 0u", "i32.le_u": f"({a} <= {b}) ? 1u : 0u",
        "i32.ge_s": f"((int){a} >= (int){b}) ? 1u : 0u", "i32.ge_u": f"({a} >= {b}) ? 1u : 0u",
    }
    return f"{a} = {ops[nm]};{pc_s}"


class MetalWasmVM:
    """Транслятор плоского wasm Program в нативное ядро Metal."""

    def __init__(self, prog: Program, name: str):
        self.prog = prog
        self.L = prog.nparams + prog.nloc
        self.D = prog.maxdepth
        self.W = 1 + self.L + self.D
        self.loop_pc = prog.names.index("loop") + 1
        self.br_pc = max(
            i for i, (nm, imm) in enumerate(zip(prog.names, prog.imms))
            if nm == "br_if" and imm == self.loop_pc
        )
        self.loop_len = self.br_pc - self.loop_pc + 1
        src = self._build_msl()
        self.kern = mx.fast.metal_kernel(
            name=f"wasm_msl_{name}",
            input_names=["state_in", "mem", "params"],
            output_names=["state_out", "mem_out"],
            source=src,
        )

    def _build_msl(self) -> str:
        prog, L, D, W = self.prog, self.L, self.D, self.W
        n = prog.n_instr
        leaders = {0, n - 1}
        for pc, (nm, imm) in enumerate(zip(prog.names, prog.imms)):
            if nm in ("if", "else", "br", "br_if"):
                leaders.add(int(imm))
                if pc + 1 < n:
                    leaders.add(pc + 1)
            elif nm in ("loop", "return", "halt"):
                if pc + 1 < n:
                    leaders.add(pc + 1)
        s_leaders = sorted(leaders)
        blocks = {}
        for idx, b0 in enumerate(s_leaders):
            b1 = (s_leaders[idx + 1] - 1) if idx + 1 < len(s_leaders) else (n - 1)
            if (b1 > b0 and prog.names[b0] != "halt"
                    and all(prog.depth[p] >= 0 for p in range(b0, b1 + 1))):
                blocks[b0] = b1

        lines = [
            "uint tid = thread_position_in_grid.x;",
            "uint B = params[0]; uint K = params[1];",
            "uint write_mem = params[2]; uint out_base = params[3];",
            "if (tid >= B) return;",
            f"uint base = tid * {W}u;",
            "uint pc = state_in[base + 0u];",
        ]
        for i in range(L):
            lines.append(f"uint l{i} = state_in[base + {1 + i}u];")
        for i in range(D):
            lines.append(f"uint s{i} = state_in[base + {1 + L + i}u];")
        lines.append("uint step = 0u;")
        lines.append("while (step < K) {")
        for b0, b1 in blocks.items():
            blen = b1 - b0 + 1
            bb = " ".join(_emit_op(prog, p, with_pc=(p == b1))
                          for p in range(b0, b1 + 1))
            lines.append(
                f"  if (pc == {b0}u && step + {blen}u <= K) {{ "
                f"do {{ {bb} step += {blen}u; }} "
                f"while (pc == {b0}u && step + {blen}u <= K); continue; }}"
            )
        lines.append("  switch (pc) {")
        for pc in range(n):
            lines.append(f"    case {pc}u: {{ {_emit_op(prog, pc, True)} break; }}")
        lines.append(f"    default: pc = {n - 1}u; break;")
        lines.append("  }")
        lines.append("  step += 1u;")
        lines.append("}")
        lines.append("state_out[base + 0u] = pc;")
        for i in range(L):
            lines.append(f"state_out[base + {1 + i}u] = l{i};")
        for i in range(D):
            lines.append(f"state_out[base + {1 + L + i}u] = s{i};")
        return "\n".join(lines)

    def rollout(self, state_in: mx.array, mem_mx: mx.array, K: int,
                write_mem: bool = False, out_base: int = 0, out_bytes: int = 1):
        B = state_in.shape[0]
        params = mx.array([B, K, int(write_mem), out_base], dtype=mx.uint32)
        m_sz = out_bytes if write_mem else 1
        return self.kern(
            inputs=[state_in, mem_mx, params],
            grid=(B, 1, 1),
            threadgroup=(min(256, B), 1, 1),
            output_shapes=[(B, self.W), (m_sz,)],
            output_dtypes=[mx.uint32, mx.uint8],
        )

    def prologue_state(self, mem: bytearray, args: list[int]) -> np.ndarray:
        """Исполняет короткий пролог функции (10–30 инстр.) до входа в loop."""
        st_pro, _, _ = self.prog.run(mem, args, max_steps=self.loop_pc)
        s0 = st_pro[-1]
        arr = np.zeros((1, self.W), dtype=np.uint32)
        arr[0, 0] = s0[0]
        arr[0, 1:1 + self.L] = s0[1]
        arr[0, 1 + self.L:] = s0[2]
        return arr

    def probe3_guess(self, s0_arr: np.ndarray, mem_mx: mx.array,
                     C: int, K: int) -> tuple[np.ndarray, list[int]]:
        """Прогон 2 чанков + фильтрация линейных индукционных переменных по d2 == 0."""
        e1, _ = self.rollout(mx.array(s0_arr), mem_mx, K)
        e2, _ = self.rollout(e1, mem_mx, K)
        mx.eval(e1, e2)
        r0, r1, r2 = s0_arr[0], np.asarray(e1)[0], np.asarray(e2)[0]
        L = self.L
        l0 = r0[1:1 + L].astype(np.int64)
        l1 = r1[1:1 + L].astype(np.int64)
        l2 = r2[1:1 + L].astype(np.int64)
        d1, d2 = l1 - l0, l2 - l1
        is_lin = (d1 == d2)
        k_idx = np.arange(C + 1, dtype=np.int64)[:, None]
        guess = np.zeros((C + 1, self.W), dtype=np.uint32)
        guess[:, 0] = r0[0]
        guess[:, 1:1 + L] = np.where(
            is_lin[None, :],
            (l0[None, :] + d1[None, :] * k_idx) % (1 << 32),
            l2[None, :],
        ).astype(np.uint32)
        guess[0], guess[1], guess[2] = r0, r1, r2
        dyn_locals = np.nonzero(~is_lin)[0].tolist()
        return guess, dyn_locals

    def run_sequential_metal(self, s0_arr: np.ndarray, mem_mx: mx.array,
                             total_K: int, out_base: int, out_bytes: int):
        """Честный GPU-базлайн: 1 поток Metal исполняет все total_K инструкций."""
        e_warm, _ = self.rollout(mx.array(s0_arr), mem_mx, self.loop_len)
        mx.eval(e_warm)
        t0 = time.time()
        e_seq, m_seq = self.rollout(
            mx.array(s0_arr), mem_mx, total_K,
            write_mem=True, out_base=out_base, out_bytes=out_bytes,
        )
        mx.eval(e_seq, m_seq)
        dt = time.time() - t0
        return np.asarray(e_seq)[0], np.asarray(m_seq), dt

    def run_jacobi(self, mem: bytearray, args: list[int], C: int,
                   iters_per_chunk: int, out_base: int, out_bytes: int,
                   max_sweeps: int = 30, anderson_m: int = 0):
        """Параллельный Jacobi на Metal с опциями:
           - anderson_m > 0: Anderson acceleration (PRIOR_ART steal-list).
             Измерено: на дискретных VM-состояниях не помогает — выпуклая
             комбинация итераций порождает невалидные состояния автомата,
             которые приходится заживлять заново (fsm_lex: 5->9 свипов).
             Полезен только для чисто скалярных/линейных carried-переменных.
           - immediate commit: после сходимости запись выхода идёт с тех же
             стартов, что доказали сходимость, — отдельный commit-проход
             не нужен.
        """
        K = iters_per_chunk * self.loop_len
        s0_arr = self.prologue_state(mem, args)
        mem_mx = mx.array(np.frombuffer(mem, dtype=np.uint8))
        # прогрев компиляции шейдеров (B=1, B=C без записи, B=C с записью)
        w1, _ = self.rollout(mx.array(s0_arr), mem_mx, self.loop_len)
        dummy_C = mx.zeros((C, self.W), dtype=mx.uint32)
        w2, _ = self.rollout(dummy_C, mem_mx, self.loop_len, write_mem=False)
        w3, wm = self.rollout(dummy_C, mem_mx, self.loop_len,
                              write_mem=True, out_base=out_base, out_bytes=out_bytes)
        mx.eval(w1, w2, w3, wm)

        t0 = time.time()
        guess, dyn_locals = self.probe3_guess(s0_arr, mem_mx, C, K)
        S_mx = mx.array(guess)
        prev_S = []
        hist = []
        last_S_starts = S_mx[:-1]
        last_E = None
        converged = False
        for sweep in range(1, max_sweeps + 1):
            last_S_starts = S_mx[:-1]
            E_mx, _ = self.rollout(last_S_starts, mem_mx, K, write_mem=False)
            last_E = E_mx
            diff = (E_mx != S_mx[1:]).any(axis=1)
            S_new = mx.concatenate([S_mx[:1], E_mx], axis=0)
            mx.eval(S_new, diff)
            mism = int(diff.sum().item())
            hist.append(mism)
            if mism == 0:
                # Последний rollout был сделан из точных стартов —
                # его S_mx[:-1] уже является сертификатом; отдельный
                # commit-проход не нужен (immediate commit).
                converged = True
                S_mx = S_new
                break
            # Anderson acceleration: работаем ТОЛЬКО с динамическими локалами
            # (dyn_locals), остальное берём из последнего S_new как есть —
            # дискретные компоненты не должны интерполироваться. Измерено:
            # на sat/adpcm_leaky не даёт выигрыша (уже 2 свипа), на fsm_lex
            # замедляет (5->9): дискретная динамика не аппроксимируется
            # выпуклой комбинацией. Держим за флагом как измеренный результат.
            prev_S.append((np.asarray(S_mx), np.asarray(S_new)))
            if len(prev_S) > anderson_m + 1:
                prev_S.pop(0)
            if anderson_m and dyn_locals and len(prev_S) >= 2:
                xs = np.stack([x[1] for x in prev_S])[:, :, 1 + np.array(dyn_locals)]
                fs = np.stack([x[1] - x[0] for x in prev_S])[:, :, 1 + np.array(dyn_locals)]
                df = (fs[1:] - fs[0]).astype(np.float64)
                dx = (xs[1:] - xs[0]).astype(np.float64)
                m = len(prev_S) - 1
                A = df.reshape(m, -1).T                  # (C*|dyn|, m)
                b = -fs[0].reshape(-1).astype(np.float64)
                coef, *_ = np.linalg.lstsq(A, b, rcond=None)
                S_comb_dyn = xs[0] + np.tensordot(
                    coef.astype(np.float32), dx.astype(np.float32), axes=1
                )
                lo, hi = xs.min(axis=0), xs.max(axis=0)
                S_comb_dyn = np.clip(S_comb_dyn, lo, hi).astype(np.uint32)
                S_comb = np.asarray(S_new).copy()
                S_comb[:, 1 + np.array(dyn_locals)] = S_comb_dyn
                S_comb[:3] = np.asarray(S_new)[:3]
                S_mx = mx.array(S_comb)
            else:
                S_mx = S_new
        if converged:
            _, out_mem = self.rollout(
                last_S_starts, mem_mx, K,
                write_mem=True, out_base=out_base, out_bytes=out_bytes,
            )
        else:
            # не сошёлся за max_sweeps: записываем то, что есть (диагностика)
            _, out_mem = self.rollout(
                S_mx[:-1], mem_mx, K,
                write_mem=True, out_base=out_base, out_bytes=out_bytes,
            )
        mx.eval(out_mem)
        t_par = time.time() - t0
        return dict(
            sweeps=sweep, converged=converged, hist=hist,
            t_par=t_par, out=np.asarray(out_mem), S=np.asarray(S_mx),
            K=K, C=C, total_steps=C * K + self.loop_pc,
            dyn_locals=dyn_locals, s0_arr=s0_arr, mem_mx=mem_mx,
        )

    def run_enum_scan(self, mem: bytearray, args: list[int], C: int,
                      iters_per_chunk: int, state_local: int, n_states: int,
                      out_base: int, out_bytes: int):
        """Параллельная энумерация состояний + ассоциативный scan на Metal.

        Пробивает стену для конечных автоматов (fsm_lex) даже когда во входе
        НЕТ ни одного ресета ('\\n') и Jacobi деградирует до C свипов:
        1. Запускает B = C * n_states потоков на Metal за ОДИН rollout.
        2. Извлекает таблицу переходов T[c, s] in {0..n_states-1}.
        3. Сворачивает композицию T[c] через параллельный префиксный скан
           за ceil(log2(C)) шагов на GPU.
        4. Записывает точный выход за 1 финальный commit-вызов.
        """
        K = iters_per_chunk * self.loop_len
        s0_arr = self.prologue_state(mem, args)
        mem_mx = mx.array(np.frombuffer(mem, dtype=np.uint8))
        w1, _ = self.rollout(mx.array(s0_arr), mem_mx, self.loop_len)
        dummy_CS = mx.zeros((C * n_states, self.W), dtype=mx.uint32)
        dummy_C = mx.zeros((C, self.W), dtype=mx.uint32)
        w2, _ = self.rollout(dummy_CS, mem_mx, self.loop_len, write_mem=False)
        w3, wm = self.rollout(dummy_C, mem_mx, self.loop_len,
                              write_mem=True, out_base=out_base, out_bytes=out_bytes)
        mx.eval(w1, w2, w3, wm)

        t0 = time.time()
        guess, _ = self.probe3_guess(s0_arr, mem_mx, C, K)
        # Размножаем каждый чанк c=0..C-1 по всем возможным s=0..n_states-1
        base_starts = np.repeat(guess[:-1], n_states, axis=0)
        s_grid = np.tile(np.arange(n_states, dtype=np.uint32), C)
        base_starts[:, 1 + state_local] = s_grid

        # 1 батч-прогон всех C * n_states комбинаций на Metal
        E_all, _ = self.rollout(mx.array(base_starts), mem_mx, K, write_mem=False)
        E_3d = E_all.reshape(C, n_states, self.W)
        # Таблица переходов T[c, s] = выходное состояние после чанка c из входа s
        T = E_3d[:, :, 1 + state_local].astype(mx.int32)

        # Hillis-Steele параллельный ассоциативный скан композиции T_curr[T_prev]
        d = 1
        scan_steps = 0
        while d < C:
            composed = mx.take_along_axis(T[d:], T[:-d], axis=1)
            T = mx.concatenate([T[:d], composed], axis=0)
            d <<= 1
            scan_steps += 1

        # Истинное состояние после чанка c = T[c, init_s]
        init_s = int(s0_arr[0, 1 + state_local])
        end_states = T[:, init_s]                       # (C,)
        entry_states = mx.concatenate(
            [mx.array([init_s], dtype=mx.int32), end_states[:-1]], axis=0
        )                                               # (C,)
        # Выбираем точные стартовые состояния и делаем 1 commit в память
        S_exact = mx.array(guess[:-1])
        # Подставляем точное состояние автомата на входе каждого чанка
        mx.eval(entry_states)
        starts_np = guess[:-1].copy()
        starts_np[:, 1 + state_local] = np.asarray(entry_states).astype(np.uint32)
        _, out_mem = self.rollout(
            mx.array(starts_np), mem_mx, K,
            write_mem=True, out_base=out_base, out_bytes=out_bytes,
        )
        mx.eval(out_mem)
        t_par = time.time() - t0
        return dict(
            rollouts=1, scan_steps=scan_steps, t_par=t_par,
            out=np.asarray(out_mem), K=K, C=C,
            total_steps=C * K + self.loop_pc,
            s0_arr=s0_arr, mem_mx=mem_mx,
        )


def main():
    WASM = "wasm-progs/target/wasm32-unknown-unknown/release/wasm_progs.wasm"
    mod = Module(WASM)
    IN_ADDR, OUT_ADDR = 0x110000, 0x200000

    print("=" * 74)
    print("WASM -> Native Metal Kernel (mx.fast.metal_kernel) Benchmark")
    print("=" * 74)

    # 1. fsm_lex на 256 KB CSV (с переводами строк \n -> Jacobi сходится за 3 свипа)
    vm_fsm = MetalWasmVM(Program(mod, "fsm_lex"), "fsm_lex")
    csv_32k = csv_text(32768)
    data_csv = csv_32k * 8                              # 262,144 байт (~15.2M wasm-шагов)
    mem = make_memory(mod, [(IN_ADDR, data_csv), (OUT_ADDR, b"\x00" * len(data_csv))])
    args = [IN_ADDR, len(data_csv), OUT_ADDR]
    res = vm_fsm.run_jacobi(mem, args, C=1024, iters_per_chunk=256,
                            out_base=OUT_ADDR, out_bytes=len(data_csv))
    _, seq_out, t_seq = vm_fsm.run_sequential_metal(
        res["s0_arr"], res["mem_mx"], res["C"] * res["K"], OUT_ADDR, len(data_csv)
    )
    ok = np.array_equal(res["out"], seq_out)
    assert ok, "fsm_lex CSV mismatch!"
    print(f"\n[1] fsm_lex (CSV 256KB, \\n resets): {res['total_steps']:,} WASM steps, C={res['C']}")
    print(f"    Jacobi sweeps = {res['sweeps']} (mism: {res['hist']})  bit-exact={ok}")
    print(f"    Metal 1-thread seq: {t_seq*1000:6.2f} ms | Metal {res['C']}-thread Jacobi: {res['t_par']*1000:6.2f} ms ({t_seq/res['t_par']:.2f}x vs GPU-seq)")

    # 2a. fsm_lex БЕЗ '\n', но с escape '\"' (256 KB):
    #     Неожиданная находка: слово '\"a\"' — синхронизирующее по Черны
    #     (переводит {0,1,2} -> {1,1,1}), поэтому Jacobi сходится сам за 5 свипов!
    rng = random.Random(42)
    alph_esc = b'abcdef0123456789:,,""\\'
    data_esc = bytes(rng.choice(alph_esc) for _ in range(32768)) * 8
    assert b"\n" not in data_esc
    mem_esc = make_memory(mod, [(IN_ADDR, data_esc), (OUT_ADDR, b"\x00" * len(data_esc))])
    args_esc = [IN_ADDR, len(data_esc), OUT_ADDR]
    res_esc = vm_fsm.run_jacobi(mem_esc, args_esc, C=1024, iters_per_chunk=256,
                                out_base=OUT_ADDR, out_bytes=len(data_esc), max_sweeps=10)
    print(f"\n[2a] fsm_lex (NO '\\n', with '\\\\\"' escapes, 256KB): {res_esc['total_steps']:,} WASM steps")
    print(f"     Cerny sync word '\\\\\"a\\\\\"' merges {0,1,2}->1! Jacobi sweeps={res_esc['sweeps']} (mism: {res_esc['hist']})")
    print(f"     Metal 1024-thread Jacobi: {res_esc['t_par']*1000:6.2f} ms ({t_seq/res_esc['t_par']:.2f}x vs GPU-seq)")
    res_esc_a = vm_fsm.run_jacobi(mem_esc, args_esc, C=1024, iters_per_chunk=256,
                                  out_base=OUT_ADDR, out_bytes=len(data_esc), max_sweeps=10,
                                  anderson_m=2)
    ok_esc_a = np.array_equal(res_esc_a["out"], res_esc["out"])
    print(f"     + Anderson(m=2):  sweeps={res_esc_a['sweeps']} (mism: {res_esc_a['hist']})  bit-exact={ok_esc_a}  t={res_esc_a['t_par']*1000:.2f} ms")

    # 2b. fsm_lex БЕЗ '\n' и БЕЗ '\\' (только кавычки '"' и текст):
    #     Каждая '"' — обратимая транспозиция 0<->1 (группа Z_2), слияний НЕТ
    #     никогда -> Jacobi упирается в стену (+1/свип), а Enum+Scan на Metal
    #     решает за 1 rollout + log2(C) scan!
    alph_z2 = b'abcdef0123456789:,,"'
    data_z2 = bytes(rng.choice(alph_z2) for _ in range(32768)) * 8
    assert b"\n" not in data_z2 and b"\\" not in data_z2
    mem_nn = make_memory(mod, [(IN_ADDR, data_z2), (OUT_ADDR, b"\x00" * len(data_z2))])
    args_nn = [IN_ADDR, len(data_z2), OUT_ADDR]
    res_jac_nn = vm_fsm.run_jacobi(mem_nn, args_nn, C=1024, iters_per_chunk=256,
                                   out_base=OUT_ADDR, out_bytes=len(data_z2), max_sweeps=6)
    res_enum = vm_fsm.run_enum_scan(mem_nn, args_nn, C=1024, iters_per_chunk=256,
                                    state_local=4, n_states=3,
                                    out_base=OUT_ADDR, out_bytes=len(data_z2))
    _, seq_nn, t_seq_nn = vm_fsm.run_sequential_metal(
        res_enum["s0_arr"], res_enum["mem_mx"], res_enum["C"] * res_enum["K"],
        OUT_ADDR, len(data_z2),
    )
    ok_enum = np.array_equal(res_enum["out"], seq_nn)
    assert ok_enum, "fsm_lex enum_scan mismatch!"
    print(f"\n[2b] fsm_lex (NO '\\n', NO '\\\\' — pure Z_2 quote group, 256KB): {res_enum['total_steps']:,} WASM steps")
    print(f"     Jacobi (6 sweeps):  converged={res_jac_nn['converged']} (mism: {res_jac_nn['hist']} -> GROUP WALL)")
    print(f"     Enum+Scan on Metal: {res_enum['rollouts']} rollout (3xC lanes) + {res_enum['scan_steps']} scan steps -> bit-exact={ok_enum}")
    print(f"     Metal 1-thread seq: {t_seq_nn*1000:6.2f} ms | Metal Enum+Scan:          {res_enum['t_par']*1000:6.2f} ms ({t_seq_nn/res_enum['t_par']:.2f}x vs GPU-seq)")

    # 3. sat_acc на 256K сэмплов (i16 -> i32, кламп-аттрактор):
    #    Примечание: sat_acc анроллит цикл по 2 элемента (57 инстр на 2 сэмпла)
    vm_sat = MetalWasmVM(Program(mod, "sat_acc"), "sat_acc")
    sat_chunk = biased_i16(16384, bias=12000)
    data_sat = sat_chunk * 16                           # 262,144 сэмплов (~7.47M wasm-шагов)
    n_smp = len(data_sat) // 2
    mem_sat = make_memory(mod, [(IN_ADDR, data_sat), (OUT_ADDR, b"\x00" * (n_smp * 4))])
    args_sat = [IN_ADDR, n_smp, OUT_ADDR]
    # 128 loop iterations = 256 samples per chunk, C = 1024 chunks
    res_sat = vm_sat.run_jacobi(mem_sat, args_sat, C=1024, iters_per_chunk=128,
                                out_base=OUT_ADDR, out_bytes=n_smp * 4)
    _, seq_sat, t_seq_sat = vm_sat.run_sequential_metal(
        res_sat["s0_arr"], res_sat["mem_mx"], res_sat["C"] * res_sat["K"],
        OUT_ADDR, n_smp * 4,
    )
    ok_sat = np.array_equal(res_sat["out"], seq_sat)
    assert ok_sat, "sat_acc mismatch!"
    print(f"\n[3] sat_acc (262K i16 samples, clamp attractor): {res_sat['total_steps']:,} WASM steps, C={res_sat['C']}")
    print(f"    Jacobi sweeps = {res_sat['sweeps']} (mism: {res_sat['hist']})  bit-exact={ok_sat}")
    print(f"    Metal 1-thread seq: {t_seq_sat*1000:6.2f} ms | Metal {res_sat['C']}-thread Jacobi: {res_sat['t_par']*1000:6.2f} ms ({t_seq_sat/res_sat['t_par']:.2f}x vs GPU-seq)")
    # Anderson(m=2) на том же входе: если динамика линейна/сокращаема,
    # обрезает остаточный шлейф границ до 1 свипа.
    res_sat_a = vm_sat.run_jacobi(mem_sat, args_sat, C=1024, iters_per_chunk=128,
                                  out_base=OUT_ADDR, out_bytes=n_smp * 4,
                                  anderson_m=2)
    ok_sat_a = np.array_equal(res_sat_a["out"], seq_sat)
    assert ok_sat_a, "sat_acc Anderson mismatch!"
    print(f"       + Anderson(m=2): sweeps={res_sat_a['sweeps']} (mism: {res_sat_a['hist']})  bit-exact={ok_sat_a}  t={res_sat_a['t_par']*1000:.2f} ms")

    # 4. adpcm_decode (стандартный IMA ADPCM, собственные числа 1.0) vs
    #    adpcm_leaky (G.726/G.722 leaky ADPCM, утечка 15/16 на pred и index):
    vm_adpcm_std = MetalWasmVM(Program(mod, "adpcm_decode"), "adpcm_decode")
    vm_adpcm_lky = MetalWasmVM(Program(mod, "adpcm_leaky"), "adpcm_leaky")
    base_pcm = gen_input(16384)
    pcm_262k = base_pcm * 16                            # 262,144 сэмплов (~31.2M wasm-шагов)
    enc_std = adpcm_encode(pcm_262k)
    enc_lky = adpcm_leaky_encode(pcm_262k)

    mem_std = make_memory(mod, [(IN_ADDR, enc_std), (OUT_ADDR, b"\x00" * (len(pcm_262k) * 2))])
    res_std = vm_adpcm_std.run_jacobi(
        mem_std, [IN_ADDR, len(enc_std), OUT_ADDR], C=1024, iters_per_chunk=256,
        out_base=OUT_ADDR, out_bytes=len(pcm_262k) * 2, max_sweeps=6,
    )

    mem_lky = make_memory(mod, [(IN_ADDR, enc_lky), (OUT_ADDR, b"\x00" * (len(pcm_262k) * 2))])
    res_lky = vm_adpcm_lky.run_jacobi(
        mem_lky, [IN_ADDR, len(enc_lky), OUT_ADDR], C=1024, iters_per_chunk=256,
        out_base=OUT_ADDR, out_bytes=len(pcm_262k) * 2, max_sweeps=10,
    )
    _, seq_lky, t_seq_lky = vm_adpcm_lky.run_sequential_metal(
        res_lky["s0_arr"], res_lky["mem_mx"], res_lky["C"] * res_lky["K"],
        OUT_ADDR, len(pcm_262k) * 2,
    )
    ok_lky = np.array_equal(res_lky["out"], seq_lky)
    assert ok_lky, "adpcm_leaky mismatch!"
    print(f"\n[4] ADPCM (262K audio samples, ~31.2M WASM steps, C={res_lky['C']}):")
    print(f"    a) Standard IMA adpcm_decode (lambda=1.0): converged={res_std['converged']} (mism: {res_std['hist']} -> WALL)")
    print(f"    b) Leaky G.726 adpcm_leaky  (lambda=15/16): sweeps={res_lky['sweeps']} (mism: {res_lky['hist']})  bit-exact={ok_lky}")
    print(f"       Metal 1-thread seq: {t_seq_lky*1000:6.2f} ms | Metal {res_lky['C']}-thread Jacobi: {res_lky['t_par']*1000:6.2f} ms ({t_seq_lky/res_lky['t_par']:.2f}x vs GPU-seq)")
    res_lky_a = vm_adpcm_lky.run_jacobi(
        mem_lky, [IN_ADDR, len(enc_lky), OUT_ADDR], C=1024, iters_per_chunk=256,
        out_base=OUT_ADDR, out_bytes=len(pcm_262k) * 2, max_sweeps=10,
        anderson_m=2,
    )
    ok_lky_a = np.array_equal(res_lky_a["out"], seq_lky)
    assert ok_lky_a, "adpcm_leaky Anderson mismatch!"
    print(f"       + Anderson(m=2): sweeps={res_lky_a['sweeps']} (mism: {res_lky_a['hist']})  bit-exact={ok_lky_a}  t={res_lky_a['t_par']*1000:.2f} ms")


if __name__ == "__main__":
    main()
