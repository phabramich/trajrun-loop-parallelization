"""Chunked-Jacobi исполнитель поверх batched wasm-VM.

Схема = parareal/MSSP: неизвестные — состояния на границах каждые K
инструкций. Свип: rollout всех чанков батчем (K VM-шагов на линию),
обновление Якоби S[c+1] = E[c]. Fixpoint = сертификат точности.

Угадчик: короткий последовательный пробег 2 чанков (coarse propagator),
линейная экстраполяция локалов (ловит индукционные переменные точно),
pc — по фазе цикла. Это измеряет заживляемость ДАННЫХ в чистом виде.
"""

import time
import numpy as np
import vm_metal
from vm_ref import Program
from vm_metal import BatchVM


def boundary_states(ref_states, K, nsteps):
    """S[c] = состояние после cK инструкций; ref_states[i] = состояние
    ПОСЛЕ i шагов (ref_states[0] = начальное)."""
    return [ref_states[min(c * K, nsteps)] for c in range(nsteps // K + 1)]


def pack_state_rows(rows, L, D):
    C = len(rows)
    pc = np.array([r[0] for r in rows], np.int32)
    loc = np.array([r[1] for r in rows], np.uint32)
    loc = np.pad(loc, ((0, 0), (0, L - loc.shape[1])))
    stk = np.array([r[2] for r in rows], np.uint32)
    stk = np.pad(stk, ((0, 0), (0, D - stk.shape[1])))
    return pc, loc, stk


def probe_guess(rows, C, L, D, n_known=2):
    """Угадчик: точные первые n_known границ + линейная экстраполяция
    локалов и pc по двум последним точкам."""
    pc, loc, stk = pack_state_rows(rows[:n_known + 1] +
                                   [rows[-1]] * (C + 1 - n_known - 1),
                                   L, D)
    a, b = rows[n_known - 1], rows[n_known]
    dl = (np.asarray(b[1], np.int64) - np.asarray(a[1], np.int64))
    dp = int(b[0]) - int(a[0])
    for c in range(n_known + 1, C + 1):
        k = c - n_known
        loc[c] = (np.asarray(b[1], np.int64) + dl * k) % (1 << 32)
        pc[c] = b[0] + dp * k
        stk[c] = np.asarray(b[2], np.uint32)
    return pc, loc, stk


def zeros_guess(rows, C, L, D):
    pc, loc, stk = pack_state_rows([rows[0]] * (C + 1), L, D)
    return pc, loc, stk


def state_pack(S):
    pc, loc, stk = S
    return np.asarray(pc), np.asarray(loc), np.asarray(stk)


def state_diff(E, S):
    """Число позиций, где E и S различаются."""
    ep, el, es = state_pack(E)
    sp, sl, ss = state_pack(S)
    diff = (ep != sp) | (el != sl).any(1) | (es != ss).any(1)
    return diff


def run_parallel(prog: Program, mem, args, K, guesser="probe2",
                 max_sweeps=10000, ref_states=None, verbose=True,
                 backend=None):
    """Полный прогон: скалярная разметка -> угадывание -> Jacobi на MLX.

    Возвращает dict с метриками. ref_states — опорная трасса для сверки
    (в бою её нет — используем только для диагностики)."""
    t0 = time.time()
    states, stores, nsteps = prog.run(mem, args, keep_every=K)
    t_ref = time.time() - t0
    rows = states              # rows[c] = состояние после cK шагов
    C = len(rows) - 1
    vm = BatchVM(prog, bytes(mem), backend=backend)
    L, D = vm.L, vm.D

    if guesser == "probe2":
        g = probe_guess(rows, C, L, D)
    else:
        g = zeros_guess(rows, C, L, D)
    mx = vm_metal.mx
    S = (mx.array(g[0]), mx.array(g[1]), mx.array(g[2]))
    # S[0] — точное входное состояние
    ref_b = pack_state_rows(rows, L, D)

    hist = []
    t1 = time.time()
    for sweep in range(1, max_sweeps + 1):
        E, _ = vm.rollout((S[0][:-1], S[1][:-1], S[2][:-1]), K)
        diff = state_diff(E, (S[0][1:], S[1][1:], S[2][1:]))
        mism = int(diff.sum())
        # Jacobi update: S[c+1] = E[c]
        S = (mx.concatenate([S[0][:1], E[0]]),
             mx.concatenate([S[1][:1], E[1]]),
             mx.concatenate([S[2][:1], E[2]]))
        mx.eval(S)
        # диагностика: длина точного префикса против ref
        if ref_states is not None:
            sp, sl, ss = state_pack(S)
            exact = (sp == ref_b[0]) & (sl == ref_b[1]).all(1) & \
                    (ss == ref_b[2]).all(1)
            pref = int(np.argmin(exact)) if not exact.all() else C + 1
        else:
            pref = -1
        hist.append((sweep, mism, pref))
        if verbose and (sweep <= 10 or sweep % 10 == 0):
            print(f"  sweep {sweep:4d}: mism={mism:6d} exact_prefix={pref}")
        if mism == 0:
            break
    t_par = time.time() - t1

    # сверка с опорной трассой
    sp, sl, ss = state_pack(S)
    ok = (sp == ref_b[0]) & (sl == ref_b[1]).all(1) & (ss == ref_b[2]).all(1)
    n_exact = int(ok.sum())
    return dict(
        nsteps=nsteps, K=K, chunks=C, sweeps=sweep, t_ref=t_ref,
        t_par=t_par, exact_boundaries=n_exact, total_boundaries=C + 1,
        depth_ratio=nsteps / max(1, sweep), hist=hist, S=S,
        stores=stores, rows=rows, vm=vm,
    )


def assemble_output(res, store_kind="i32.store16"):
    """Финальный проход: rollout с логом store'ов -> словарь addr->val."""
    vm = res["vm"]; K = res["K"]; S = res["S"]
    E, (la, lv) = vm.rollout((S[0][:-1], S[1][:-1], S[2][:-1]), K,
                           collect_stores=True)
    out = {}
    for sa, sv in zip(la, lv):
        a = np.asarray(sa); v = np.asarray(sv)
        for i in np.nonzero(a >= 0)[0]:
            out[int(a[i])] = int(v[i])
    return out
