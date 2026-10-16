"""Бенч: реальный wasm (rustc) -> batched-VM на Metal -> chunked Jacobi.

Метрики:
  nsteps   — последовательные VM-шаги (serial depth)
  sweeps   — параллельные раунды Якоби до fixpoint'а (parallel depth)
  crit     — sweeps*K: критический путь в VM-инструкциях
  ratio    — nsteps/crit: идеальный предел ускорения
  t_ref    — скалярный интерпретатор (python), сек
  t_par    — batched-исполнитель на MLX, сек
"""

import math, random, struct, time
import numpy as np
from wasm_load import Module
from vm_ref import Program, make_memory, adpcm_encode, gen_input
from vm_exec import run_parallel
from np_shim import np_shim

WASM = "wasm-progs/target/wasm32-unknown-unknown/release/wasm_progs.wasm"
IN, OUT = 0x20000, 0x40000


def csv_text(n, seed=3):
    random.seed(seed)
    out = bytearray()
    while len(out) < n:
        nfields = random.randint(3, 6)
        line = []
        for _ in range(nfields):
            w = "".join(random.choice("abc,\"\\de\n"[:-1])
                        for _ in range(random.randint(2, 9)))
            if random.random() < 0.4:
                w = '"' + w + '"'
            line.append(w)
        out += ",".join(line).encode() + b"\n"
    return bytes(out[:n])


def biased_i16(n, bias=3000, seed=4):
    random.seed(seed)
    return struct.pack(f"<{n}h",
                       *[bias + random.randint(-6000, 3000)
                         for _ in range(n)])


def run(mod, name, mem, args, K, max_sweeps, label, backend=None):
    prog = Program(mod, name)
    print(f"\n== {label}: {name} K={K} ==")
    res = run_parallel(prog, mem, args, K=K, guesser="probe2",
                       max_sweeps=max_sweeps, ref_states=True,
                       verbose=False, backend=backend)
    # точность выхода: собрать store'ы сходившейся трассы
    from vm_exec import assemble_output
    out = assemble_output(res)
    # опорный выход из scalar-прогона
    ref_out = {a: v for a, v, _ in res["stores"]}
    same = all(out.get(a, -1) == v for a, v in ref_out.items()) and \
        len(out) == len(ref_out)
    crit = res["sweeps"] * K
    print(f"  steps={res['nsteps']:>9,} chunks={res['chunks']:>6,} "
          f"sweeps={res['sweeps']:>5}")
    print(f"  crit={crit:>9,} ratio=T/crit={res['nsteps']/crit:>7.2f}x  "
          f"exact={res['exact_boundaries']}/{res['total_boundaries']} "
          f"out_ok={same}")
    print(f"  t_ref={res['t_ref']:6.2f}s t_par={res['t_par']:6.2f}s")
    # история: как рос exact_prefix
    hs = res["hist"]
    marks = [h for h in hs if h[0] in (1, 2, 5, 10, 20, 50, 100, 200)]
    print("  prefix:", ", ".join(f"s{s}={p}" for s, _, p in marks[:8]))
    return res


def main():
    mod = Module(WASM)

    def period_of(name, data, mkargs, nprobe):
        """Инструкций на байт/элемент входа: probe на первых nprobe байтах."""
        p = Program(mod, name)
        mem = make_memory(mod, [(IN, data[:nprobe])])
        _, _, n = p.run(mem, mkargs(nprobe))
        return n / nprobe

    # --- 1. fsm_lex: текст с \n ~ раз в 64 байта -> слияние на ресетах
    data = csv_text(32768)
    mem = make_memory(mod, [(IN, data)])
    args = [IN, len(data), OUT]
    per = period_of("fsm_lex", data, lambda n: [IN, n, OUT], 2048)
    K = int(round(per)) * 64
    run(mod, "fsm_lex", mem, args, K=K, max_sweeps=2000,
        label=f"csv 32KB (per={per:.0f})", backend=np_shim)

    # --- 2. sat_acc: biased вход -> клампы частые
    data = biased_i16(8192, bias=12000)  # насыщение -> acc=1M аттрактор
    mem = make_memory(mod, [(IN, data)])
    args = [IN, 8192, OUT]
    per = period_of("sat_acc", data, lambda n: [IN, n // 2, OUT], 4096)
    # per=14.25/байт -> паттерн 57 инстр. на 4 байта; K кратен паттерну
    K = 57 * 50
    run(mod, "sat_acc", mem, args, K=K, max_sweeps=2000,
        label=f"sat 8K smp (per={per:.2f})", backend=np_shim)

    # --- 3. adpcm: состояние не заживает -> волновой фронт
    data = adpcm_encode(gen_input(1024))
    mem = make_memory(mod, [(IN, data)])
    args = [IN, len(data), OUT]
    per = period_of("adpcm_decode", data, lambda n: [IN, n, OUT], 256)
    K = int(round(per)) * 8
    run(mod, "adpcm_decode", mem, args, K=K, max_sweeps=40,
        label=f"adpcm 1K smp (per={per:.0f})", backend=np_shim)

    # --- 4. hash: стена
    data = bytes(random.Random(7).randbytes(8192))
    mem = make_memory(mod, [(IN, data)])
    args = [IN, len(data), OUT]
    per = period_of("hash_chain", data, lambda n: [IN, n, OUT], 2048)
    K = 49 * 32  # паттерн 49 инстр. на 2 байта
    run(mod, "hash_chain", mem, args, K=K, max_sweeps=40,
        label=f"hash 8KB (per={per:.2f})", backend=np_shim)


if __name__ == "__main__":
    main()
