"""run_parallel.py — тонкая обёртка над всем пайплайном.

Конвенция для Rust-функции:
    pub extern "C" fn name(in_ptr: *const u8, n_bytes: usize,
                           out_ptr: *mut u8, extra: u32...) -> usize
Один цикл, без вызовов, i32/f32. Вход кладётся в IN_ADDR, выход читается
из OUT_ADDR; выход = out_bytes байт (по умолчанию = len(input)).

CLI:  python run_parallel.py prog.wasm:fn in.bin out.bin [extra u32...]
API:  out, info = run_wasm("wasm_progs.wasm", "fsm_lex", data)
info: method, converged, hist, predict, enum_info — честная диагностика.
"""

import sys
import numpy as np
from wasm_load import Module
from vm_ref import Program, make_memory
from wasm_metal import MetalWasmVM

IN_ADDR, OUT_ADDR = 0x110000, 0x200000


def run_wasm(wasm_path, fn_name, data: bytes, extra_args=(),
             C=1024, iters_per_chunk=None, out_bytes=None,
             max_sweeps=30):
    """Гоняет fn на data на Metal. Возвращает (out_bytes, info)."""
    mod = Module(wasm_path)
    prog = Program(mod, fn_name)
    vm = MetalWasmVM(prog, fn_name)
    out_bytes = len(data) if out_bytes is None else out_bytes
    if iters_per_chunk is None:
        # грубая оценка: байт на итерацию (верно для стриминг-циклов)
        iters_per_chunk = max(1, (len(data) + C - 1) // C)
    mem = make_memory(mod, [(IN_ADDR, data), (OUT_ADDR, b"\x00" * out_bytes)])
    args = [IN_ADDR, len(data), OUT_ADDR, *extra_args]
    res = vm.run_auto(mem, args, C=C, iters_per_chunk=iters_per_chunk,
                      out_base=OUT_ADDR, out_bytes=out_bytes,
                      max_sweeps=max_sweeps)
    info = {k: res[k] for k in ("method", "converged", "hist", "predict",
                              "enum_info") if k in res}
    return res["out"].tobytes(), info


if __name__ == "__main__":
    target, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    wpath, fn = target.rsplit(":", 1)
    extra = [int(x) for x in sys.argv[4:]]
    with open(inp, "rb") as f:
        data = f.read()
    out, info = run_wasm(wpath, fn, data, extra)
    with open(outp, "wb") as f:
        f.write(out)
    print(f"{fn}: method={info['method']} converged={info['converged']} "
          f"hist={info['hist']} predict={info['predict']} "
          f"enum={info['enum_info']} -> {outp}")
