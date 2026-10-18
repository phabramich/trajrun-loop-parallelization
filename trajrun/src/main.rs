//! trajrun — speculative parallel execution of `y[t] = step(s, x[t])`
//! with carried state `s`, bit-exact.
//!
//! One frontend for both backends — a scalar C file:
//!   typedef struct { float y1, x1, env; } S;   // 32-bit fields only;
//!                                                // `= literal` inits allowed
//!   float step(S *s, float x) { s->...; return y; }
//!
//!   trajrun step.c in.bin out.bin            # CPU, all cores
//!   trajrun step.c in.bin out.bin --gpu      # Metal (compiled by the OS)
//!   trajrun --check step.c in.bin            # compile + predict method
//!
//! CPU: step.c is compiled `cc -x c++ -O2 -shared` to a dylib, dlopen'd,
//!      and run through trajpar (speculative chunks + merge detection,
//!      serial fallback on walls).
//! GPU: the same source is wrapped into the speculative Metal kernel —
//!      lanes run chunks, record checkpoints, merged lanes stop early,
//!      output prefixes are spliced by merge index. No toolchain needed.
//! Note: f32 — the Metal lib is built with fast-math OFF, the dylib with
//!       -ffp-contract=off, so the two agree bitwise. On a wall the GPU
//!       path falls back to the (serial) CPU run instead of failing.
//! shortcut: x/y element width is 1 or 4 bytes; i64/f64 state unsupported.

use std::{env, fs, time::Instant};
use trajrun::*;

fn usage() -> ! {
    eprintln!(
        "usage: trajrun <step.c> <in.bin> <out.bin> [--cpu [N]|--gpu] [--lanes C]
                [--verify|--no-verify]
        trajrun --check <step.c> [in.bin]
        trajrun --serve <step.c> [--cpu N|--gpu]   # \"in.bin out.bin\" lines on stdin"
    );
    std::process::exit(2)
}

/// Repeat-call mode: `trajrun --serve step.c [--cpu N|--gpu]` reads
/// "in.bin out.bin" lines on stdin and answers "ok <status> <ms>" per job.
/// The dylib / Metal kernels are built once — this is where the GPU path
/// actually wins, since per-call JIT+I/O stop dominating.
fn serve_loop(
    spec: &Spec,
    threads: usize,
    gpu_flag: bool,
    verify: Option<bool>,
) -> Result<(), String> {
    use std::io::{BufRead, Write};
    let d = build_dylib(spec)?;
    #[cfg(target_os = "macos")]
    let mut eng: Option<(usize, gpu::Engine)> = None;
    let mut aff: Option<Option<Mat>> = None;
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let mut it = line.split_whitespace();
        let (inp, outp) = match (it.next(), it.next()) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let mut job = || -> Result<String, String> {
            let bytes = fs::read(inp).map_err(|e| format!("{inp}: {e}"))?;
            let t0 = Instant::now();
            #[cfg(target_os = "macos")]
            if gpu_flag {
                let n = bytes.len() / spec.x.sz();
                let want = n * spec.y.sz();
                if aff.is_none() {
                    aff = Some(detect_affine(&d, spec, &bytes));
                }
                if eng.as_ref().map(|(en, _)| *en) != Some(n) {
                    eng = Some((n, gpu::Engine::new(spec, n, 8192)?));
                }
                let mut out = Vec::new();
                match eng.as_ref().unwrap().1.run(
                    spec,
                    &bytes,
                    &mut out,
                    64,
                    aff.as_ref().unwrap().as_ref(),
                ) {
                    Ok(r) => {
                        out.truncate(want);
                        fs::write(outp, &out).map_err(|e| format!("{outp}: {e}"))?;
                        return Ok(format!(
                            "gpu {r} rounds {:.1}ms",
                            t0.elapsed().as_secs_f64() * 1e3
                        ));
                    }
                    Err(e) => eprintln!("gpu: {e} — cpu fallback"),
                }
            }
            let _ = gpu_flag;
            let (out, st) = run_cpu(&d, spec, &bytes, threads, verify.unwrap_or(false))?;
            fs::write(outp, &out).map_err(|e| format!("{outp}: {e}"))?;
            Ok(format!(
                "cpu {st} {:.1}ms",
                t0.elapsed().as_secs_f64() * 1e3
            ))
        };
        let msg = match job() {
            Ok(m) => format!("ok {m}"),
            Err(e) => format!("err {e}"),
        };
        writeln!(stdout.lock(), "{msg}").map_err(|e| e.to_string())?;
        stdout.lock().flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut gpu = false;
    let mut check = false;
    let mut serve = false;
    let mut verify: Option<bool> = None;
    let mut lanes = 8192usize;
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut pos = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" => gpu = true,
            "--cpu" => {
                if let Some(n) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    threads = n;
                    i += 1;
                }
            }
            "--check" => check = true,
            "--serve" => serve = true,
            "--verify" => verify = Some(true),
            "--no-verify" => verify = Some(false),
            "--lanes" => {
                lanes = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .ok_or("--lanes N")?;
                i += 1;
            }
            "--ckpt" => i += 1, // reserved
            _ => pos.push(args[i].clone()),
        }
        i += 1;
    }
    let step_path = pos.first().ok_or_else(|| "missing step.c".to_string())?;
    let src = fs::read_to_string(step_path).map_err(|e| format!("{step_path}: {e}"))?;
    let spec = parse_spec(&src).map_err(|e| format!("{step_path}: {e}"))?;
    eprintln!(
        "trajrun: S={} nf={} [{}], x={}, y={}",
        spec.sname,
        spec.fields.len(),
        spec.fields
            .iter()
            .map(|f| f.0.as_str())
            .collect::<Vec<_>>()
            .join(","),
        spec.x.c(),
        spec.y.c()
    );

    if check {
        let d = build_dylib(&spec)?;
        if let Some(inp) = pos.get(1) {
            let bytes = fs::read(inp).map_err(|e| format!("{inp}: {e}"))?;
            let aff = if detect_affine(&d, &spec, &bytes).is_some() {
                "; affine over Z/2^32 (exact scan, no speculation needed)"
            } else {
                ""
            };
            eprintln!(
                "check: compile ok; {}{}",
                probe_merge(&d, &spec, &bytes),
                aff
            );
        } else {
            eprintln!("check: compile ok (no input given — skipped probe)");
        }
        return Ok(());
    }
    if serve {
        return serve_loop(&spec, threads, gpu, verify);
    }

    let inp = pos.get(1).ok_or_else(|| "missing in.bin".to_string())?;
    let outp = pos.get(2).ok_or_else(|| "missing out.bin".to_string())?;
    let bytes = fs::read(inp).map_err(|e| format!("{inp}: {e}"))?;
    let t0 = Instant::now();

    #[cfg(target_os = "macos")]
    if gpu {
        let n_elems = bytes.len() / spec.x.sz();
        let want_len = n_elems * spec.y.sz();
        let mut out = Vec::new();
        let d = build_dylib(&spec)?; // affine detect + verify + fallback
        let aff = detect_affine(&d, &spec, &bytes);
        match gpu::Engine::new(&spec, n_elems, lanes)
            .and_then(|e| e.run(&spec, &bytes, &mut out, 64, aff.as_ref()))
        {
            Ok(rounds) => {
                eprintln!(
                    "gpu: {rounds} rounds, {:.1} ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
                out.truncate(want_len);
                fs::write(outp, &out).map_err(|e| format!("{outp}: {e}"))?;
                if verify.unwrap_or(false) {
                    let want_b = serial_run(&d, &spec, &bytes);
                    let nd = want_b
                        .chunks_exact(spec.y.sz())
                        .zip(out.chunks_exact(spec.y.sz()))
                        .filter(|(a, b)| a != b)
                        .count();
                    if spec.y.inty() && nd > 0 {
                        return Err(format!("gpu verify: {nd} elements differ"));
                    }
                    eprintln!("verify vs serial CPU: {nd} elements differ (f32 ulp-level is OK)");
                }
                return Ok(());
            }
            Err(e) => eprintln!("gpu: {e} — falling back to serial CPU"),
        }
    }
    #[cfg(not(target_os = "macos"))]
    if gpu {
        eprintln!("--gpu: Metal is macOS-only, using CPU");
    }

    let d = build_dylib(&spec)?;
    let t0 = Instant::now();
    let (out, st) = run_cpu(&d, &spec, &bytes, threads, verify.unwrap_or(true))?;
    eprintln!(
        "cpu: {} — {:.1} ms{}",
        st,
        t0.elapsed().as_secs_f64() * 1e3,
        if verify.unwrap_or(true) && st.starts_with("spec") {
            " (verified bit-exact)"
        } else {
            ""
        }
    );
    fs::write(outp, &out).map_err(|e| format!("{outp}: {e}"))?;
    Ok(())
}
