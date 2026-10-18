//! trajrun engine — speculative parallel execution of `y[t] = step(s, x[t])`
//! with carried state `s`, bit-exact vs serial.
//!
//! One frontend for both backends — a scalar C file:
//!   typedef struct { float y1, x1, env; } S;   // 32-bit fields only
//!   float step(S *s, float x) { s->...; return y; }
//!
//! CPU: step.c -> `cc -x c++ -O2 -shared` dylib, speculative chunks.
//! GPU: the same source wrapped into speculative Metal kernels.
//! f32: fast-math OFF on both sides so they agree bitwise.
//! shortcut: x/y element width is 1 or 4 bytes; i64/f64 state unsupported.

use std::{env, fs, process::Command};

// ---------------------------------------------------------------- spec --

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Ty {
    F32,
    U32,
    I32,
    U8,
}

impl Ty {
    pub fn parse(s: &str) -> Option<Ty> {
        Some(match s.trim() {
            "float" | "f32" => Ty::F32,
            "uint32_t" | "unsigned" | "unsigned int" | "uint" | "u32" => Ty::U32,
            "int32_t" | "int" | "i32" => Ty::I32,
            "uint8_t" | "unsigned char" | "uchar" | "u8" => Ty::U8,
            _ => return None,
        })
    }
    pub fn sz(self) -> usize {
        usize::from(self != Ty::U8) * 3 + 1
    }
    pub fn c(self) -> &'static str {
        match self {
            Ty::F32 => "float",
            Ty::U32 => "uint32_t",
            Ty::I32 => "int32_t",
            Ty::U8 => "uint8_t",
        }
    }
    pub fn msl(self) -> &'static str {
        match self {
            Ty::F32 => "float",
            Ty::U32 => "uint",
            Ty::I32 => "int",
            Ty::U8 => "uchar",
        }
    }
    pub fn inty(self) -> bool {
        self != Ty::F32
    }
}

pub struct Spec {
    pub src: String,                    // user source, comments stripped
    pub sname: String,                  // state struct name
    pub fields: Vec<(String, Ty, u32)>, // (name, type, init bits)
    pub x: Ty,
    pub y: Ty,
}

pub fn strip_comments(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
        } else {
            o.push(b[i] as char);
            i += 1;
        }
    }
    o
}

pub fn lit_bits(ty: Ty, lit: &str) -> Result<u32, String> {
    let t = lit.trim().trim_end_matches(['f', 'F', 'u', 'U', 'l', 'L']);
    if ty == Ty::F32 {
        t.parse::<f64>()
            .map(|v| (v as f32).to_bits())
            .map_err(|_| format!("init `{lit}` is not a literal — use literals only"))
    } else if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u32::from_str_radix(h, 16).map_err(|_| format!("bad hex init `{lit}`"))
    } else {
        t.parse::<i64>()
            .map(|v| v as u32)
            .map_err(|_| format!("init `{lit}` is not a literal — use literals only"))
    }
}

pub fn parse_spec(src: &str) -> Result<Spec, String> {
    let s = strip_comments(src);
    // typedef struct { TYPE f [= lit]; ... } S;
    let i = s
        .find("typedef struct")
        .ok_or("need `typedef struct { ... } S;` (32-bit fields)")?;
    let j = s[i..].find('{').map(|k| i + k).ok_or("struct body")?;
    let k = s[j..].find('}').map(|k| j + k).ok_or("struct body")?;
    let sname = s[k + 1..]
        .split(';')
        .next()
        .ok_or("struct name")?
        .trim()
        .to_string();
    if sname.is_empty() {
        return Err("struct name missing".into());
    }
    let mut fields = vec![];
    for decl in s[j + 1..k].split(';') {
        let d = decl.trim();
        if d.is_empty() {
            continue;
        }
        // `float y1, x1, env = 0.5f` — comma-separated, type carries over
        let mut ty: Option<Ty> = None;
        for item in d.split(',') {
            let (lhs, init) = item
                .trim()
                .split_once('=')
                .map(|(a, b)| (a.trim(), b.trim()))
                .unwrap_or((item.trim(), "0"));
            let w: Vec<&str> = lhs.split_whitespace().collect();
            let (name, ity) = match ty {
                Some(t) => (*w.last().ok_or(format!("bad field `{d}`"))?, t),
                None => {
                    let name = w
                        .last()
                        .copied()
                        .filter(|n| n.chars().all(|c| c.is_alphanumeric() || c == '_'))
                        .ok_or(format!("bad field `{d}`"))?;
                    let t = Ty::parse(&w[..w.len() - 1].join(" "))
                        .ok_or(format!("field `{name}`: unsupported type"))?;
                    ty = Some(t);
                    (name, t)
                }
            };
            fields.push((name.to_string(), ity, lit_bits(ity, init)?));
        }
    }
    if fields.is_empty() || fields.len() > 16 {
        return Err("need 1..=16 state fields".into());
    }
    // find `step(` not preceded by an identifier char (so `mystep` misses)
    let mut p = 0;
    let (y_ty, params) = loop {
        let q = s[p..]
            .find("step")
            .map(|o| p + o)
            .ok_or("need a `step` function")?;
        let prev_ok =
            q == 0 || !s.as_bytes()[q - 1].is_ascii_alphanumeric() && s.as_bytes()[q - 1] != b'_';
        let open = s[q + 4..]
            .find(|c: char| !c.is_whitespace())
            .map(|o| q + 4 + o);
        if prev_ok && open == s[q + 4..].find('(').map(|o| q + 4 + o) {
            // return type = longest type-suffix of the words before `step`
            let seg = s[..q]
                .rsplit(|c| c == ';' || c == '}' || c == '{')
                .next()
                .unwrap_or("")
                .trim();
            let w: Vec<&str> = seg.split_whitespace().collect();
            let mut yt = None;
            for kk in (1..=w.len().min(3)).rev() {
                if let Some(t) = Ty::parse(&w[w.len() - kk..].join(" ")) {
                    yt = Some(t);
                    break;
                }
            }
            let r = s[open.unwrap()..]
                .find(')')
                .map(|o| open.unwrap() + o)
                .ok_or("step: missing `)`")?;
            break (
                yt.ok_or("cannot parse step's return type")?,
                &s[open.unwrap() + 1..r],
            );
        }
        p = q + 4;
    };
    let mut it = params.split(',');
    let p0 = it.next().ok_or("step needs (S *s, X x)")?;
    let p1 = it.next().ok_or("step needs (S *s, X x)")?;
    if !p0.contains('*') {
        return Err(format!("first param must be `{sname} *`"));
    }
    let toks: Vec<&str> = p1.split_whitespace().collect();
    if toks.len() < 2 {
        return Err("step: second param needs `type name`".into());
    }
    let x_ty = Ty::parse(&toks[..toks.len() - 1].join(" "))
        .ok_or(format!("step: unsupported x type `{}`", p1.trim()))?;
    Ok(Spec {
        src: s,
        sname,
        fields,
        x: x_ty,
        y: y_ty,
    })
}

// ------------------------------------------------------------- CPU path --

const PRELUDE: &str = r#"
#include <stdint.h>
#include <string.h>
#include <math.h>
typedef uint32_t u32; typedef int32_t i32; typedef uint8_t u8;
typedef unsigned char uchar;
static inline float u2f(uint32_t u){float f;memcpy(&f,&u,4);return f;}
static inline uint32_t f2u(float f){uint32_t u;memcpy(&u,&f,4);return u;}
"#;

pub fn shim_src(spec: &Spec) -> String {
    let x_in = if spec.x == Ty::F32 {
        "u2f(xb)".to_string()
    } else {
        format!("({})xb", spec.x.c())
    };
    let y_out = if spec.y == Ty::F32 {
        "f2u(y)".to_string()
    } else {
        "(uint32_t)y".to_string()
    };
    format!(
        "{PRELUDE}\n{}\n\
         extern \"C\" __attribute__((visibility(\"default\"),used))\n\
         uint32_t traj_step(uint32_t *w, uint32_t xb){{\n\
         \x20   {sn} *s = ({sn}*)w;\n\
         \x20   {yt} y = step(s, {xin});\n\
         \x20   return {yout};\n\
         }}\n\
         extern \"C\" __attribute__((visibility(\"default\"),used))\n\
         void traj_run(uint32_t *w, const {xt} *xs, {yt} *ys, uint64_t n,\n\
         \x20             uint32_t *ck, uint64_t stride){{\n\
         \x20   {sn} *s = ({sn}*)w;\n\
         \x20   for (uint64_t i = 0; i < n; i++){{\n\
         \x20       ys[i] = step(s, xs[i]);\n\
         \x20       if (stride && (i+1)%stride==0){{\n\
         \x20           {sn} snap = *s; memcpy(ck+((i+1)/stride-1)*{nf}, &snap, {nf}*4);\n\
         \x20       }}\n\
         \x20   }}\n\
         }}\n\
         extern \"C\" __attribute__((visibility(\"default\"),used))\n\
         void traj_init(uint32_t *w){{ {sn} s{{}}; memcpy(w,&s,sizeof s); }}",
        spec.src,
        sn = spec.sname,
        yt = spec.y.c(),
        xt = spec.x.c(),
        xin = x_in,
        yout = y_out,
        nf = spec.fields.len()
    )
}

type RunFn = unsafe extern "C" fn(*mut u32, *const u8, *mut u8, u64, *mut u32, u64);

pub struct Dylib {
    _lib: libloading::Library,
    step: unsafe extern "C" fn(*mut u32, u32) -> u32,
    run: RunFn,
    init: [u32; 16],
}

pub fn build_dylib(spec: &Spec) -> Result<Dylib, String> {
    let dir = env::temp_dir().join(format!("trajrun_{}", std::process::id()));
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let srcp = dir.join("step_all.cc");
    let libp = dir.join("step_all.dylib");
    fs::write(&srcp, shim_src(spec)).map_err(|e| e.to_string())?;
    // c++ driver (not cc -x c++) so the right C++ runtime is linked on every OS
    let out = Command::new("c++")
        .args([
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            "-ffp-contract=off",
            "-o",
        ])
        .arg(&libp)
        .arg(&srcp)
        .output()
        .map_err(|e| format!("cc failed to start: {e} (need Xcode CLT for --cpu)"))?;
    if !out.status.success() {
        return Err(format!(
            "cc failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    unsafe {
        let lib = libloading::Library::new(&libp).map_err(|e| e.to_string())?;
        let step = *lib
            .get::<unsafe extern "C" fn(*mut u32, u32) -> u32>(b"traj_step")
            .map_err(|e| e.to_string())?;
        let initf = *lib
            .get::<unsafe extern "C" fn(*mut u32)>(b"traj_init")
            .map_err(|e| e.to_string())?;
        let run = *lib.get::<RunFn>(b"traj_run").map_err(|e| e.to_string())?;
        let mut init = [0u32; 16];
        initf(init.as_mut_ptr());
        Ok(Dylib {
            _lib: lib,
            step,
            run,
            init,
        })
    }
}

// element access on the raw input/output byte vectors
pub fn xbits(bytes: &[u8], i: usize, xsz: usize) -> u32 {
    if xsz == 1 {
        bytes[i] as u32
    } else {
        u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
    }
}
pub fn yput(ys: &mut [u8], i: usize, v: u32, ysz: usize) {
    if ysz == 1 {
        ys[i] = v as u8;
    } else {
        ys[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// Serial whole-input run through the batch entry point (native speed —
/// the step call is inlined inside the compiled loop, not FFI per element).
/// Serial whole-input run through the batch entry point (native speed —
/// the step call is inlined inside the compiled loop, not FFI per element).
pub fn serial_run(d: &Dylib, spec: &Spec, bytes: &[u8]) -> Vec<u8> {
    let n = bytes.len() / spec.x.sz();
    let mut ys = vec![0u8; n * spec.y.sz()];
    let mut s = d.init;
    unsafe {
        (d.run)(
            s.as_mut_ptr(),
            bytes.as_ptr(),
            ys.as_mut_ptr(),
            n as u64,
            std::ptr::null_mut(),
            0,
        );
    }
    ys
}

// ------------------------------------------- affine Z/2^32 exact scan --
//
// If the step is affine in state over Z/2^32 — f(s,x) = A·s + b(x) — the
// chunk map is (A^L, B_c) and boundary states compose EXACTLY: s*_c =
// A^L·s*_{c-1} + B_{c-1}, with B_c = (chunk end from any known start) -
// A^L·start. No speculation, no merging required: works for LCGs, pure
// integrators, counters — recurrences that never merge and are otherwise
// walls. Detection is exact (integer probes), so false positives just
// fail --verify rather than corrupt output.

pub type Mat = Vec<Vec<u32>>; // nf x nf over Z/2^32 (wrapping arithmetic)

pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    let n = a.len();
    let mut r = vec![vec![0u32; n]; n];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                r[i][k] = r[i][k].wrapping_add(a[i][j].wrapping_mul(b[j][k]));
            }
        }
    }
    r
}
pub fn matvec(a: &Mat, v: &[u32]) -> Vec<u32> {
    a.iter()
        .map(|row| {
            row.iter()
                .zip(v)
                .fold(0u32, |s, (a, b)| s.wrapping_add(a.wrapping_mul(*b)))
        })
        .collect()
}
pub fn matpow(a: &Mat, mut e: u64) -> Mat {
    let n = a.len();
    let mut r: Mat = (0..n)
        .map(|i| (0..n).map(|j| u32::from(i == j)).collect())
        .collect();
    let mut b = a.clone();
    while e > 0 {
        if e & 1 == 1 {
            r = matmul(&r, &b);
        }
        e >>= 1;
        if e > 0 {
            b = matmul(&b, &b);
        }
    }
    r
}

/// Probe whether step(s,x) is affine in s over Z/2^32; returns A.
/// Integer-only fields (float affine is never bit-exact anyway).
pub fn detect_affine(d: &Dylib, spec: &Spec, bytes: &[u8]) -> Option<Mat> {
    let (xsz, nf) = (spec.x.sz(), spec.fields.len());
    if nf == 0 || nf > 8 || !spec.fields.iter().all(|f| f.1.inty()) {
        return None;
    }
    let n = bytes.len() / xsz;
    let st = |s: &mut [u32; 16], x: u32| unsafe { (d.step)(s.as_mut_ptr(), x) };
    let mut a_opt: Option<Mat> = None;
    let zero = [0u32; 16];
    // columns of A from unit basis states; must be identical for all x
    for i in 0..8 {
        let x = if n == 0 {
            0
        } else {
            xbits(bytes, i * n / 8, xsz)
        };
        let mut v0 = zero;
        st(&mut v0, x);
        let mut cols = vec![vec![0u32; nf]; nf]; // cols[j][field]
        for j in 0..nf {
            let mut e = zero;
            e[j] = 1;
            st(&mut e, x);
            for f2 in 0..nf {
                cols[j][f2] = e[f2].wrapping_sub(v0[f2]);
            }
        }
        let a: Mat = (0..nf)
            .map(|f2| (0..nf).map(|j| cols[j][f2]).collect())
            .collect();
        match &a_opt {
            None => a_opt = Some(a),
            Some(prev) => {
                if *prev != a {
                    return None;
                }
            }
        }
    }
    let a = a_opt.unwrap();
    // verify linearity on random states (exact — integer ring)
    let mut rng = 0x9e3779b9u32;
    let mut nrng = move || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        rng
    };
    for i in 0..16 {
        let x = if n == 0 {
            0
        } else {
            xbits(bytes, (nrng() as usize) % n, xsz)
        };
        let _ = i;
        let mut v0 = zero;
        st(&mut v0, x);
        let mut s = [0u32; 16];
        for j in 0..nf {
            s[j] = nrng();
        }
        let mut got = s;
        st(&mut got, x);
        let av = matvec(&a, &s[..nf]);
        for f2 in 0..nf {
            if got[f2] != v0[f2].wrapping_add(av[f2]) {
                return None;
            }
        }
    }
    Some(a)
}

/// Exact parallel run for affine steps: two native passes, no fixup —
/// pass 1 gets each chunk's constant term, a serial scan of `nch`
/// boundaries recovers every true start state, pass 2 emits outputs.
pub fn run_affine(
    d: &Dylib,
    spec: &Spec,
    bytes: &[u8],
    threads: usize,
    verify: bool,
    a: &Mat,
) -> Result<(Vec<u8>, String), String> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (xsz, ysz, nf) = (spec.x.sz(), spec.y.sz(), spec.fields.len());
    let n = bytes.len() / xsz;
    let chunks = (4 * threads).max(1);
    let len = n.div_ceil(chunks);
    let nch = n.div_ceil(len);
    let al = matpow(a, len as u64);
    let init: Vec<u32> = d.init[..nf].to_vec();
    let mut ys = vec![0u8; n * ysz];
    let mut ends = vec![[0u32; 16]; nch];
    let mut starts = vec![[0u32; 16]; nch];
    let pass = |starts: &[[u32; 16]], ends: &mut Vec<[u32; 16]>, ys: &mut Vec<u8>| {
        let yp: Vec<usize> = ys
            .chunks_mut(len * ysz)
            .map(|c| c.as_mut_ptr() as usize)
            .collect();
        let ep = ends.as_mut_ptr() as usize;
        let bp = bytes.as_ptr() as usize;
        let sp = starts.as_ptr() as usize;
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..threads.min(nch) {
                sc.spawn(|| loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nch {
                        break;
                    }
                    let clen = len.min(n - c * len);
                    let mut s = unsafe { *(sp as *const [u32; 16]).add(c) };
                    unsafe {
                        (d.run)(
                            s.as_mut_ptr(),
                            (bp + c * len * xsz) as *const u8,
                            yp[c] as *mut u8,
                            clen as u64,
                            std::ptr::null_mut(),
                            0,
                        );
                        *(ep as *mut [u32; 16]).add(c) = s;
                    }
                });
            }
        });
    };
    pass(
        &vec![
            {
                let mut v = [0u32; 16];
                v[..nf].copy_from_slice(&init);
                v
            };
            nch
        ],
        &mut ends,
        &mut ys,
    );
    // B_c = ends[c] - A^L·init; s*_{c+1} = A^L·s*_c + B_c
    let mut s = init.clone();
    for c in 0..nch - 1 {
        starts[c][..nf].copy_from_slice(&s);
        let bc: Vec<u32> = (0..nf)
            .map(|f2| ends[c][f2].wrapping_sub(matvec(&al, &init)[f2]))
            .collect();
        let av = matvec(&al, &s);
        for f2 in 0..nf {
            s[f2] = av[f2].wrapping_add(bc[f2]);
        }
    }
    starts[nch - 1][..nf].copy_from_slice(&s);
    pass(&starts, &mut vec![[0u32; 16]; nch], &mut ys);
    if verify && serial_run(d, spec, bytes) != ys {
        return Err("affine output != serial — BUG".into());
    }
    Ok((ys, format!("affine chunks={nch} (exact Z/2^32 scan)")))
}

/// Speculative multicore run — same algorithm as trajpar::run_speculative,
/// but phase 1 calls the batch `traj_run` once per chunk (native loop
/// speed); only the sequential fix-up pays the per-element FFI call, and
/// only for merge distances.
pub fn run_cpu(
    d: &Dylib,
    spec: &Spec,
    bytes: &[u8],
    threads: usize,
    verify: bool,
) -> Result<(Vec<u8>, String), String> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (xsz, ysz, nf) = (spec.x.sz(), spec.y.sz(), spec.fields.len());
    if bytes.len() % xsz != 0 {
        return Err("input size must be a multiple of the element size".into());
    }
    let n = bytes.len() / xsz;
    if n == 0 {
        return Ok((vec![], "empty".into()));
    }
    let m = 64usize;
    let f = |s: &mut [u32; 16], x: u32| unsafe { (d.step)(s.as_mut_ptr(), x) };

    // probe: does a realistic state merge with the init state?
    let p = (1usize << 16).min(n / 4);
    let mut speculative = false;
    if threads > 1 && p > 0 {
        let mut a = d.init;
        let mut scratch = vec![0u8; p * ysz];
        unsafe {
            (d.run)(
                a.as_mut_ptr(),
                bytes.as_ptr(),
                scratch.as_mut_ptr(),
                p as u64,
                std::ptr::null_mut(),
                0,
            );
        }
        let mut b = d.init;
        let off = n / 2;
        for j in 0..p {
            let x = xbits(bytes, off + j, xsz);
            f(&mut a, x);
            f(&mut b, x);
            if a[..nf] == b[..nf] {
                speculative = true;
                break;
            }
        }
    }
    if !speculative {
        if let Some(a) = detect_affine(d, spec, bytes) {
            return run_affine(d, spec, bytes, threads, verify, &a);
        }
        let ys = serial_run(d, spec, bytes);
        let m = if threads <= 1 || p == 0 {
            "serial"
        } else {
            "serial(wall)"
        };
        return Ok((ys, m.into()));
    }

    // phase 1: every chunk runs from init in parallel, checkpoints every m
    let chunks = (4 * threads).max(1);
    let len = n.div_ceil(chunks);
    let nck = len / m;
    let nch = n.div_ceil(len);
    let mut ys = vec![0u8; n * ysz];
    let mut cks = vec![vec![0u32; nck.max(1) * nf]; nch];
    let mut ends = vec![[0u32; 16]; nch];
    {
        let yp: Vec<usize> = ys
            .chunks_mut(len * ysz)
            .map(|c| c.as_mut_ptr() as usize)
            .collect();
        let cp: Vec<usize> = cks.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
        let ep = ends.as_mut_ptr() as usize;
        let bp = bytes.as_ptr() as usize;
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..threads.min(nch) {
                sc.spawn(|| loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nch {
                        break;
                    }
                    let clen = len.min(n - c * len);
                    let mut s = d.init;
                    unsafe {
                        (d.run)(
                            s.as_mut_ptr(),
                            (bp + c * len * xsz) as *const u8,
                            yp[c] as *mut u8,
                            clen as u64,
                            cp[c] as *mut u32,
                            m as u64,
                        );
                        *(ep as *mut [u32; 16]).add(c) = s;
                    }
                });
            }
        });
    }
    // phase 2 (sequential): true start of chunk c = true end of chunk c-1;
    // rewrite the output prefix until the state equals an old checkpoint
    let mut s = ends[0];
    let (mut merged, mut full, mut fix) = (0usize, 0usize, 0usize);
    for c in 1..nch {
        if s[..nf] == d.init[..nf] {
            s = ends[c];
            continue;
        }
        let clen = len.min(n - c * len);
        let mut hit = false;
        for j in 0..clen {
            let y = f(&mut s, xbits(bytes, c * len + j, xsz));
            yput(&mut ys, c * len + j, y, ysz);
            fix += 1;
            if (j + 1) % m == 0 && s[..nf] == cks[c][((j + 1) / m - 1) * nf..((j + 1) / m) * nf] {
                hit = true;
                break;
            }
        }
        if hit {
            merged += 1;
            s = ends[c];
        } else {
            full += 1;
        }
    }
    if verify {
        if serial_run(d, spec, bytes) != ys {
            return Err("parallel output != serial — BUG".into());
        }
    }
    Ok((
        ys,
        format!("speculative chunks={nch} merged={merged} reruns={full} fix={fix}"),
    ))
}

// ------------------------------------------------------------- GPU path --

pub fn msl_src(spec: &Spec, l: usize, m: usize, c: usize) -> String {
    let nf = spec.fields.len();
    // strip #includes (MSL has no filesystem; we emit our own prelude)
    let mut user: String = spec
        .src
        .lines()
        .filter(|ln| !ln.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    // MSL pointers need an address space: state lives in thread registers,
    // so `S *` becomes `thread S *` (identifier-boundary aware).
    let mut u2 = String::with_capacity(user.len());
    let mut it = user.match_indices(&spec.sname).peekable();
    let mut last = 0;
    while let Some((o, _)) = it.next() {
        let b = user.as_bytes();
        let prev_ok = o == 0 || !(b[o - 1].is_ascii_alphanumeric() || b[o - 1] == b'_');
        let mut k = o + spec.sname.len();
        let next_ok = k >= b.len() || !(b[k].is_ascii_alphanumeric() || b[k] == b'_');
        while k < b.len() && b[k] == b' ' {
            k += 1;
        }
        if prev_ok && next_ok && k < b.len() && b[k] == b'*' {
            u2.push_str(&user[last..o]);
            u2.push_str("thread ");
            u2.push_str(&spec.sname);
            last = o + spec.sname.len();
        }
    }
    u2.push_str(&user[last..]);
    user = u2;
    for (a, b) in [
        ("fabsf", "fabs"),
        ("fminf", "fmin"),
        ("fmaxf", "fmax"),
        ("sqrtf", "sqrt"),
        ("expf", "exp"),
        ("logf", "log"),
        ("powf", "pow"),
        ("sinf", "sin"),
        ("cosf", "cos"),
        ("floorf", "floor"),
        ("roundf", "round"),
        ("fmaf", "fma"),
    ] {
        user = user.replace(&format!("{a}("), &format!("{b}("));
    }
    let b2t = |ty: Ty, e: String| {
        if ty == Ty::F32 {
            format!("as_type<float>({e})")
        } else {
            format!("({})({e})", ty.msl())
        }
    };
    let t2b = |ty: Ty, e: String| {
        if ty == Ty::F32 {
            format!("as_type<uint>({e})")
        } else {
            format!("(uint)({e})")
        }
    };
    let f = |i: usize| spec.fields[i].clone();
    let load = (0..nf)
        .map(|i| format!("s.{} = {};", f(i).0, b2t(f(i).1, format!("s_in[{i}u*C+c]"))))
        .collect::<String>();
    let same = (0..nf)
        .map(|i| format!("s_in[{i}u*C+c]==s_prev[{i}u*C+c]"))
        .collect::<Vec<_>>()
        .join(" && ");
    let eq = (0..nf)
        .map(|i| format!("ck_in[o+{i}u*C]=={}", t2b(f(i).1, format!("s.{}", f(i).0))))
        .collect::<Vec<_>>()
        .join(" && ");
    let rec = (0..nf)
        .map(|i| format!("ck[o+{i}u*C]={};", t2b(f(i).1, format!("s.{}", f(i).0))))
        .collect::<String>();
    let store = (0..nf)
        .map(|i| format!("s_out[{i}u*C+c]={};", t2b(f(i).1, format!("s.{}", f(i).0))))
        .collect::<String>();
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
typedef uint32_t u32; typedef int32_t i32; typedef uint8_t u8;
typedef unsigned int uint; // uchar/u8 already exist
{user}
#define L {l}u
#define M {m}u
#define NF {nf}u
#define C {cc}u
kernel void gpustep(device const {xe} *xs, device {ye} *ys,
        device const uint *params, device const uint *s_in,
        device const uint *s_prev, device const uint *ck_in,
        device uint *s_out, device uint *ck, device uint *merge,
        uint c [[thread_position_in_grid]]) {{
  uint mode = params[0];
  {sn} s = {{}};
  if (mode != 0u) {{ {load} }}
  uint mq = L;
  if (mode != 0u && ({same})) mq = 0u;
  for (uint j = 0; j < mq; ++j) {{
    {xe} x = xs[j * C + c];
    {ye} y = step(&s, x);
    ys[j * C + c] = y;
    if (((j + 1u) % M) == 0u) {{
      uint o = ((j + 1u) / M - 1u) * NF * C + c;
      if (mode != 0u && ({eq})) {{ mq = j + 1u; break; }}
      {rec}
    }}
  }}
  merge[c] = mq;
  {store}
}}
kernel void tr_in(device const {xe} *xs, device {xe} *xi,
        uint t [[thread_position_in_grid]]) {{
  uint j = t / C, c = t % C;
  xi[t] = xs[c * L + j];
}}
kernel void tr_out(device const {ye} *yi, device {ye} *ys,
        uint t [[thread_position_in_grid]]) {{
  uint j = t / C, c = t % C;
  ys[c * L + j] = yi[t];
}}
kernel void splice_y(device const {ye} *y2, device {ye} *ys,
        device const uint *mq,
        uint t [[thread_position_in_grid]]) {{
  uint j = t / C, c = t % C;
  if (j < mq[c]) ys[t] = y2[t];
}}
kernel void splice_ck(device const uint *ck2, device uint *ck,
        device const uint *cut,
        uint t [[thread_position_in_grid]]) {{
  uint c = t % C, q = t / (NF * C);
  if (q < cut[c]) ck[t] = ck2[t];
}}"#,
        xe = spec.x.msl(),
        ye = spec.y.msl(),
        sn = spec.sname,
        cc = c,
    )
}

#[cfg(target_os = "macos")]
pub mod gpu {
    use super::Spec;
    use metal::*;
    use std::time::Instant;

    pub struct Engine {
        dev: Device,
        q: CommandQueue,
        k_step: ComputePipelineState,
        k_in: ComputePipelineState,
        k_out: ComputePipelineState,
        k_sy: ComputePipelineState,
        k_sck: ComputePipelineState,
        pub nf: usize,
        pub c: usize,
        pub l: usize,
        m: usize,
        init: Vec<u32>,
    }

    impl Engine {
        pub fn new(spec: &Spec, n_elems: usize, lanes: usize) -> Result<Engine, String> {
            let m = 64usize;
            let c = lanes.min(n_elems.div_ceil(4 * m).max(1));
            let l = n_elems.div_ceil(c).div_ceil(m) * m;
            let dev = Device::system_default().ok_or("no Metal device")?;
            let opts = CompileOptions::new();
            opts.set_fast_math_enabled(false); // strict IEEE: match the dylib
            let src = super::msl_src(spec, l, m, c);
            let lib = dev
                .new_library_with_source(&src, &opts)
                .map_err(|e| format!("metal compile: {e}"))?;
            let get = |name: &str| {
                dev.new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
                    .unwrap()
            };
            Ok(Engine {
                q: dev.new_command_queue(),
                k_step: get("gpustep"),
                k_in: get("tr_in"),
                k_out: get("tr_out"),
                k_sy: get("splice_y"),
                k_sck: get("splice_ck"),
                nf: spec.fields.len(),
                c,
                l,
                m,
                dev,
                init: spec.fields.iter().map(|f| f.2).collect(),
            })
        }

        fn buf(&self, bytes: usize) -> Buffer {
            self.dev
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
        }

        fn enc(&self, p: &ComputePipelineState, bufs: &[&Buffer], thr: usize) {
            let cb = self.q.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(p);
            for (i, b) in bufs.iter().enumerate() {
                e.set_buffer(i as u64, Some(*b), 0);
            }
            let tg = thr.min(256);
            e.dispatch_threads(
                MTLSize::new(thr as u64, 1, 1),
                MTLSize::new(tg as u64, 1, 1),
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        fn w32(b: &Buffer, v: &[u32]) {
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.contents() as *mut u32, v.len()) }
        }
        fn r32(b: &Buffer, n: usize) -> Vec<u32> {
            unsafe { std::slice::from_raw_parts(b.contents() as *const u32, n).to_vec() }
        }

        /// xs_bytes: raw input file; returns rounds used, fills ys_bytes.
        /// `aff`: optional Z/2^32-affine matrix of the step — lets us inject
        /// exact boundary states after the first round (one extra round
        /// confirms the fixpoint instead of merging).
        pub fn run(
            &self,
            spec: &Spec,
            xs_bytes: &[u8],
            ys_bytes: &mut Vec<u8>,
            max_rounds: usize,
            aff: Option<&super::Mat>,
        ) -> Result<usize, String> {
            let (c, l, m, nf) = (self.c, self.l, self.m, self.nf);
            let nck = l / m;
            let (xsz, ysz) = (spec.x.sz(), spec.y.sz());
            let xs = self.buf(c * l * xsz);
            let xs_il = self.buf(c * l * xsz);
            let ys = self.buf(c * l * ysz);
            let y2 = self.buf(c * l * ysz);
            let params = self.buf(4);
            let s_in = self.buf(nf * c * 4);
            let s_prev = self.buf(nf * c * 4);
            let ck_in = self.buf(4); // unused in mode 0
            let s_out = self.buf(nf * c * 4);
            let ck = self.buf(nck * nf * c * 4);
            let ck2 = self.buf(nck * nf * c * 4);
            let mq = self.buf(c * 4);
            let cut = self.buf(c * 4);
            unsafe {
                let d = xs.contents() as *mut u8;
                d.copy_from_nonoverlapping(xs_bytes.as_ptr(), xs_bytes.len());
                d.add(xs_bytes.len())
                    .write_bytes(0, c * l * xsz - xs_bytes.len());
            }
            self.enc(&self.k_in, &[&xs, &xs_il], c * l);
            Self::w32(&params, &[0]);
            self.enc(
                &self.k_step,
                &[
                    &xs_il, &ys, &params, &s_in, &s_prev, &ck_in, &s_out, &ck, &mq,
                ],
                c,
            );
            let mut e = Self::r32(&s_out, nf * c);
            let mut s_used: Vec<u32> = self
                .init
                .iter()
                .flat_map(|&v| std::iter::repeat(v).take(c))
                .collect();
            // affine step: exact boundaries s*_c = A^L·s*_{c-1} + B_{c-1}
            // with B_c = e[c] - A^L·init, computed once from the init run.
            let mut scanned: Option<Vec<u32>> = aff.map(|a| {
                let al = super::matpow(a, l as u64);
                let init_al = super::matvec(&al, &self.init);
                let mut s = self.init.clone();
                let mut out = vec![0u32; nf * c];
                for cc in 0..c {
                    for f2 in 0..nf {
                        out[f2 * c + cc] = s[f2];
                    }
                    let av = super::matvec(&al, &s);
                    for f2 in 0..nf {
                        s[f2] = av[f2].wrapping_add(e[f2 * c + cc].wrapping_sub(init_al[f2]));
                    }
                }
                out
            });
            let mut rounds = 1;
            let mut converged = false;
            for _ in 0..max_rounds {
                let s_new = scanned.take().unwrap_or_else(|| {
                    let mut v = vec![0u32; nf * c];
                    for f in 0..nf {
                        v[f * c] = self.init[f];
                        for cc in 1..c {
                            v[f * c + cc] = e[f * c + cc - 1];
                        }
                    }
                    v
                });
                if s_new == s_used {
                    converged = true;
                    break;
                }
                Self::w32(&s_in, &s_new);
                Self::w32(&s_prev, &s_used);
                Self::w32(&params, &[1]);
                let td = std::env::var("TRAJ_DEBUG").is_ok();
                let tt = Instant::now();
                self.enc(
                    &self.k_step,
                    &[&xs_il, &y2, &params, &s_in, &s_prev, &ck, &s_out, &ck2, &mq],
                    c,
                );
                let t_step = tt.elapsed();
                let tt = Instant::now();
                let mqv = Self::r32(&mq, c);
                let out2 = Self::r32(&s_out, nf * c);
                // merged/skipped lanes exited early: their end stays old
                let mut e2 = e.clone();
                for cc in 0..c {
                    if mqv[cc] as usize == l {
                        for f in 0..nf {
                            e2[f * c + cc] = out2[f * c + cc];
                        }
                    }
                }
                let cuts: Vec<u32> = mqv
                    .iter()
                    .map(|&q| {
                        if q as usize == l {
                            nck as u32
                        } else {
                            (q / m as u32).saturating_sub(1)
                        }
                    })
                    .collect();
                Self::w32(&cut, &cuts);
                let cb = self.q.new_command_buffer();
                {
                    let en = cb.new_compute_command_encoder();
                    en.set_compute_pipeline_state(&self.k_sy);
                    en.set_buffer(0, Some(&y2), 0);
                    en.set_buffer(1, Some(&ys), 0);
                    en.set_buffer(2, Some(&mq), 0);
                    en.dispatch_threads(
                        MTLSize::new((c * l) as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    en.end_encoding();
                }
                {
                    let en = cb.new_compute_command_encoder();
                    en.set_compute_pipeline_state(&self.k_sck);
                    en.set_buffer(0, Some(&ck2), 0);
                    en.set_buffer(1, Some(&ck), 0);
                    en.set_buffer(2, Some(&cut), 0);
                    en.dispatch_threads(
                        MTLSize::new((nck * nf * c) as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    en.end_encoding();
                }
                cb.commit();
                cb.wait_until_completed();
                let t_splice = tt.elapsed();
                if td {
                    eprintln!(
                        "    step={:.1}ms splice={:.1}ms",
                        t_step.as_secs_f64() * 1e3,
                        t_splice.as_secs_f64() * 1e3
                    );
                }
                if std::env::var("TRAJ_DEBUG").is_ok() {
                    let (sk, mg, fr) = mqv.iter().fold((0, 0, 0), |(s, m2, f), &q| {
                        if q == 0 {
                            (s + 1, m2, f)
                        } else if (q as usize) < l {
                            (s, m2 + 1, f)
                        } else {
                            (s, m2, f + 1)
                        }
                    });
                    eprintln!(
                        "  round {rounds}: skip={sk} merged={mg} full={fr} moved={}",
                        e2 != e
                    );
                }
                let conv = e2 == e;
                e = e2;
                s_used = s_new;
                rounds += 1;
                if conv {
                    converged = true;
                    break;
                }
            }
            if !converged {
                return Err(format!("no convergence in {rounds} rounds (wall)"));
            }
            self.enc(&self.k_out, &[&ys, &y2], c * l);
            let need = c * l * ysz;
            unsafe {
                ys_bytes.clear();
                ys_bytes.reserve(need);
                ys_bytes.set_len(need);
                std::ptr::copy_nonoverlapping(
                    y2.contents() as *const u8,
                    ys_bytes.as_mut_ptr(),
                    need,
                );
            }
            Ok(rounds)
        }
    }
}

pub fn probe_merge(d: &Dylib, spec: &Spec, bytes: &[u8]) -> String {
    let (xsz, ysz, nf) = (spec.x.sz(), spec.y.sz(), spec.fields.len());
    let n = bytes.len() / xsz;
    let p = (1usize << 16).min(n / 4);
    if p == 0 {
        return "input too small to probe".into();
    }
    let f = |s: &mut [u32; 16], x: u32| unsafe { (d.step)(s.as_mut_ptr(), x) };
    let mut a = d.init;
    let mut scratch = vec![0u8; p * ysz];
    unsafe {
        (d.run)(
            a.as_mut_ptr(),
            bytes.as_ptr(),
            scratch.as_mut_ptr(),
            p as u64,
            std::ptr::null_mut(),
            0,
        );
    }
    let mut b = d.init;
    let off = n / 2;
    let mut steps = 0;
    for j in 0..p {
        let x = xbits(bytes, off + j, xsz);
        f(&mut a, x);
        f(&mut b, x);
        steps += 1;
        if a[..nf] == b[..nf] {
            break;
        }
    }
    if a[..nf] == b[..nf] {
        format!("speculative (init<->real merge after ~{steps} steps)")
    } else {
        "wall (no merge in probe window — or slow; rerun with bigger input)".into()
    }
}
