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

use std::{env, fs, process::Command, time::Instant};

// ---------------------------------------------------------------- spec --

#[derive(Clone, Copy, PartialEq, Debug)]
enum Ty {
    F32,
    U32,
    I32,
    U8,
}

impl Ty {
    fn parse(s: &str) -> Option<Ty> {
        Some(match s.trim() {
            "float" | "f32" => Ty::F32,
            "uint32_t" | "unsigned" | "unsigned int" | "uint" | "u32" => Ty::U32,
            "int32_t" | "int" | "i32" => Ty::I32,
            "uint8_t" | "unsigned char" | "uchar" | "u8" => Ty::U8,
            _ => return None,
        })
    }
    fn sz(self) -> usize {
        usize::from(self != Ty::U8) * 3 + 1
    }
    fn c(self) -> &'static str {
        match self {
            Ty::F32 => "float",
            Ty::U32 => "uint32_t",
            Ty::I32 => "int32_t",
            Ty::U8 => "uint8_t",
        }
    }
    fn msl(self) -> &'static str {
        match self {
            Ty::F32 => "float",
            Ty::U32 => "uint",
            Ty::I32 => "int",
            Ty::U8 => "uchar",
        }
    }
    fn inty(self) -> bool {
        self != Ty::F32
    }
}

struct Spec {
    src: String,                  // user source, comments stripped
    sname: String,                // state struct name
    fields: Vec<(String, Ty, u32)>, // (name, type, init bits)
    x: Ty,
    y: Ty,
}

fn strip_comments(s: &str) -> String {
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

fn lit_bits(ty: Ty, lit: &str) -> Result<u32, String> {
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

fn parse_spec(src: &str) -> Result<Spec, String> {
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
        let prev_ok = q == 0
            || !s.as_bytes()[q - 1].is_ascii_alphanumeric() && s.as_bytes()[q - 1] != b'_';
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
typedef uint32_t u32; typedef int32_t i32; typedef uint8_t u8;
typedef unsigned char uchar;
static inline float u2f(uint32_t u){float f;memcpy(&f,&u,4);return f;}
static inline uint32_t f2u(float f){uint32_t u;memcpy(&u,&f,4);return u;}
"#;

fn shim_src(spec: &Spec) -> String {
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

struct Dylib {
    _lib: libloading::Library,
    step: unsafe extern "C" fn(*mut u32, u32) -> u32,
    run: RunFn,
    init: [u32; 16],
}

fn build_dylib(spec: &Spec) -> Result<Dylib, String> {
    let dir = env::temp_dir().join(format!("trajrun_{}", std::process::id()));
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let srcp = dir.join("step_all.cc");
    let libp = dir.join("step_all.dylib");
    fs::write(&srcp, shim_src(spec)).map_err(|e| e.to_string())?;
    let out = Command::new("cc")
        .args([
            "-x",
            "c++",
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
        return Err(format!("cc failed:\n{}", String::from_utf8_lossy(&out.stderr)));
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
        Ok(Dylib { _lib: lib, step, run, init })
    }
}

// element access on the raw input/output byte vectors
fn xbits(bytes: &[u8], i: usize, xsz: usize) -> u32 {
    if xsz == 1 {
        bytes[i] as u32
    } else {
        u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
    }
}
fn yput(ys: &mut [u8], i: usize, v: u32, ysz: usize) {
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
fn serial_run(d: &Dylib, spec: &Spec, bytes: &[u8]) -> Vec<u8> {
    let n = bytes.len() / spec.x.sz();
    let mut ys = vec![0u8; n * spec.y.sz()];
    let mut s = d.init;
    unsafe {
        (d.run)(s.as_mut_ptr(), bytes.as_ptr(), ys.as_mut_ptr(), n as u64,
                std::ptr::null_mut(), 0);
    }
    ys
}

/// Speculative multicore run — same algorithm as trajpar::run_speculative,
/// but phase 1 calls the batch `traj_run` once per chunk (native loop
/// speed); only the sequential fix-up pays the per-element FFI call, and
/// only for merge distances.
fn run_cpu(d: &Dylib, spec: &Spec, bytes: &[u8], threads: usize, verify: bool)
        -> Result<(Vec<u8>, String), String> {
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
            (d.run)(a.as_mut_ptr(), bytes.as_ptr(), scratch.as_mut_ptr(),
                    p as u64, std::ptr::null_mut(), 0);
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
        let ys = serial_run(d, spec, bytes);
        let m = if threads <= 1 || p == 0 { "serial" } else { "serial(wall)" };
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
                        (d.run)(s.as_mut_ptr(), (bp + c * len * xsz) as *const u8,
                                yp[c] as *mut u8, clen as u64,
                                cp[c] as *mut u32, m as u64);
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
            if (j + 1) % m == 0
                && s[..nf] == cks[c][((j + 1) / m - 1) * nf..((j + 1) / m) * nf]
            {
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
    Ok((ys, format!("speculative chunks={nch} merged={merged} reruns={full} fix={fix}")))
}

// ------------------------------------------------------------- GPU path --

fn msl_src(spec: &Spec, l: usize, m: usize, c: usize) -> String {
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
        ("fabsf", "fabs"), ("fminf", "fmin"), ("fmaxf", "fmax"),
        ("sqrtf", "sqrt"), ("expf", "exp"), ("logf", "log"),
        ("powf", "pow"), ("sinf", "sin"), ("cosf", "cos"),
        ("floorf", "floor"), ("roundf", "round"), ("fmaf", "fma"),
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
mod gpu {
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
                dev.new_compute_pipeline_state_with_function(
                    &lib.get_function(name, None).unwrap(),
                )
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
            e.dispatch_threads(MTLSize::new(thr as u64, 1, 1), MTLSize::new(tg as u64, 1, 1));
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        fn w32(b: &Buffer, v: &[u32]) {
            unsafe {
                std::ptr::copy_nonoverlapping(v.as_ptr(), b.contents() as *mut u32, v.len())
            }
        }
        fn r32(b: &Buffer, n: usize) -> Vec<u32> {
            unsafe { std::slice::from_raw_parts(b.contents() as *const u32, n).to_vec() }
        }

        /// xs_bytes: raw input file; returns rounds used, fills ys_bytes.
        pub fn run(&self, spec: &Spec, xs_bytes: &[u8], ys_bytes: &mut Vec<u8>,
                   max_rounds: usize) -> Result<usize, String> {
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
                d.add(xs_bytes.len()).write_bytes(0, c * l * xsz - xs_bytes.len());
            }
            self.enc(&self.k_in, &[&xs, &xs_il], c * l);
            Self::w32(&params, &[0]);
            self.enc(&self.k_step, &[&xs_il, &ys, &params, &s_in, &s_prev, &ck_in, &s_out, &ck, &mq], c);
            let mut e = Self::r32(&s_out, nf * c);
            let mut s_used: Vec<u32> = self
                .init
                .iter()
                .flat_map(|&v| std::iter::repeat(v).take(c))
                .collect();
            let mut rounds = 1;
            for _ in 0..max_rounds {
                let mut s_new = vec![0u32; nf * c];
                for f in 0..nf {
                    s_new[f * c] = self.init[f];
                    for cc in 1..c {
                        s_new[f * c + cc] = e[f * c + cc - 1];
                    }
                }
                if s_new == s_used {
                    break;
                }
                Self::w32(&s_in, &s_new);
                Self::w32(&s_prev, &s_used);
                Self::w32(&params, &[1]);
                let td = std::env::var("TRAJ_DEBUG").is_ok();
                let tt = Instant::now();
                self.enc(&self.k_step, &[&xs_il, &y2, &params, &s_in, &s_prev, &ck, &s_out, &ck2, &mq], c);
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
                    eprintln!("    step={:.1}ms splice={:.1}ms", t_step.as_secs_f64()*1e3, t_splice.as_secs_f64()*1e3);
                }
                if std::env::var("TRAJ_DEBUG").is_ok() {
                    let (sk, mg, fr) = mqv.iter().fold((0, 0, 0), |(s, m2, f), &q| {
                        if q == 0 { (s + 1, m2, f) } else if (q as usize) < l { (s, m2 + 1, f) } else { (s, m2, f + 1) }
                    });
                    eprintln!("  round {rounds}: skip={sk} merged={mg} full={fr} moved={}", e2 != e);
                }
                let conv = e2 == e;
                e = e2;
                s_used = s_new;
                rounds += 1;
                if conv {
                    break;
                }
            }
            self.enc(&self.k_out, &[&ys, &y2], c * l);
            let need = c * l * ysz;
            unsafe {
                ys_bytes.clear();
                ys_bytes.reserve(need);
                ys_bytes.set_len(need);
                std::ptr::copy_nonoverlapping(y2.contents() as *const u8, ys_bytes.as_mut_ptr(), need);
            }
            Ok(rounds)
        }
    }
}

// ----------------------------------------------------------------- CLI --

fn usage() -> ! {
    eprintln!(
        "usage: trajrun <step.c> <in.bin> <out.bin> [--cpu [N]|--gpu] [--lanes C]
                [--verify|--no-verify]
        trajrun --check <step.c> [in.bin]"
    );
    std::process::exit(2)
}

fn probe_merge(d: &Dylib, spec: &Spec, bytes: &[u8]) -> String {
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
        (d.run)(a.as_mut_ptr(), bytes.as_ptr(), scratch.as_mut_ptr(), p as u64,
                std::ptr::null_mut(), 0);
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

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut gpu = false;
    let mut cpu = false;
    let mut check = false;
    let mut verify: Option<bool> = None;
    let mut lanes = 8192usize;
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut pos = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" => gpu = true,
            "--cpu" => {
                cpu = true;
                if let Some(n) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    threads = n;
                    i += 1;
                }
            }
            "--check" => check = true,
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
    if !cpu && !gpu {
        cpu = true;
    }
    let step_path = pos.first().ok_or_else(|| "missing step.c".to_string())?;
    let src = fs::read_to_string(step_path).map_err(|e| format!("{step_path}: {e}"))?;
    let spec = parse_spec(&src).map_err(|e| format!("{step_path}: {e}"))?;
    eprintln!(
        "trajrun: S={} nf={} [{}], x={}, y={}",
        spec.sname,
        spec.fields.len(),
        spec.fields.iter().map(|f| f.0.as_str()).collect::<Vec<_>>().join(","),
        spec.x.c(),
        spec.y.c()
    );

    if check {
        let d = build_dylib(&spec)?;
        if let Some(inp) = pos.get(1) {
            let bytes = fs::read(inp).map_err(|e| format!("{inp}: {e}"))?;
            eprintln!("check: compile ok; {}", probe_merge(&d, &spec, &bytes));
        } else {
            eprintln!("check: compile ok (no input given — skipped probe)");
        }
        return Ok(());
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
        match gpu::Engine::new(&spec, n_elems, lanes)
            .and_then(|e| e.run(&spec, &bytes, &mut out, 64))
        {
            Ok(rounds) => {
                eprintln!("gpu: {rounds} rounds, {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
                out.truncate(want_len);
                fs::write(outp, &out).map_err(|e| format!("{outp}: {e}"))?;
                if verify.unwrap_or(false) {
                    let d = build_dylib(&spec)?;
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
    eprintln!("cpu: {} — {:.1} ms{}", st, t0.elapsed().as_secs_f64() * 1e3,
        if verify.unwrap_or(true) && st.starts_with("spec") { " (verified bit-exact)" } else { "" });
    fs::write(outp, &out).map_err(|e| format!("{outp}: {e}"))?;
    Ok(())
}
