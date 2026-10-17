//! Speculative multicore execution of a sequential trajectory
//! `s[t+1] = f(s[t], x[t])`, `y[t] = out(s, x[t])` — bit-exact.
//!
//! * `run_speculative` — chunks start from `s0` in parallel, saving a
//!   checkpoint every `ckpt` steps; then a sequential fix-up re-runs each
//!   chunk from its true start and **stops as soon as the state equals a
//!   saved checkpoint** (trajectories merged; the rest is already right).
//!   Work ≈ T + Σ merge distances; on a wall it degrades to ≈ serial.
//! * `run_enum` — small finite domain: each chunk is run from every domain
//!   state in lockstep (lanes collapse once they agree), boundary states
//!   are chained in O(chunks), outputs are written in a final parallel pass.
//! * `run_auto` — merge probe picks speculative / enum / serial.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub trait Step: Sync {
    type S: Copy + PartialEq + Send + Sync;
    type X: Sync;
    type Y: Send;
    fn step(&self, s: &mut Self::S, x: &Self::X) -> Self::Y;
    /// Small finite state domain, if known (enables `run_enum`).
    fn domain(&self) -> Option<Vec<Self::S>> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct Cfg {
    pub threads: usize,
    pub chunks: usize,
    pub ckpt: usize,
    pub probe: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        // 4x oversubscription: dynamic queue evens out P/E-core speed gaps
        Cfg { threads, chunks: 4 * threads, ckpt: 64, probe: 1 << 16 }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub method: &'static str,
    pub chunks: usize,
    pub merged: usize,
    pub full_reruns: usize,
    pub fix_steps: usize,
}

pub fn run_serial<F: Step>(f: &F, mut s: F::S, xs: &[F::X], ys: &mut [F::Y]) -> F::S {
    for (x, y) in xs.iter().zip(ys.iter_mut()) {
        *y = f.step(&mut s, x);
    }
    s
}

/// Work-queue parallel map over owned items (index, item) -> result.
fn par_map<T: Send, R: Send>(threads: usize, items: Vec<T>,
                             work: impl Fn(usize, T) -> R + Sync) -> Vec<R> {
    let n = items.len();
    let inp: Vec<Mutex<Option<T>>> = items.into_iter().map(|t| Mutex::new(Some(t))).collect();
    let out: Vec<Mutex<Option<R>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads.min(n) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let t = inp[i].lock().unwrap().take().unwrap();
                *out[i].lock().unwrap() = Some(work(i, t));
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

fn chunk_len(n: usize, chunks: usize) -> usize {
    n.div_ceil(chunks.max(1)).max(1)
}

pub fn run_speculative<F: Step>(f: &F, s0: F::S, xs: &[F::X], ys: &mut [F::Y],
                                cfg: &Cfg) -> (F::S, Stats) {
    let len = chunk_len(xs.len(), cfg.chunks);
    let k = cfg.ckpt.max(1);
    let xch: Vec<&[F::X]> = xs.chunks(len).collect();
    let ych: Vec<&mut [F::Y]> = ys.chunks_mut(len).collect();
    let mut res = par_map(cfg.threads, ych, |c, y| {
        let mut s = s0;
        let mut ck = Vec::with_capacity(y.len() / k);
        for (j, (x, yo)) in xch[c].iter().zip(y.iter_mut()).enumerate() {
            *yo = f.step(&mut s, x);
            if (j + 1) % k == 0 {
                ck.push(s);
            }
        }
        (y, ck, s)
    });
    let mut st = Stats { method: "speculative", chunks: res.len(), ..Default::default() };
    let mut s = res[0].2;
    for c in 1..res.len() {
        let (y, ck, end) = &mut res[c];
        if s == s0 {
            s = *end; // assumed start was right
            continue;
        }
        let mut merged = false;
        for (j, (x, yo)) in xch[c].iter().zip(y.iter_mut()).enumerate() {
            *yo = f.step(&mut s, x);
            st.fix_steps += 1;
            if (j + 1) % k == 0 && s == ck[(j + 1) / k - 1] {
                merged = true;
                break;
            }
        }
        if merged {
            s = *end;
            st.merged += 1;
        } else {
            st.full_reruns += 1;
        }
    }
    (s, st)
}

pub fn run_enum<F: Step>(f: &F, s0: F::S, dom: &[F::S], xs: &[F::X], ys: &mut [F::Y],
                         cfg: &Cfg) -> (F::S, Stats) {
    let len = chunk_len(xs.len(), cfg.chunks);
    let k = cfg.ckpt.max(1);
    let xch: Vec<&[F::X]> = xs.chunks(len).collect();
    // phase 1: end state of every chunk from every domain start (chunk 0: from s0)
    let maps = par_map(cfg.threads, (0..xch.len()).collect(), |c, _| {
        let starts: &[F::S] = if c == 0 { std::slice::from_ref(&s0) } else { dom };
        // lane 0 records checkpoints; others stop once they hit lane 0's path
        let mut s = starts[0];
        let mut ck = Vec::with_capacity(xch[c].len() / k);
        for (j, x) in xch[c].iter().enumerate() {
            f.step(&mut s, x);
            if (j + 1) % k == 0 {
                ck.push(s);
            }
        }
        let end0 = s;
        let mut ends = vec![end0];
        for &d in &starts[1..] {
            let mut s = d;
            for (j, x) in xch[c].iter().enumerate() {
                f.step(&mut s, x);
                if (j + 1) % k == 0 && s == ck[(j + 1) / k - 1] {
                    s = end0;
                    break;
                }
            }
            ends.push(s);
        }
        ends
    });
    // phase 2: chain boundary states
    let mut st = Stats { method: "enum", chunks: xch.len(), ..Default::default() };
    let mut starts = vec![s0];
    let mut s = maps[0][0];
    for c in 1..xch.len() {
        match dom.iter().position(|d| *d == s) {
            Some(i) => {
                starts.push(s);
                s = maps[c][i];
            }
            None => {
                // domain incomplete: finish the tail serially (still exact)
                let a = c * len;
                st.method = "enum+serial_tail";
                st.full_reruns = xch.len() - c;
                let ych: Vec<&mut [F::Y]> = ys[..a].chunks_mut(len).collect();
                par_map(cfg.threads, ych, |c, y| run_serial(f, starts[c], xch[c], y));
                return (run_serial(f, s, &xs[a..], &mut ys[a..]), st);
            }
        }
    }
    // phase 3: write outputs from exact starts
    let ych: Vec<&mut [F::Y]> = ys.chunks_mut(len).collect();
    par_map(cfg.threads, ych, |c, y| run_serial(f, starts[c], xch[c], y));
    (s, st)
}

/// Do trajectories from `a` and `b` merge within `xs`?
fn merges<F: Step>(f: &F, mut a: F::S, mut b: F::S, xs: &[F::X]) -> bool {
    for x in xs {
        f.step(&mut a, x);
        f.step(&mut b, x);
        if a == b {
            return true;
        }
    }
    false
}

pub fn run_auto<F: Step>(f: &F, s0: F::S, xs: &[F::X], ys: &mut [F::Y],
                         cfg: &Cfg) -> (F::S, Stats) {
    let n = xs.len();
    let p = cfg.probe.min(n / 4);
    if cfg.threads <= 1 || p == 0 {
        return (run_serial(f, s0, xs, ys), Stats { method: "serial", ..Default::default() });
    }
    // alternative starts: a realistic state (s0 run over a prefix) + domain;
    // all must merge with s0 — a single lucky equal donor proves nothing
    let mut real = s0;
    for x in &xs[..p] {
        f.step(&mut real, x);
    }
    let dom = f.domain();
    let mut alts = vec![real];
    alts.extend(dom.iter().flatten().copied());
    let win = &xs[n / 2..n / 2 + p];
    if alts.iter().filter(|a| **a != s0).all(|a| merges(f, s0, *a, win)) {
        return run_speculative(f, s0, xs, ys, cfg);
    }
    if let Some(d) = dom {
        return run_enum(f, s0, &d, xs, ys, cfg);
    }
    (run_serial(f, s0, xs, ys), Stats { method: "serial(wall)", ..Default::default() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quote lexer: '\n' resets (merges); without '\n' — Z2 parity (enum).
    struct Lex;
    impl Step for Lex {
        type S = u8;
        type X = u8;
        type Y = u8;
        fn step(&self, s: &mut u8, &c: &u8) -> u8 {
            *s = match (c, *s) {
                (b'\n', _) => 0,
                (b'"', 0) => 1,
                (b'"', 1) => 0,
                (b'\\', 1) => 2,
                (_, 2) => 1,
                (_, s) => s,
            };
            *s
        }
        fn domain(&self) -> Option<Vec<u8>> {
            Some(vec![0, 1, 2])
        }
    }

    /// Hash chain: never merges, no domain -> serial.
    struct Hash;
    impl Step for Hash {
        type S = u32;
        type X = u8;
        type Y = u8;
        fn step(&self, h: &mut u32, &c: &u8) -> u8 {
            *h = (*h ^ c as u32).wrapping_mul(16777619);
            *h ^= *h >> 13;
            (*h >> 24) as u8
        }
    }

    fn text(n: usize, alpha: &[u8], seed: u64) -> Vec<u8> {
        let mut r = seed;
        (0..n).map(|_| {
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            alpha[(r % alpha.len() as u64) as usize]
        }).collect()
    }

    fn check<F: Step<Y = u8>>(f: &F, s0: F::S, xs: &[F::X]) -> &'static str {
        let mut want = vec![0u8; xs.len()];
        let mut got = vec![0u8; xs.len()];
        let es = run_serial(f, s0, xs, &mut want);
        let cfg = Cfg { threads: 4, chunks: 16, ckpt: 8, probe: 4096 };
        let (gs, st) = run_auto(f, s0, xs, &mut got, &cfg);
        assert!(es == gs && want == got, "{} mismatch", st.method);
        // every forced mode must also be exact
        let mut g2 = vec![0u8; xs.len()];
        assert!(run_speculative(f, s0, xs, &mut g2, &cfg).0 == es && g2 == want);
        if let Some(d) = f.domain() {
            let mut g3 = vec![0u8; xs.len()];
            assert!(run_enum(f, s0, &d, xs, &mut g3, &cfg).0 == es && g3 == want);
        }
        st.method
    }

    #[test]
    fn modes_exact_and_auto_picks_right() {
        let n = 200_000;
        assert_eq!(check(&Lex, 0, &text(n, b"ab,\"\\\n", 1)), "speculative");
        assert_eq!(check(&Lex, 0, &text(n, b"ab\"", 2)), "enum");
        assert_eq!(check(&Hash, 2166136261, &text(n, b"abcdefgh", 3)), "serial(wall)");
    }

    #[test]
    fn enum_incomplete_domain_falls_back_exact() {
        struct Bad;
        impl Step for Bad {
            type S = u8; type X = u8; type Y = u8;
            fn step(&self, s: &mut u8, x: &u8) -> u8 { Lex.step(s, x) }
            fn domain(&self) -> Option<Vec<u8>> { Some(vec![0]) } // misses 1, 2
        }
        let xs = text(100_000, b"ab\"", 4);
        let mut want = vec![0u8; xs.len()];
        let mut got = vec![0u8; xs.len()];
        run_serial(&Bad, 0, &xs, &mut want);
        let cfg = Cfg { threads: 4, chunks: 16, ckpt: 8, probe: 4096 };
        let (_, st) = run_enum(&Bad, 0, &[0], &xs, &mut got, &cfg);
        assert_eq!(st.method, "enum+serial_tail");
        assert!(want == got);
    }
}
