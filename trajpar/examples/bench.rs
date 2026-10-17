//! Same workloads as wasm-progs/src/lib.rs, as `Step`s. Serial vs run_auto,
//! median of 5, bit-exact check on outputs and final state.
use std::time::Instant;
use trajpar::*;

static STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
    253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
    1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
    3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442,
    11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767,
];
static INDEX_TABLE: [i32; 8] = [-1, -1, -1, -1, 2, 4, 6, 8];

struct FsmLex;
impl Step for FsmLex {
    type S = u8; type X = u8; type Y = u8;
    #[inline(always)]
    fn step(&self, st: &mut u8, &c: &u8) -> u8 {
        let (q, b, nl) = (c == 34, c == 92, c == 10);
        *st = if nl { 0 } else {
            match *st { 0 => q as u8, 1 => if b { 2 } else if q { 0 } else { 1 }, _ => 1 }
        };
        *st
    }
    fn domain(&self) -> Option<Vec<u8>> { Some(vec![0, 1, 2]) }
}

struct SatAcc;
impl Step for SatAcc {
    type S = i32; type X = i16; type Y = i32;
    #[inline(always)]
    fn step(&self, acc: &mut i32, &x: &i16) -> i32 {
        *acc = (*acc + x as i32).clamp(-1_000_000, 1_000_000);
        *acc
    }
}

#[inline(always)]
fn adpcm_diff(nib: i32, step: i32) -> i32 {
    let mut diff = step >> 3;
    diff += ((nib & 4) >> 2) * step;
    diff += ((nib & 2) >> 1) * (step >> 1);
    diff += (nib & 1) * (step >> 2);
    let signed = 0i32.wrapping_sub(diff) & 0i32.wrapping_sub((nib >> 3) & 1);
    diff + signed * 2
}

struct AdpcmLeaky;
impl Step for AdpcmLeaky {
    type S = (i32, i32); type X = u8; type Y = i16;
    #[inline(always)]
    fn step(&self, (pred, index): &mut (i32, i32), &nib: &u8) -> i16 {
        let nib = nib as i32;
        *index = (((*index * 15) >> 4) + INDEX_TABLE[(nib & 7) as usize] + 2).clamp(0, 88);
        let d = adpcm_diff(nib, STEP_TABLE[*index as usize]);
        *pred = (((*pred * 15) >> 4) + d).clamp(-32768, 32767);
        *pred as i16
    }
}

struct AdpcmIma;
impl Step for AdpcmIma {
    type S = (i32, i32); type X = u8; type Y = i16;
    #[inline(always)]
    fn step(&self, (pred, index): &mut (i32, i32), &nib: &u8) -> i16 {
        let nib = nib as i32;
        *pred = (*pred + adpcm_diff(nib, STEP_TABLE[*index as usize])).clamp(-32768, 32767);
        *index = (*index + INDEX_TABLE[(nib & 7) as usize]).clamp(0, 88);
        *pred as i16
    }
}

/// HP80 -> HP30 -> peak envelope -> gain; state kept as f32 *bits* so the
/// merge test is bit-exact (f32 PartialEq says -0.0 == 0.0).
struct AudioChain;
impl Step for AudioChain {
    type S = [u32; 5]; type X = f32; type Y = f32;
    #[inline(always)]
    fn step(&self, s: &mut [u32; 5], &x: &f32) -> f32 {
        let [y1, x1, y2, x2, env] = s.map(f32::from_bits);
        let a = x - x1 + 0.9886 * y1;
        let b = a - x2 + 0.9957 * y2;
        let m = b.abs();
        let env = if m > env { m } else { env * 0.995 };
        let g = if env > 0.5 { 0.5 / env } else { 1.0 };
        *s = [a, x, b, a, env].map(f32::to_bits);
        b * g
    }
}

struct HashChain;
impl Step for HashChain {
    type S = u32; type X = u8; type Y = u8;
    #[inline(always)]
    fn step(&self, h: &mut u32, &c: &u8) -> u8 {
        *h ^= c as u32;
        *h = h.wrapping_mul(16777619);
        *h ^= *h >> 13;
        (*h >> 24) as u8
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn pick(&mut self, a: &[u8], n: usize) -> Vec<u8> { (0..n).map(|_| a[(self.next() % a.len() as u64) as usize]).collect() }
}

fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn bench<F: Step>(name: &str, f: &F, s0: F::S, xs: &[F::X], threads: &[usize])
where F::Y: Default + Clone + PartialEq {
    let mut want = vec![F::Y::default(); xs.len()];
    let mut ts = vec![];
    let mut end = s0;
    for _ in 0..5 {
        let t = Instant::now();
        end = run_serial(f, s0, xs, &mut want);
        ts.push(t.elapsed().as_secs_f64());
    }
    let t_ser = median(ts);
    print!("{name:<26} serial {:8.2} ms", t_ser * 1e3);
    for &th in threads {
        let cfg = Cfg { threads: th, chunks: 4 * th, ..Cfg::default() };
        let mut got = vec![F::Y::default(); xs.len()];
        let mut ts = vec![];
        let mut st = Stats::default();
        let mut ok = true;
        for _ in 0..5 {
            got.iter_mut().for_each(|y| *y = F::Y::default());
            let t = Instant::now();
            let (e, s) = run_auto(f, s0, xs, &mut got, &cfg);
            ts.push(t.elapsed().as_secs_f64());
            ok &= e == end && got == want;
            st = s;
        }
        let t = median(ts);
        print!(" | {th}T {:7.2} ms {:5.2}x {}{}", t * 1e3, t_ser / t, st.method,
               if ok { "" } else { " MISMATCH!" });
        if th == *threads.last().unwrap() {
            print!(" (fix {} steps, {} full reruns)", st.fix_steps, st.full_reruns);
        }
    }
    println!();
}

fn main() {
    let th = [4, 8];
    let mut r = Rng(0x9E3779B97F4A7C15);
    let n = 32 << 20;
    // CSV-like: newlines reset the lexer every ~40 bytes
    let csv = r.pick(b"abcdefg,,\"\"\\\n", n);
    bench("fsm_lex CSV 32MB", &FsmLex, 0, &csv, &th);
    let z2 = r.pick(b"abcdefg\"", n);
    bench("fsm_lex quote-only 32MB", &FsmLex, 0, &z2, &th);
    let sat: Vec<i16> = (0..16 << 20).map(|_| (3000 + (r.next() % 9001) as i32 - 6000) as i16).collect();
    bench("sat_acc 16M i16", &SatAcc, 0, &sat, &th);
    let nibs: Vec<u8> = (0..32 << 20).map(|_| (r.next() & 15) as u8).collect();
    bench("adpcm_leaky 32M nibbles", &AdpcmLeaky, (0, 0), &nibs, &th);
    // random nibbles keep slamming pred into ±32768 clamps -> trajectories merge
    bench("adpcm IMA random nibbles", &AdpcmIma, (0, 0), &nibs, &th);
    // real encoded audio never clamps -> pred offset never heals: true wall
    let mut st = (0, 0);
    let ima: Vec<u8> = (0..32 << 20).map(|i| {
        let x = 8000.0 * (i as f32 * 0.0123).sin() + ((r.next() % 1001) as f32 - 500.0);
        let mut diff = x as i32 - st.0;
        let mut nib = if diff < 0 { diff = -diff; 8 } else { 0 };
        let mut step = STEP_TABLE[st.1 as usize];
        for bit in [4, 2, 1] {
            if diff >= step { nib |= bit; diff -= step; }
            step >>= 1;
        }
        AdpcmIma.step(&mut st, &nib);
        nib
    }).collect();
    bench("adpcm IMA encoded audio", &AdpcmIma, (0, 0), &ima, &th);
    let audio: Vec<f32> = (0..16 << 20).map(|i| {
        let noise = (r.next() % 2001) as f32 / 1000.0 - 1.0;
        (0.6 * (i as f32 * 0.0157).sin() + 0.05 * noise).clamp(-1.0, 1.0)
    }).collect();
    bench("audio_chain 16M f32", &AudioChain, [0; 5], &audio, &th);
    let bytes = r.pick(b"abcdefghijklmnop", n);
    bench("hash_chain 32MB (wall)", &HashChain, 2166136261, &bytes, &th);
}
