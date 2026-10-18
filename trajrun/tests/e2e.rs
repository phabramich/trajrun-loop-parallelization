// End-to-end: compile step.c -> dylib, run CPU auto, compare vs serial.
// Skips silently if `cc` is unavailable (GPU path needs macOS + Metal).
use std::process::Command;

fn have(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn fsm_cpu_bit_exact_vs_reference() {
    if !have("cc") {
        return;
    }
    let bin = env!("CARGO_BIN_EXE_trajrun");
    let dir = std::env::temp_dir().join("trajrun_test");
    std::fs::create_dir_all(&dir).unwrap();
    let step = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/fsm.c");
    // synthetic CSV-ish input, deterministic
    let mut xs = vec![0u8; 1 << 20];
    let alpha = b"abcdefg,,\"\"\\\n";
    let mut r = 0x12345678u64;
    for x in xs.iter_mut() {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        *x = alpha[(r % alpha.len() as u64) as usize];
    }
    let inp = dir.join("in.bin");
    let out = dir.join("out.bin");
    std::fs::write(&inp, &xs).unwrap();
    let o = Command::new(bin)
        .args([step, inp.to_str().unwrap(), out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let got = std::fs::read(&out).unwrap();
    // scalar oracle: same FSM, one thread
    let mut st = 0u8;
    for (i, &x) in xs.iter().enumerate() {
        st = if x == b'\n' {
            0
        } else if st == 0 {
            u8::from(x == b'"')
        } else if st == 1 {
            if x == b'\\' {
                2
            } else {
                u8::from(x != b'"')
            }
        } else {
            1
        };
        assert_eq!(got[i], st, "mismatch at {i}");
    }
}

#[test]
fn lcg_affine_bit_exact_vs_reference() {
    if !have("cc") {
        return;
    }
    let bin = env!("CARGO_BIN_EXE_trajrun");
    let dir = std::env::temp_dir().join("trajrun_test");
    std::fs::create_dir_all(&dir).unwrap();
    let step = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/lcg.c");
    let mut xs = vec![0u8; 1 << 20];
    let mut r = 0xdeadbeefu64;
    for x in xs.iter_mut() {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        *x = r as u8;
    }
    let inp = dir.join("lcg_in.bin");
    let out = dir.join("lcg_out.bin");
    std::fs::write(&inp, &xs).unwrap();
    let o = Command::new(bin)
        .args([step, inp.to_str().unwrap(), out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let got = std::fs::read(&out).unwrap();
    // scalar oracle: same LCG
    let mut s = 0u32;
    for (i, &x) in xs.iter().enumerate() {
        s = s
            .wrapping_mul(1664525)
            .wrapping_add(1013904223)
            .wrapping_add(x as u32);
        let want = &got[i * 4..i * 4 + 4];
        assert_eq!(s.to_le_bytes(), want, "mismatch at {i}");
    }
}
