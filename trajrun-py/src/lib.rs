use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn err(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
}

fn threads_or_all(t: usize) -> usize {
    if t == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        t
    }
}

/// run(step_src, input, threads=0, gpu=False) -> (output_bytes, status)
///
/// `step_src` is the text of a step.c file. `input` is a flat byte array
/// of X elements (u8 bytes or little-endian u32/f32 words).
/// status reports the method actually used: speculative / affine / serial(wall) / gpu.
#[pyfunction]
#[pyo3(signature = (src, input, threads = 0, gpu = false))]
fn run(src: &str, input: &[u8], threads: usize, gpu: bool) -> PyResult<(Vec<u8>, String)> {
    let spec = ::trajrun::parse_spec(src).map_err(err)?;
    let d = ::trajrun::build_dylib(&spec).map_err(err)?;
    #[cfg(target_os = "macos")]
    if gpu {
        let n = input.len() / spec.x.sz();
        let aff = ::trajrun::detect_affine(&d, &spec, input);
        let mut out = Vec::new();
        if let Ok(rounds) = ::trajrun::gpu::Engine::new(&spec, n, 8192)
            .and_then(|e| e.run(&spec, input, &mut out, 64, aff.as_ref()))
        {
            out.truncate(n * spec.y.sz());
            return Ok((out, format!("gpu {rounds} rounds")));
        }
        // fall through to CPU on any GPU failure / non-convergence
    }
    let _ = gpu;
    ::trajrun::run_cpu(&d, &spec, input, threads_or_all(threads), false).map_err(err)
}

/// check(step_src, input) -> human-readable verdict:
/// speculative merge distance, affine-over-Z/2^32 detection, or wall.
#[pyfunction]
fn check(src: &str, input: &[u8]) -> PyResult<String> {
    let spec = ::trajrun::parse_spec(src).map_err(err)?;
    let d = ::trajrun::build_dylib(&spec).map_err(err)?;
    let mut s = ::trajrun::probe_merge(&d, &spec, input);
    if ::trajrun::detect_affine(&d, &spec, input).is_some() {
        s.push_str("; affine over Z/2^32 (exact scan, no speculation needed)");
    }
    Ok(s)
}

#[pymodule]
fn trajrun(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(check, m)?)?;
    Ok(())
}
