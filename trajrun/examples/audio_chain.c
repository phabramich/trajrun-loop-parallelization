// Audio DSP chain: highpass 80Hz -> highpass 30Hz -> peak-hold envelope ->
// gain limiter. Same math as wasm-progs audio_chain (ffmpeg -af analog).
// `volatile` on the products feeding adds = same f32 freeze the WASM path
// gets for free; without it, contraction may shift results by an ulp.
typedef struct { float y1, x1, y2, x2, env; } S;

float step(S *s, float x) {
    volatile float t1 = 0.9886f * s->y1;
    float a = x - s->x1; a = a + t1; s->x1 = x; s->y1 = a;
    volatile float t2 = 0.9957f * s->y2;
    float b = a - s->x2; b = b + t2; s->x2 = a; s->y2 = b;
    float m = b < 0.0f ? -b : b;
    volatile float d = s->env * 0.995f;
    s->env = m > s->env ? m : d;
    float g = s->env > 0.5f ? 0.5f / s->env : 1.0f;
    return b * g;
}
