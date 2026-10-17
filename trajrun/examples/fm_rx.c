// FM receiver chain on interleaved IQ f32 stream (simplified):
// conjugate discriminator -> de-emphasis IIR -> pilot PLL loop filter.
// All loops have |lambda| < 1 -> carried state contracts (merges).
typedef struct {
    uint32_t p;   // 0: x is I-sample, 1: x is Q-sample
    float i0, q0; // previous IQ pair
    float de;     // de-emphasis state
    float ph;     // PLL phase accumulator
    float lpf;    // PLL loop filter
} S;

float step(S *s, float x) {
    if (s->p == 0) { s->i0 = x; s->p = 1; return s->de; }
    s->p = 0;
    // discriminator ~ instantaneous frequency: q_prev*I - i_prev*Q
    volatile float c1 = s->q0 * s->i0;
    volatile float c2 = s->i0 * x;
    float d = c1 - c2;
    s->q0 = x;
    s->de = s->de + 0.18f * (d - s->de);            // ~75us de-emphasis
    float e = s->de * s->ph;                         // phase error vs pilot
    s->lpf = s->lpf + 0.02f * (e - s->lpf);          // loop filter
    volatile float pull = 0.4f * s->lpf;
    s->ph = s->ph * 0.9995f + pull;                  // weak pull-in
    return s->de;
}
