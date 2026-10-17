// RiskMetrics-style per-tick risk pass:
//   EWMA midprice (fast) + EWMA variance (slow) -> standardized tick z-score.
// This is the shape of real risk-engine filters: sequential carried state,
// both loops contract (|1-a|,|1-b| < 1) -> speculative merge converges.
typedef struct { float mu, var; } S;

float step(S *s, float x) {
    // fast mean: half-life ~35 ticks
    float dev = x - s->mu;
    s->mu = s->mu + 0.02f * dev;
    // slow variance: half-life ~700 ticks (RiskMetrics lambda ~= 0.999)
    float d2 = dev * dev;
    s->var = s->var + 0.001f * (d2 - s->var);
    // z-score output: (x - mu) / sqrt(var); guard var underflow
    float sd = sqrtf(s->var);
    return sd > 1e-9f ? dev / sd : 0.0f;
}
