// Nested accumulators: s += x; c += s  (sum of prefix sums).
// Two carried integrators — never merges, affine over Z/2^32 -> exact scan.
typedef struct { uint32_t s, c; } S;

uint32_t step(S *s, uint32_t x) {
    s->s += x;
    s->c += s->s;
    return s->c;
}
