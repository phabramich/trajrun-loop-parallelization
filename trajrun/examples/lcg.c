// Linear congruential generator — the textbook "sequential" recurrence.
// Never merges (chaotic-looking state), but s' = a*s + (c + x) is affine
// over Z/2^32 -> trajrun detects it and runs an EXACT parallel scan.
typedef struct { uint32_t s; } S;

uint32_t step(S *s, uint8_t x) {
    s->s = s->s * 1664525u + 1013904223u + x;
    return s->s;
}
