// UTF-8 structural validity per byte (no overlong/surrogate checks).
// Sticky "file is valid" flag is NOT carried in state: y[t]=0 marks
// bad bytes, the global verdict is AND over y (associative, free).
// Carrying a monotonic flag would make wrong-start states never merge.
typedef struct { uint32_t cont; } S;

uint8_t step(S *s, uint8_t x) {
    uint8_t ok = 1;
    if (s->cont == 0) {
        if (x < 0x80) {}
        else if ((x & 0xE0) == 0xC0) s->cont = 1;
        else if ((x & 0xF0) == 0xE0) s->cont = 2;
        else if ((x & 0xF8) == 0xF0) s->cont = 3;
        else ok = 0;
    } else {
        if ((x & 0xC0) == 0x80) s->cont--;
        else { ok = 0; s->cont = 0; }
    }
    return ok;
}
