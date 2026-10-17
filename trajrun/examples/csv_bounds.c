// CSV record-boundary detector (RFC 4180, ""-escaped quotes).
// y[t]=1 marks real record ends (newlines OUTSIDE quoted fields).
// This is what cuDF/DuckDB do by hand: speculate quote-state per chunk.
typedef struct { uint32_t st; } S;
// st: 0 field start / 1 unquoted field / 2 inside quotes / 3 quote in quotes

uint8_t step(S *s, uint8_t x) {
    uint8_t y = 0;
    switch (s->st) {
    case 0:
        if (x == '"') s->st = 2;
        else if (x == '\n') y = 1;
        else if (x != ',') s->st = 1;
        break;
    case 1:
        if (x == '\n') { y = 1; s->st = 0; }
        else if (x == ',') s->st = 0;
        break;
    case 2:
        if (x == '"') s->st = 3;
        break;
    case 3:
        if (x == '"') s->st = 2;          // "" -> literal quote, stay quoted
        else if (x == '\n') { y = 1; s->st = 0; }
        else s->st = 1;
        break;
    }
    return y;
}
