// NDJSON record-boundary detector: y=1 at newlines outside string context.
// This is what ClickHouse/DuckDB parallel input pipelines hand-implement
// per format — here it's 15 lines of C.
typedef struct { uint32_t st; } S;
// st: 0 normal / 1 inside string / 2 after backslash inside string

uint8_t step(S *s, uint8_t x) {
    uint8_t y = 0;
    switch (s->st) {
    case 0:
        if (x == '"') s->st = 1;
        else if (x == '\n') y = 1;
        break;
    case 1:
        if (x == '\\') s->st = 2;
        else if (x == '"') s->st = 0;
        break;
    case 2:
        s->st = 1; // any escaped char consumed, still inside string
        break;
    }
    return y;
}
