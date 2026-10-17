// CSV quote-state lexer — same FSM as wasm-progs/src/lib.rs fsm_lex.
// State: 0 plain / 1 quoted / 2 escaped. '\n' resets -> merges fast.
typedef struct { uint32_t st; } S;

uint8_t step(S *s, uint8_t x) {
    uint32_t st = s->st;
    if (x == 10) st = 0;
    else if (st == 0) st = (x == 34) ? 1 : 0;
    else if (st == 1) st = (x == 92) ? 2 : (x == 34) ? 0 : 1;
    else st = 1;
    s->st = st;
    return (uint8_t)st;
}
