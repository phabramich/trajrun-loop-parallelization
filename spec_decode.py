"""Toy speculative decoding: a learned guesser in the same framework.

Greedy char-level n-gram models: a small draft (order-d) proposes the next
`gamma` tokens; the big target (order-6) verifies them in one pass --
accept the longest correct prefix + 1 bonus token. Same math as
jacobi(): guess the trajectory, verify positions in parallel. The only
difference is that the guesser is learned, and it works because the
trajectory is predictable. This is how every LLM-serving stack gets
~2-3x; here it's ~40 lines of n-grams.

Corpus: this repo's own files (self-contained, ~real char statistics).
True stream = greedy continuation of the target model (temperature-0),
so verification = "does the target argmax agree" = compare with stream.
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

corpus = "".join(p.read_text() for p in Path(__file__).parent.glob("*.py"))
corpus += "".join(p.read_text() for p in Path(__file__).parent.glob("*.md"))


def build(text, order):
    m = defaultdict(Counter)
    for i in range(len(text) - order):
        m[text[i:i + order]][text[i + order]] += 1
    return m


def argmax(m, ctx, fallback):
    c = m.get(ctx)
    return c.most_common(1)[0][0] if c else fallback


ORDER = 6
target = build(corpus, ORDER)
space = sorted(set(corpus))[0]

S = 4000
ctx = corpus[:ORDER]
stream = ctx
for _ in range(S):
    c = argmax(target, ctx, space)
    stream += c
    ctx = (ctx + c)[-ORDER:]


def spec(stream, start, draft, do, gamma):
    """Decode `stream` speculatively. Each round: draft proposes `gamma`
    tokens (cheap, sequential), target verifies them all at once --
    accept the matching prefix, take 1 bonus token, repeat.
    Returns (mean accepted per round, tokens per round)."""
    i, rounds, acc_tot = start, 0, 0
    while i < len(stream):
        dctx = stream[max(0, i - do):i]
        props = []
        for _ in range(gamma):
            c = argmax(draft, dctx, space)
            props.append(c)
            dctx = (dctx + c)[-do:]
        acc = 0
        while acc < len(props) and i + acc < len(stream) \
                and props[acc] == stream[i + acc]:
            acc += 1
        acc_tot += acc
        i += acc + 1
        rounds += 1
    return acc_tot / rounds, (len(stream) - start) / rounds


print(f"corpus {len(corpus)} chars, stream {len(stream) - ORDER} tokens, "
      f"target = order-{ORDER} n-gram (greedy)\n")
print(f"{'draft':>10} {'gamma':>5} {'mean acc':>9} {'tok/round':>9} "
      f"{'speedup':>8}")
for do in (1, 3, 5):
    d = build(corpus, do)
    for g in (4, 8):
        acc, tok = spec(stream, ORDER, d, do, g)
        print(f"order-{do:<5} {g:>5} {acc:>9.2f} {tok:>9.2f} {tok:>7.1f}x")
