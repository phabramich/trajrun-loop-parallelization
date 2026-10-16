"""AV1 in miniature: speculative decode of an adaptive arithmetic coder.

Witten-Neal-Cleary range coder (16-bit, multi-symbol CDF, order-0 adaptive
model -- same mechanism as AV1's od_ec: CDF state is updated after every
symbol and remembers the whole history).

Question: can you jump into the middle of the bitstream and decode in
parallel? The coder-internal state (lo/hi/val) is flushed by renormalization
every few bits; the *model* state (CDF) is what never heals. Measure it:
- adaptive stream + wrong CDF guess at a known entry point -> ?
- static CDF + known entry point                  -> ?
- cost of making the stream parallel-ready (CDF reset every C symbols,
  the FFV1/JPEG XL move) in extra bytes.
"""

import numpy as np
from bisect import bisect_right

BITS = 16
TOP = (1 << BITS) - 1
Q1, Q2, Q3 = (TOP + 1) // 4, (TOP + 1) // 2, 3 * (TOP + 1) // 4

N = 16        # alphabet size
MAXF = 4096   # rescale threshold
STEP = 16     # adaptation step per symbol


class Model:
    def __init__(self, adaptive, freq=None):
        self.freq = list(freq) if freq is not None else [1] * N
        self.adaptive = adaptive

    @property
    def total(self):
        return sum(self.freq)

    def cum(self):
        c = [0]
        for f in self.freq:
            c.append(c[-1] + f)
        return c

    def update(self, s):
        if self.adaptive:
            self.freq[s] += STEP
            if self.total > MAXF:
                self.freq = [max(1, f >> 1) for f in self.freq]


class Encoder:
    def __init__(self, adaptive, freq=None, reset_prior=None, reset_every=0):
        self.m = Model(adaptive, freq)
        self.reset_prior, self.reset_every = reset_prior, reset_every
        self.lo, self.hi, self.pending, self.bits = 0, TOP, 0, []

    def _emit(self, b):
        self.bits.append(b)
        self.bits += [1 - b] * self.pending
        self.pending = 0

    def encode(self, i, s):
        if self.reset_every and i % self.reset_every == 0:
            self.m.freq = list(self.reset_prior)
        c = self.m.cum()
        rng_ = self.hi - self.lo + 1
        self.hi = self.lo + rng_ * c[s + 1] // self.m.total - 1
        self.lo = self.lo + rng_ * c[s] // self.m.total
        while True:
            if self.hi < Q2:
                self._emit(0)
            elif self.lo >= Q2:
                self._emit(1)
                self.lo -= Q2
                self.hi -= Q2
            elif self.lo >= Q1 and self.hi < Q3:
                self.pending += 1
                self.lo -= Q1
                self.hi -= Q1
            else:
                break
            self.lo <<= 1
            self.hi = (self.hi << 1) | 1
        self.m.update(s)

    def finish(self):
        self.pending += 1
        self._emit(1 if self.lo >= Q1 else 0)
        return np.array(self.bits, dtype=np.uint8)


class Decoder:
    def __init__(self, bits, adaptive, freq=None, entry=None):
        self.bits, self.m = bits, Model(adaptive, freq)
        if entry is None:  # stream start: canonical state, preload window
            self.pos, self.lo, self.hi = 0, 0, TOP
            w = np.zeros(BITS, dtype=np.uint8)
            w[: len(bits[:BITS])] = bits[:BITS]
            self.val = int(w @ (1 << np.arange(BITS - 1, -1, -1)))
            self.pos = BITS
        else:
            self.pos, self.lo, self.hi, self.val = entry

    def _bit(self):
        b = int(self.bits[self.pos]) if self.pos < len(self.bits) else 0
        self.pos += 1
        return b

    @property
    def entry(self):
        """Checkpoint: full coder-internal state at a symbol boundary
        (~48 bits for a 16-bit WNC; rANS-style coders need ~1 word)."""
        return (self.pos, self.lo, self.hi, self.val)

    def decode(self):
        c = self.m.cum()
        rng_ = self.hi - self.lo + 1
        target = ((self.val - self.lo + 1) * self.m.total - 1) // rng_
        s = bisect_right(c, target) - 1
        self.hi = self.lo + rng_ * c[s + 1] // self.m.total - 1
        self.lo = self.lo + rng_ * c[s] // self.m.total
        while True:
            if self.hi < Q2:
                pass
            elif self.lo >= Q2:
                self.lo -= Q2
                self.hi -= Q2
                self.val -= Q2
            elif self.lo >= Q1 and self.hi < Q3:
                self.val -= Q1
                self.lo -= Q1
                self.hi -= Q1
            else:
                break
            self.lo <<= 1
            self.hi = (self.hi << 1) | 1
            self.val = (self.val << 1) | self._bit()
        self.m.update(s)
        return s


def encode_all(syms, adaptive, freq=None, reset_prior=None, reset_every=0):
    e = Encoder(adaptive, freq, reset_prior, reset_every)
    for i, s in enumerate(syms):
        e.encode(i, int(s))
    return e.finish()


def decode_trace(bits, n, adaptive, freq=None, reset_prior=None,
                 reset_every=0):
    d = Decoder(bits, adaptive, freq)
    out, entries, snaps = [], [], []
    for i in range(n):
        if reset_every and i % reset_every == 0:
            d.m.freq = list(reset_prior)
        entries.append(d.entry)
        snaps.append(list(d.m.freq))
        out.append(d.decode())
    return np.array(out), entries, snaps


def spec_decode(bits, entry, freq, n, adaptive=True):
    d = Decoder(bits, adaptive=adaptive, freq=freq, entry=entry)
    return np.array([d.decode() for _ in range(n)])


def first_divergence(a, b):
    diff = np.nonzero(a != b)[0]
    return int(diff[0]) if len(diff) else len(a)


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


rng = np.random.default_rng(1)
p = 1.0 / np.arange(1, N + 1) ** 1.2
p /= p.sum()
SYMS = rng.choice(N, 4000, p=p).astype(int)
T0, W = 2000, 400  # speculate at symbol 2000, look 400 ahead

# --- sanity: round-trip -------------------------------------------------------
bits_ad = encode_all(SYMS, adaptive=True)
ref, entries, snaps = decode_trace(bits_ad, len(SYMS), adaptive=True)
assert np.array_equal(ref, SYMS), "adaptive round-trip broken"

prior = (p * 512).astype(int) + 1
bits_st = encode_all(SYMS, adaptive=False, freq=prior)
ref_st, entries_st, _ = decode_trace(bits_st, len(SYMS), adaptive=False,
                                     freq=prior)
assert np.array_equal(ref_st, SYMS), "static round-trip broken"

# --- experiment 1: healing ----------------------------------------------------
print(f"adaptive stream: {len(bits_ad)} bits for {len(SYMS)} syms "
      f"({len(bits_ad)/len(SYMS):.2f} b/sym, src entropy "
      f"{-(p*np.log2(p)).sum():.2f} b/sym)\n")
print(f"{'spec decode at sym 2000':<42} {'1st div':>8} {'match%':>7} "
      f"{'heal run':>9}")

for name, bits, entry, freq, refw, ad in [
    ("adaptive + uniform CDF", bits_ad, entries[T0], [1] * N,
     ref[T0:T0 + W], True),
    ("adaptive + marginal CDF (good guess)", bits_ad, entries[T0],
     list(prior), ref[T0:T0 + W], True),
    ("adaptive + exact CDF snapshot", bits_ad, entries[T0], snaps[T0],
     ref[T0:T0 + W], True),
    ("static + right entry", bits_st, entries_st[T0], list(prior),
     ref_st[T0:T0 + W], False),
    ("static + entry off by ONE bit", bits_st,
     (entries_st[T0][0] + 1,) + entries_st[T0][1:], list(prior),
     ref_st[T0:T0 + W], False),
]:
    got = spec_decode(bits, entry, freq, W, adaptive=ad)
    match = got == refw
    fd = first_divergence(got, refw)
    print(f"{name:<42} {fd:>8} {100*match.mean():>6.1f} "
          f"{longest_run(match[fd:]):>9}")

# --- experiment 2: cost of a parallel-ready stream ----------------------------
print("\nCDF reset every C symbols (entry-point per chunk = only TOC):")
print(f"{'C':>6} {'bytes':>7} {'overhead':>9} {'chunks':>7} {'decode ok':>10}")
base = len(bits_ad) / 8
for C in (0, 1024, 512, 256, 128, 64, 32):
    b = encode_all(SYMS, adaptive=True, reset_prior=[1] * N, reset_every=C)
    ok = ""
    if C:
        r, e, _ = decode_trace(b, len(SYMS), adaptive=True,
                               reset_prior=[1] * N, reset_every=C)
        assert np.array_equal(r, SYMS)
        got = spec_decode(b, e[3 * C], [1] * N, min(C, len(SYMS) - 3 * C))
        ok = "yes" if np.array_equal(got, r[3 * C:3 * C + len(got)]) \
            else "NO"
    print(f"{C or 'inf':>6} {len(b)/8:>7} {(len(b)/8-base)/base:>8.1%} "
          f"{len(SYMS)//C if C else 1:>7} {ok:>10}")
