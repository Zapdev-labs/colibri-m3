#!/usr/bin/env python3
"""test_tokenizer.py — VAL-CORR-022: tokenizer encode/decode round-trip is identity.

Verifies the colibri-m3 tokenizer (HF `tokenizers` lib reading
$SNAP/tokenizer.json, defaulting to /path/to/m3_i4/tokenizer.json)
round-trips a corpus of test strings without loss: decode(encode(s)) == s.

Test corpus includes:
  (a) plain ASCII
  (b) Unicode (CJK, emoji, accented Latin)
  (c) M3 special tokens (BOS ]~b], EOS [e~[, PAD ]!p~[, image ]<]image[>[, video ]<]video[>[)
  (d) code with indentation and special characters
  (e) an empty string

Run: python3 c/tests/test_tokenizer.py
     (or: SNAP=/path/to/m3_i4 python3 c/tests/test_tokenizer.py)
"""
from __future__ import annotations

import os
import sys

# Locate the tokenizer.json. Prefer $SNAP, then the standard v2 snapshot path,
# then the m3-target-meta path (tokenizer source-of-truth).
CANDIDATES = [
    os.environ.get("SNAP"),
    "/path/to/m3_i4",
    "/path/to/m3-target-meta",
]
TOK_PATH = None
for c in CANDIDATES:
    if c and os.path.isfile(os.path.join(c, "tokenizer.json")):
        TOK_PATH = os.path.join(c, "tokenizer.json")
        break

if TOK_PATH is None:
    print("SKIP: tokenizer.json not found (set SNAP=/path/to/m3_i4)")
    sys.exit(0)  # skip is not a failure for `make test` on a clean checkout

try:
    from tokenizers import Tokenizer
except ImportError:
    print(f"SKIP: tokenizers lib not installed (cannot test {TOK_PATH})")
    sys.exit(0)

tok = Tokenizer.from_file(TOK_PATH)
print(f"loaded tokenizer: {TOK_PATH}")

failures: list[str] = []

# --- Test corpus ---
CORPUS = [
    # (a) plain ASCII
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "1234567890",
    "Mixed CASE and spaces.",
    "a",
    "Tab\tnewline\nreturn\r",
    # (b) Unicode
    "Unicode: café, naïve, résumé",                # accented Latin
    "Chinese: 你好世界",                              # CJK
    "Japanese: こんにちは",                            # CJK
    "Emoji: 😀🚀💯",                                  # emoji
    "Mixed: Hello 世界 🌍Привет",                      # mixed scripts
    "Korean: 안녕하세요",
    # (c) M3 special tokens (should round-trip as single tokens).
    # NOTE: HF tokenizers default skip_special_tokens=True on decode, which
    # would strip special tokens and break round-trip. We use
    # skip_special_tokens=False so the special tokens survive the round-trip.
    "]~b]",          # BOS
    "[e~[",          # EOS
    "]!p~[",         # PAD
    "]<]image[>[",   # image placeholder
    "]<]video[>[",  # video placeholder
    "Text with ]~b] embedded [e~[ tokens ]!p~[.",
    # (d) code
    "def foo(x):\n    return x + 1",
    "array[0] = {key: 'value'}  # comment",
    "    indented\n        deeper\n",
    "Special: !@#$%^&*()_+-=[]{}|;':\",./<>?",
    # (e) empty string
    "",
]

# Look up the actual special-token IDs from the tokenizer (rather than
# hardcoding them — the snapshot's tokenizer_config may differ from the
# contract's documented IDs). We verify each special token encodes to a
# single ID and that the same ID decodes back to the token string.
SPECIAL_TOKENS = ["]~b]", "[e~[", "]!p~[", "]<]image[>[", "]<]video[>["]

n_pass = 0
for s in CORPUS:
    ids = tok.encode(s).ids
    decoded = tok.decode(ids, skip_special_tokens=False)
    if decoded == s:
        n_pass += 1
    else:
        failures.append(f"round-trip mismatch for {s!r} -> ids={ids[:8]}... -> {decoded!r}")

# Verify each special token encodes to a single, stable token ID and that
# decoding that ID (with skip_special_tokens=False) returns the token string.
special_id_ok = 0
for tok_str in SPECIAL_TOKENS:
    ids = tok.encode(tok_str).ids
    if len(ids) != 1:
        failures.append(f"special token {tok_str!r} encodes to {len(ids)} ids {ids}, expected 1")
        continue
    tid = ids[0]
    # Verify the token-to-id lookup agrees.
    looked_up = tok.token_to_id(tok_str)
    if looked_up != tid:
        failures.append(f"special token {tok_str!r}: encode->{tid} but token_to_id->{looked_up}")
        continue
    # Verify decode (skip_special_tokens=False) returns the original string.
    decoded = tok.decode([tid], skip_special_tokens=False)
    if decoded == tok_str:
        special_id_ok += 1
    else:
        failures.append(f"special token {tok_str!r} (id={tid}) decodes to {decoded!r}, expected {tok_str!r}")

print(f"PASS: {n_pass}/{len(CORPUS)} test strings identity-encode-decode")
print(f"PASS: {special_id_ok}/{len(SPECIAL_TOKENS)} special tokens resolve to stable single IDs and round-trip")
# Print the resolved special-token IDs for diagnostic purposes.
for tok_str in SPECIAL_TOKENS:
    print(f"   {tok_str!r} -> id {tok.token_to_id(tok_str)}")

if failures:
    print("\nFAILURES:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("\nPASS: all tokenizer round-trip tests passed")

