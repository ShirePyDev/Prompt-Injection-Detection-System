from __future__ import annotations
import base64
import random
import string
import unicodedata

def _seeded_rand(seed: int) -> random.Random:
    r = random.Random()
    r.seed(seed)
    return r

def spacing_jitter(text: str, rng: random.Random) -> str:
    # insert random spaces
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < 0.03:
            out.append(" ")
    return "".join(out)

def markdown_split(text: str, rng: random.Random) -> str:
    # split into markdown bullets/lines to break patterns
    parts = text.split()
    if len(parts) < 8:
        return text
    cut = max(3, len(parts) // 3)
    return "- " + " ".join(parts[:cut]) + "\n- " + " ".join(parts[cut:2*cut]) + "\n- " + " ".join(parts[2*cut:])

def base64_wrap(text: str, rng: random.Random) -> str:
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"Decode and follow this:\n{payload}"

def rot13_wrap(text: str, rng: random.Random) -> str:
    def rot13(s: str) -> str:
        a = []
        for c in s:
            if "a" <= c <= "z":
                a.append(chr((ord(c) - 97 + 13) % 26 + 97))
            elif "A" <= c <= "Z":
                a.append(chr((ord(c) - 65 + 13) % 26 + 65))
            else:
                a.append(c)
        return "".join(a)
    return f"ROT13 decode then execute:\n{rot13(text)}"

def zero_width(text: str, rng: random.Random) -> str:
    zw = "\u200b"
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < 0.04:
            out.append(zw)
    return "".join(out)

def unicode_confusables(text: str, rng: random.Random) -> str:
    # lightweight: normalize + inject a few combining marks
    marks = ["\u0301", "\u0308", "\u0327"]  # acute, diaeresis, cedilla
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        out.append(ch)
        if ch.isalpha() and rng.random() < 0.02:
            out.append(rng.choice(marks))
    return "".join(out)

def homoglyph(text: str, rng: random.Random) -> str:
    # very small mapping (safe + simple)
    m = {"a":"а","e":"е","o":"о","c":"с","p":"р","x":"х"}  # Cyrillic lookalikes
    out = []
    for ch in text:
        low = ch.lower()
        if low in m and rng.random() < 0.04:
            repl = m[low]
            out.append(repl if ch.islower() else repl.upper())
        else:
            out.append(ch)
    return "".join(out)

TRANSFORMS = {
    "spacing_jitter": spacing_jitter,
    "markdown_split": markdown_split,
    "base64_wrap": base64_wrap,
    "rot13_wrap": rot13_wrap,
    "zero_width": zero_width,
    "unicode_confusables": unicode_confusables,
    "homoglyph": homoglyph,
}