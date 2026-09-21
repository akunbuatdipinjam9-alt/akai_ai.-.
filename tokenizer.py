import re

_PUNCT_RE = re.compile(r"[.,!?;:()\[\]{}<>\"'`~*/\\|_+=^]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text):
    """Lowercase, buang tanda baca, rapikan spasi. Tidak memecah jadi token."""
    if text is None:
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text):
    """Pecah teks jadi daftar token kata, setelah dinormalisasi."""
    return normalize_text(text).split()


if __name__ == "__main__":
    contoh = [
        "Halo, apa kabar?",
        "APA ITU AI!!!",
        "gimana   caranya   nih...",
        "kamu suka warna apa?",
    ]
    for kalimat in contoh:
        print(f"{kalimat!r:35} -> {tokenize(kalimat)}")
