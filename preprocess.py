import re

# ===============================
# KAMUS NORMALISASI SLANG
# ===============================

kamus_normalisasi = {
    "gue": "saya", "gua": "saya", "gw": "saya", "gwe": "saya",
    "lu": "kamu", "lo": "kamu", "loe": "kamu", "elo": "kamu",

    "ga": "tidak", "gak": "tidak", "gk": "tidak",
    "ngga": "tidak", "nggak": "tidak", "engga": "tidak",
    "kagak": "tidak", "kgk": "tidak",
    "gapapa": "tidak apa apa", "gabisa": "tidak bisa",
    "gajadi": "tidak jadi", "gajelas": "tidak jelas",

    "ajg": "anjing", "ajgg": "anjing", "ajgajg": "anjing",
    "anj": "anjing", "anjir": "anjing", "anjirr": "anjing",
    "anjr": "anjing", "anjay": "anjing", "anjayyy": "anjing",
    "anjg": "anjing", "anjinggg": "anjing", "anjeng": "anjing",
    "anjenggg": "anjing",

    "bgt": "banget", "bgtt": "banget", "bgttt": "banget",
    "bngt": "banget", "bngtt": "banget",

    "udah": "sudah", "udahh": "sudah", "udahan": "sudah",
    "udahlah": "sudah", "dah": "sudah", "dahh": "sudah",

    "bodo": "bodoh", "bodoamat": "bodoh amat",
    "bodohamat": "bodoh amat", "gemes": "gemas",

    "aja": "saja", "ajaa": "saja",
    "yg": "yang",
    "tp": "tapi", "tpi": "tapi",
    "jd": "jadi", "jdnya": "jadinya",
    "sm": "sama", "smua": "semua",
    "dr": "dari",
    "pdhl": "padahal",
    "krn": "karena",
    "klo": "kalau", "kl": "kalau",
    "skrg": "sekarang",
    "sdh": "sudah",
    "blm": "belum",
    "dgn": "dengan",

    "bagusbgt": "bagus banget",
    "harubgt": "haru banget",
    "nangisbgt": "nangis banget"
}

# ===============================
# FUNGSI CLEANING & NORMALISASI
# ===============================

def bersihkan_noise(teks: str) -> str:
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)
    teks = re.sub(r"@\w+", " ", teks)
    teks = re.sub(r"#\w+", " ", teks)
    teks = re.sub(r"[^0-9a-zA-ZÀ-ÿ .,!?-]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks

def singkatkan_karakter_berulang(kata: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", kata)

def normalisasi_token(token: str) -> str:
    t = token.lower()
    t = singkatkan_karakter_berulang(t)

    if t in kamus_normalisasi:
        return kamus_normalisasi[t]
    if re.fullmatch(r"w+k+w+k\w*", t):
        return "tertawa"
    if re.fullmatch(r"(ha)+\w*", t):
        return "tertawa"
    if t.startswith("anj") or t.startswith("ajg"):
        return "anjing"

    return t

def normalisasi_kalimat(teks: str) -> str:
    return " ".join(normalisasi_token(k) for k in teks.split())

def pra_pemrosesan(teks: str) -> str:
    teks = bersihkan_noise(teks)
    teks = teks.lower()
    teks = normalisasi_kalimat(teks)
    return teks
