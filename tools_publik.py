import ast
import itertools
import json
import operator as op
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo


_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("hanya angka yang diperbolehkan")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("ekspresi tidak didukung")


def hitung(ekspresi):
    """Hitung ekspresi matematika sederhana. Return (hasil, error) —
    salah satu selalu None."""
    if not ekspresi or not ekspresi.strip():
        return None, "ekspresi kosong"

    bersih = (
        ekspresi.strip()
        .replace(",", ".")
        .replace("x", "*")
        .replace("X", "*")
        .replace(":", "/")
    )
    try:
        tree = ast.parse(bersih, mode="eval")
        hasil = _eval_node(tree.body)
    except ZeroDivisionError:
        return None, "gak bisa bagi dengan nol"
    except (SyntaxError, ValueError, TypeError, KeyError) as e:
        return None, f"ekspresi gak valid ({e})"

    if isinstance(hasil, float) and hasil.is_integer():
        hasil = int(hasil)
    return hasil, None


def _load_todo(path):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _save_todo(path, items):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def todo_tambah(path, tugas):
    items = _load_todo(path)
    new_id = max((it.get("id", 0) for it in items), default=0) + 1
    items.append({"id": new_id, "tugas": tugas, "selesai": False})
    _save_todo(path, items)
    return new_id


def todo_list(path):
    return _load_todo(path)


def todo_selesai(path, item_id):
    items = _load_todo(path)
    for it in items:
        if it.get("id") == item_id:
            it["selesai"] = True
            _save_todo(path, items)
            return True
    return False


def todo_hapus(path, item_id):
    items = _load_todo(path)
    baru = [it for it in items if it.get("id") != item_id]
    if len(baru) == len(items):
        return False
    _save_todo(path, baru)
    return True


def format_todo_list(items):
    if not items:
        return "To-do list masih kosong."
    baris = ["--- To-Do List ---"]
    for it in items:
        tanda = "[x]" if it.get("selesai") else "[ ]"
        baris.append(f"{it.get('id')}. {tanda} {it.get('tugas', '')}")
    baris.append("------------------")
    return "\n".join(baris)

_WMO_TEKS = {
    0: "cerah",
    1: "cerah berawan sebagian",
    2: "berawan sebagian",
    3: "mendung",
    45: "berkabut",
    48: "kabut es",
    51: "gerimis ringan",
    53: "gerimis sedang",
    55: "gerimis lebat",
    56: "gerimis beku ringan",
    57: "gerimis beku lebat",
    61: "hujan ringan",
    63: "hujan sedang",
    65: "hujan lebat",
    66: "hujan beku ringan",
    67: "hujan beku lebat",
    71: "salju ringan",
    73: "salju sedang",
    75: "salju lebat",
    77: "butiran salju",
    80: "hujan lokal ringan",
    81: "hujan lokal sedang",
    82: "hujan lokal lebat",
    85: "hujan salju ringan",
    86: "hujan salju lebat",
    95: "badai petir",
    96: "badai petir + hujan es ringan",
    99: "badai petir + hujan es lebat",
}


def _wmo_ke_teks(kode):
    return _WMO_TEKS.get(kode, f"kondisi tidak dikenal (kode {kode})")


def _http_get_json(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "chatbot-from-scratch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cari_lokasi(nama_kota, timeout=10):
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({
        "name": nama_kota,
        "count": 1,
        "language": "id",
        "format": "json",
    })
    data = _http_get_json(url, timeout)
    hasil = data.get("results")
    if not hasil:
        return None
    lokasi = hasil[0]
    return {
        "nama": lokasi.get("name", nama_kota),
        "negara": lokasi.get("country", ""),
        "lat": lokasi["latitude"],
        "lon": lokasi["longitude"],
        "timezone": lokasi.get("timezone"),
    }


def ambil_cuaca_sekarang(lat, lon, timeout=10):
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
    })
    data = _http_get_json(url, timeout)
    return data.get("current_weather")


def cuaca_kota(nama_kota, timeout=10):
    """Ambil cuaca sekarang untuk sebuah kota lewat Open-Meteo.
    Return (teks_hasil, error) — salah satu selalu None."""
    if not nama_kota or not nama_kota.strip():
        return None, "nama kota kosong"

    try:
        lokasi = cari_lokasi(nama_kota.strip(), timeout=timeout)
    except urllib.error.URLError as e:
        return None, f"gagal menghubungi layanan geocoding ({e.reason})"
    except Exception as e:
        return None, f"gagal menghubungi layanan geocoding ({e})"

    if lokasi is None:
        return None, f"kota '{nama_kota}' tidak ditemukan"

    try:
        cuaca = ambil_cuaca_sekarang(lokasi["lat"], lokasi["lon"], timeout=timeout)
    except urllib.error.URLError as e:
        return None, f"gagal menghubungi layanan cuaca ({e.reason})"
    except Exception as e:
        return None, f"gagal menghubungi layanan cuaca ({e})"

    if not cuaca:
        return None, "data cuaca tidak tersedia untuk lokasi ini"

    suhu = cuaca.get("temperature")
    angin = cuaca.get("windspeed")
    kode = cuaca.get("weathercode")
    lokasi_teks = lokasi["nama"] + (f", {lokasi['negara']}" if lokasi["negara"] else "")
    teks = (
        f"Cuaca di {lokasi_teks}:\n"
        f"  Kondisi : {_wmo_ke_teks(kode)}\n"
        f"  Suhu    : {suhu}°C\n"
        f"  Angin   : {angin} km/j"
    )
    return teks, None


def waktu_kota(nama_kota, timeout=10):
    """Jam sekarang di sebuah kota (pakai timezone dari Open-Meteo geocoding +
    zoneinfo bawaan python, gak butuh API tambahan). Return (teks, error)."""
    if not nama_kota or not nama_kota.strip():
        return None, "nama kota kosong"

    try:
        lokasi = cari_lokasi(nama_kota.strip(), timeout=timeout)
    except urllib.error.URLError as e:
        return None, f"gagal menghubungi layanan geocoding ({e.reason})"
    except Exception as e:
        return None, f"gagal menghubungi layanan geocoding ({e})"

    if lokasi is None:
        return None, f"kota '{nama_kota}' tidak ditemukan"
    tz_nama = lokasi.get("timezone")
    if not tz_nama:
        return None, f"gak ada info zona waktu untuk '{nama_kota}'"

    try:
        sekarang = datetime.now(ZoneInfo(tz_nama))
    except Exception as e:
        return None, f"gagal baca zona waktu '{tz_nama}' ({e})"

    lokasi_teks = lokasi["nama"] + (f", {lokasi['negara']}" if lokasi["negara"] else "")
    teks = (
        f"Waktu sekarang di {lokasi_teks} ({tz_nama}):\n"
        f"  {sekarang.strftime('%A, %d %B %Y - %H:%M:%S')}"
    )
    return teks, None


_HARI_ID = {
    "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu", "Thursday": "Kamis",
    "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu",
}


def jadwal_sholat(input_kota, timeout=10, method=20):
    """Jadwal sholat harian lewat Aladhan API. `input_kota` boleh cuma nama
    kota (default negara Indonesia) atau 'kota,negara'. Return (teks, error)."""
    if not input_kota or not input_kota.strip():
        return None, "nama kota kosong"

    bagian = [b.strip() for b in input_kota.split(",")]
    kota = bagian[0]
    negara = bagian[1] if len(bagian) > 1 and bagian[1] else "Indonesia"

    url = "https://api.aladhan.com/v1/timingsByCity?" + urllib.parse.urlencode({
        "city": kota,
        "country": negara,
        "method": method,
    })
    try:
        data = _http_get_json(url, timeout)
    except urllib.error.HTTPError as e:
        return None, f"gagal ambil jadwal sholat (HTTP {e.code}, kota/negara mungkin salah)"
    except urllib.error.URLError as e:
        return None, f"gagal menghubungi layanan jadwal sholat ({e.reason})"
    except Exception as e:
        return None, f"gagal menghubungi layanan jadwal sholat ({e})"

    if data.get("code") != 200 or "data" not in data:
        return None, "data jadwal sholat tidak ditemukan untuk lokasi ini"

    timings = data["data"].get("timings", {})
    tanggal = data["data"].get("date", {}).get("readable", "")
    hari_en = data["data"].get("date", {}).get("gregorian", {}).get("weekday", {}).get("en", "")
    hari_id = _HARI_ID.get(hari_en, hari_en)

    urutan = ["Fajr", "Sunrise", "Dhuhr", "Asr", "Maghrib", "Isha"]
    nama_id = {
        "Fajr": "Subuh", "Sunrise": "Terbit", "Dhuhr": "Dzuhur",
        "Asr": "Ashar", "Maghrib": "Maghrib", "Isha": "Isya",
    }
    baris = [f"Jadwal sholat {kota.title()}, {negara.title()} ({hari_id}, {tanggal}):"]
    for kunci in urutan:
        if kunci in timings:
            baris.append(f"  {nama_id[kunci]:<8}: {timings[kunci]}")
    return "\n".join(baris), None


def terjemah_teks(teks, target="en", timeout=10):
    """Terjemahkan teks lewat MyMemory API (gratis, gak butuh API key).
    Default: id -> en. Kalau target == 'id', otomatis dianggap en -> id.
    Return (hasil, error)."""
    if not teks or not teks.strip():
        return None, "teks kosong"

    target = (target or "en").strip().lower()
    sumber = "en" if target == "id" else "id"

    url = "https://api.mymemory.translated.net/get?" + urllib.parse.urlencode({
        "q": teks.strip(),
        "langpair": f"{sumber}|{target}",
    })
    try:
        data = _http_get_json(url, timeout)
    except urllib.error.URLError as e:
        return None, f"gagal menghubungi layanan terjemahan ({e.reason})"
    except Exception as e:
        return None, f"gagal menghubungi layanan terjemahan ({e})"

    hasil = data.get("responseData", {}).get("translatedText")
    if not hasil:
        return None, "gagal menerjemahkan teks itu"
    return hasil, None


def kurs_mata_uang(dari, ke, jumlah=1.0, timeout=10):
    """Kurs mata uang lewat open.er-api.com (gratis, gak butuh API key).
    Return (teks, error)."""
    if not dari or not ke:
        return None, "kode mata uang kosong (contoh: /kurs USD IDR)"

    dari = dari.strip().upper()
    ke = ke.strip().upper()

    url = f"https://open.er-api.com/v6/latest/{urllib.parse.quote(dari)}"
    try:
        data = _http_get_json(url, timeout)
    except urllib.error.HTTPError as e:
        return None, f"kode mata uang '{dari}' gak dikenal (HTTP {e.code})"
    except urllib.error.URLError as e:
        return None, f"gagal menghubungi layanan kurs ({e.reason})"
    except Exception as e:
        return None, f"gagal menghubungi layanan kurs ({e})"

    if data.get("result") != "success":
        return None, "gagal ambil data kurs"

    rate = data.get("rates", {}).get(ke)
    if rate is None:
        return None, f"kode mata uang '{ke}' gak ditemukan"

    hasil = jumlah * rate
    teks = f"{jumlah:g} {dari} = {hasil:,.2f} {ke} (kurs 1 {dari} = {rate:,.4f} {ke})"
    return teks, None


def wiki_ringkas(topik, timeout=10):
    """Ringkasan singkat dari Wikipedia bahasa Indonesia. Return (teks, error)."""
    if not topik or not topik.strip():
        return None, "topik kosong"

    judul = urllib.parse.quote(topik.strip().replace(" ", "_"))
    url = f"https://id.wikipedia.org/api/rest_v1/page/summary/{judul}"
    try:
        data = _http_get_json(url, timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, f"topik '{topik}' gak ditemukan di Wikipedia"
        return None, f"gagal ambil data Wikipedia (HTTP {e.code})"
    except urllib.error.URLError as e:
        return None, f"gagal menghubungi Wikipedia ({e.reason})"
    except Exception as e:
        return None, f"gagal menghubungi Wikipedia ({e})"

    extract = data.get("extract")
    if not extract:
        return None, f"gak ada ringkasan buat topik '{topik}'"
    judul_asli = data.get("title", topik)
    return f"{judul_asli}:\n{extract}", None


KITAB_HADITS = [
    "bukhari", "muslim", "tirmidzi", "nasai", "abu-daud",
    "ibnu-majah", "ahmad", "darimi", "malik",
]
_ALIAS_KITAB = {
    "abu daud": "abu-daud", "abu-daud": "abu-daud", "abudaud": "abu-daud",
    "ibnu majah": "ibnu-majah", "ibnumajah": "ibnu-majah",
    "bukhori": "bukhari", "muslim": "muslim", "tirmidzi": "tirmidzi",
    "nasai": "nasai", "ahmad": "ahmad", "darimi": "darimi", "malik": "malik",
}

_cache_jumlah_hadits = {}


def _normalisasi_kitab(nama):
    nama = (nama or "bukhari").strip().lower().replace("_", "-")
    nama = _ALIAS_KITAB.get(nama, nama)
    return nama


def _jumlah_hadits_kitab(kitab, timeout=10):
    if kitab in _cache_jumlah_hadits:
        return _cache_jumlah_hadits[kitab]
    data = _http_get_json("https://api.hadith.gading.dev/books", timeout)
    for buku in data.get("data", []):
        bid = str(buku.get("id", "")).lower()
        if bid == kitab:
            jumlah = buku.get("available")
            if jumlah:
                _cache_jumlah_hadits[kitab] = jumlah
            return jumlah
    return None


def ambil_hadits(kitab=None, nomor=None, timeout=10):
    """Ambil satu hadits (teks Arab + terjemahan Indonesia) lewat API
    hadith.gading.dev. `kitab` salah satu dari KITAB_HADITS (default bukhari),
    `nomor` nomor hadits di kitab itu (kalau kosong, dipilih acak).
    Return (teks, error)."""
    kitab = _normalisasi_kitab(kitab)
    if kitab not in KITAB_HADITS:
        return None, f"kitab '{kitab}' gak dikenal. Pilihan: {', '.join(KITAB_HADITS)}"

    if nomor is None:
        try:
            jumlah = _jumlah_hadits_kitab(kitab, timeout=timeout)
        except Exception as e:
            return None, f"gagal ambil daftar kitab hadits ({e})"
        if not jumlah:
            return None, f"gak tau jumlah hadits di kitab '{kitab}'"
        nomor = random.randint(1, jumlah)
    else:
        try:
            nomor = int(nomor)
        except (TypeError, ValueError):
            return None, "nomor hadits harus angka"
        if nomor < 1:
            return None, "nomor hadits gak valid"

    url = f"https://api.hadith.gading.dev/books/{kitab}/{nomor}"
    try:
        data = _http_get_json(url, timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, f"hadits nomor {nomor} di kitab '{kitab}' gak ditemukan"
        return None, f"gagal ambil hadits (HTTP {e.code})"
    except urllib.error.URLError as e:
        return None, f"gagal menghubungi layanan hadits ({e.reason})"
    except Exception as e:
        return None, f"gagal menghubungi layanan hadits ({e})"

    if data.get("code") != 200 or not data.get("data"):
        return None, f"hadits nomor {nomor} di kitab '{kitab}' gak ditemukan"

    isi = data["data"].get("contents", {})
    nama_kitab = data["data"].get("name", kitab.title())
    arab = isi.get("arab")
    terjemahan = isi.get("id") or isi.get("translation") or isi.get("terjemahan")

    baris = [f"HR. {nama_kitab} No. {nomor}:", ""]
    if arab:
        baris.append(arab)
        baris.append("")
    if terjemahan:
        baris.append(f"Artinya: {terjemahan}")
    if not arab and not terjemahan:
        return None, "isi hadits kosong/gak lengkap dari server"
    return "\n".join(baris), None


def ayat_quran(surat, ayat, timeout=10):
    """Ambil satu ayat Al-Qur'an (teks Arab + terjemahan Indonesia) lewat
    Al Quran Cloud API. `surat` nomor 1-114, `ayat` nomor ayat di surat itu.
    Return (teks, error)."""
    try:
        surat = int(surat)
        ayat = int(ayat)
    except (TypeError, ValueError):
        return None, "nomor surat/ayat harus angka"

    if not (1 <= surat <= 114) or ayat < 1:
        return None, "nomor surat (1-114) atau nomor ayat gak valid"

    referensi = f"{surat}:{ayat}"
    url = f"https://api.alquran.cloud/v1/ayah/{referensi}/editions/quran-uthmani,id.indonesian"
    try:
        data = _http_get_json(url, timeout)
    except urllib.error.HTTPError as e:
        if e.code == 400 or e.code == 404:
            return None, f"ayat {referensi} gak ditemukan (cek lagi nomor surat/ayatnya)"
        return None, f"gagal ambil data Al-Qur'an (HTTP {e.code})"
    except urllib.error.URLError as e:
        return None, f"gagal menghubungi layanan Al-Qur'an ({e.reason})"
    except Exception as e:
        return None, f"gagal menghubungi layanan Al-Qur'an ({e})"

    if data.get("code") != 200 or not data.get("data"):
        return None, f"ayat {referensi} gak ditemukan"

    entri = data["data"]
    arab = next((d["text"] for d in entri if d.get("edition", {}).get("identifier") == "quran-uthmani"), None)
    terjemahan = next((d["text"] for d in entri if d.get("edition", {}).get("identifier") == "id.indonesian"), None)
    info_surat = entri[0].get("surah", {})
    nama_surat = info_surat.get("englishName", "")
    arti_surat = info_surat.get("englishNameTranslation", "")

    baris = [f"QS. {nama_surat} ({arti_surat}) ayat {ayat}:", ""]
    if arab:
        baris.append(arab)
        baris.append("")
    if terjemahan:
        baris.append(f"Artinya: {terjemahan}")
    return "\n".join(baris), None


_POLLINATIONS_URL = "https://text.pollinations.ai/{prompt}"
_POLLINATIONS_MODELS_URL = "https://text.pollinations.ai/models"

FALLBACK_MODELS_STATIS = ["openai", "mistral", "openai-large"]
_MODEL_PRIORITAS = ["openai", "mistral", "deepseek", "gemini", "claude", "openai-large", "grok", "qwen-coder"]

_cache_model_hidup = {"daftar": None}
_pollinations_cycle = None


def _http_get_text(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "chatbot-from-scratch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def ambil_daftar_model_pollinations(timeout=10, force_refresh=False):
    """Ambil daftar model teks yang beneran aktif sekarang dari pollinations.ai
    (daftarnya suka berubah-ubah, jadi jangan hardcode). Kalau gagal ambil,
    balik ke daftar statis cadangan."""
    if not force_refresh and _cache_model_hidup["daftar"]:
        return _cache_model_hidup["daftar"]

    try:
        teks = _http_get_text(_POLLINATIONS_MODELS_URL, timeout)
        data = json.loads(teks)
        if isinstance(data, list) and data:
            daftar = [m if isinstance(m, str) else m.get("name") for m in data]
            daftar = [m for m in daftar if m]
            if daftar:
                _cache_model_hidup["daftar"] = daftar
                return daftar
    except Exception:
        pass

    return FALLBACK_MODELS_STATIS


def pilih_3_model(timeout=10):
    """Pilih 3 model buat bergantian: utamakan yang ada di _MODEL_PRIORITAS
    (openai, mistral, deepseek, dst) kalau memang lagi hidup, sisanya diisi
    dari model apa saja yang tersedia."""
    hidup = ambil_daftar_model_pollinations(timeout=timeout)
    hidup_set = set(hidup)
    pilihan = [m for m in _MODEL_PRIORITAS if m in hidup_set][:3]
    if len(pilihan) < 3:
        for m in hidup:
            if m not in pilihan:
                pilihan.append(m)
            if len(pilihan) >= 3:
                break
    return pilihan or FALLBACK_MODELS_STATIS


def tanya_pollinations(pertanyaan, model="openai", timeout=15, system=None):
    """Tanya satu model AI lewat pollinations.ai (text.pollinations.ai).
    Return (jawaban, error) — salah satu selalu None."""
    if not pertanyaan or not pertanyaan.strip():
        return None, "pertanyaan kosong"

    query = {"model": model}
    if system:
        query["system"] = system
    url = _POLLINATIONS_URL.format(prompt=urllib.parse.quote(pertanyaan.strip())) + "?" + urllib.parse.urlencode(query)

    try:
        teks = _http_get_text(url, timeout)
    except urllib.error.HTTPError as e:
        return None, f"model '{model}' error HTTP {e.code}"
    except urllib.error.URLError as e:
        return None, f"gagal menghubungi pollinations.ai ({e.reason})"
    except Exception as e:
        return None, f"gagal menghubungi pollinations.ai ({e})"

    teks = teks.strip()
    if not teks:
        return None, f"model '{model}' tidak memberi jawaban"
    return teks, None


def pollinations_fallback(pertanyaan, models=None, timeout=15, system=None):
    """Backup kalau model lokal gak bisa jawab: coba 3 AI dari pollinations.ai
    bergantian (round-robin), lanjut ke model berikutnya kalau satu gagal atau
    modelnya udah gak ada lagi (mislanya 404). Return (jawaban, model_yang_jawab, error)
    — error cuma diisi kalau semua gagal."""
    global _pollinations_cycle

    daftar_model = models if models else pilih_3_model(timeout=min(timeout, 10))
    daftar_model = [m for m in daftar_model if m]
    if not daftar_model:
        return None, None, "tidak ada model pollinations yang dikonfigurasi"

    if _pollinations_cycle is None or models is not None:
        _pollinations_cycle = itertools.cycle(daftar_model)

    urutan = [next(_pollinations_cycle) for _ in range(len(daftar_model))]

    error_terakhir = None
    model_404 = False
    for model in urutan:
        jawaban, err = tanya_pollinations(pertanyaan, model=model, timeout=timeout, system=system)
        if jawaban:
            return jawaban, model, None
        error_terakhir = err
        if err and "404" in err:
            model_404 = True

    if model_404:
        segar = ambil_daftar_model_pollinations(timeout=min(timeout, 10), force_refresh=True)
        segar_valid = [m for m in urutan if m in segar]
        if len(segar_valid) < len(urutan):
            for model in pilih_3_model(timeout=min(timeout, 10)):
                jawaban, err = tanya_pollinations(pertanyaan, model=model, timeout=timeout, system=system)
                if jawaban:
                    return jawaban, model, None
                error_terakhir = err

    return None, None, error_terakhir or "semua model pollinations gagal dihubungi"



_POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}"


def generate_gambar(prompt, model="flux", width=1024, height=1024,
                     folder_simpan="gambar", timeout=60, seed=None):
    """Generate gambar dari teks lewat pollinations.ai (gratis, gak butuh
    API key/akun, cuma dibatasi ~1 request/15 detik buat pemakaian anonim).
    Balik (path_file, url_gambar, error) — error cuma diisi kalau gagal."""
    if not prompt or not prompt.strip():
        return None, None, "deskripsi gambar kosong"

    query = {"model": model, "width": width, "height": height, "nologo": "true"}
    if seed is not None:
        query["seed"] = seed
    url = (_POLLINATIONS_IMAGE_URL.format(prompt=urllib.parse.quote(prompt.strip()))
           + "?" + urllib.parse.urlencode(query))

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "chatbot-from-scratch/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            konten_type = resp.headers.get("Content-Type", "")
            data_gambar = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return None, None, "kena rate limit pollinations (429), tunggu ~15 detik lalu coba lagi"
        return None, None, f"gagal generate gambar (HTTP {e.code})"
    except urllib.error.URLError as e:
        return None, None, f"gagal menghubungi pollinations.ai ({e.reason})"
    except Exception as e:
        return None, None, f"gagal menghubungi pollinations.ai ({e})"

    if "image" not in konten_type or len(data_gambar) < 500:
        return None, None, "pollinations gak balikin gambar yang valid (coba lagi atau ganti deskripsi)"

    try:
        os.makedirs(folder_simpan, exist_ok=True)
        nama_file = f"gambar_{int(time.time())}.png"
        path_file = os.path.join(folder_simpan, nama_file)
        with open(path_file, "wb") as f:
            f.write(data_gambar)
    except Exception as e:
        return None, url, f"gambar berhasil dibuat tapi gagal disimpan ({e})"

    return path_file, url, None



# ============================================================================
# Integrasi Gemini API -- semua fitur AI Gemini (chat, function-calling buat
# mode agen /agen, grounding search) ada DI SINI, di file publik, biar aman
# di-push ke GitHub apa adanya. GEMINI_API_KEY di atas cuma PLACEHOLDER.
#
# Buat testing lokal pakai key asli TANPA ngedit file ini (yang di-track
# git): bikin file `tools_privat.py` (udah di-gitignore) isinya cuma:
#     GEMINI_API_KEY = "key_asli_kamu"
# Kalau file itu ada, key aslinya otomatis dipakai gantiin placeholder di
# bawah -- liat blok override di paling akhir file ini.
# ============================================================================

# TODO: ganti kalau key ini kadaluarsa / abis kuota
GEMINI_API_KEY = "MASUKKAN_API_KEY_GEMINI_KAMU_DI_SINI"

_GEMINI_LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_PRIORITAS_MODEL = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-pro-latest",
]

_cache_daftar_model = None


def _headers():
    # Google sekarang minta API key dikirim lewat header ini, bukan lewat
    # parameter URL ?key=... (itu penyebab error 401 sebelumnya).
    return {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }


def _daftar_model_gemini(timeout=10):
    """Ambil daftar model Gemini yang bisa dipakai generateContent, langsung
    dari API-nya Google (bukan hardcode) -- soalnya nama model suka berubah/
    di-deprecate seiring waktu."""
    req = urllib.request.Request(_GEMINI_LIST_URL, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    hasil = []
    for m in data.get("models", []):
        nama = m.get("name", "").replace("models/", "")
        if "generateContent" in (m.get("supportedGenerationMethods") or []):
            hasil.append(nama)
    return hasil


def _daftar_kandidat_model(timeout=10, paksa_refresh=False, maksimal=4):
    """Bikin daftar BEBERAPA model buat dicoba gantian (bukan cuma 1), biar
    kalau satu model lagi penuh/503, langsung lompat ke model lain yang
    kemungkinan kuotanya masih longgar."""
    global _cache_daftar_model
    if _cache_daftar_model and not paksa_refresh:
        return _cache_daftar_model

    try:
        daftar_live = _daftar_model_gemini(timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        daftar_live = []

    kandidat = [m for m in _PRIORITAS_MODEL if m in daftar_live]
    for nama in daftar_live:
        if len(kandidat) >= maksimal:
            break
        if nama in kandidat:
            continue
        if "flash" in nama and "vision" not in nama and "embedding" not in nama and "tts" not in nama:
            kandidat.append(nama)

    if not kandidat:
        # gagal ambil daftar live (mungkin lagi offline) -- tebak model yang
        # paling umum dipakai, tetap sedia beberapa biar ada opsi gantian
        kandidat = list(_PRIORITAS_MODEL)

    _cache_daftar_model = kandidat[:maksimal]
    return _cache_daftar_model


_KATA_KUNCI_INFO_TERBARU = (
    "hari ini", "sekarang", "terbaru", "terkini", "baru-baru ini",
    "berita", "kabar", "viral", "tren", "trending",
    "harga", "kurs", "skor", "hasil pertandingan", "jadwal",
    "cuaca", "gempa", "bencana",
    "siapa presiden", "siapa menteri", "siapa gubernur", "siapa ketua",
    "kapan rilis", "kapan keluar", "versi terbaru", "update terbaru",
    "tahun ini", "bulan ini", "minggu ini", "besok", "kemarin",
)


def butuh_info_terbaru(teks):
    """Nebak (heuristik kata kunci, murah, gak manggil API apa pun) apakah
    sebuah pertanyaan kemungkinan butuh info ter-update dari internet
    (berita, harga, cuaca, jabatan orang, dll) -- kalau iya, baru pantas
    nyalain gunakan_search=True di tanya_gemini(), biar kuota grounding yang
    kecil (~500/hari) gak abis percuma buat obrolan biasa yang gak butuh
    data terbaru sama sekali (basa-basi, curhat, coding, terjemahan, dll)."""
    teks_lower = (teks or "").lower()
    return any(k in teks_lower for k in _KATA_KUNCI_INFO_TERBARU)


def _ekstrak_sumber_search(kandidat, maksimal=3):
    """Ambil judul/URL sumber dari groundingMetadata (hasil Google Search
    yang dipakai Gemini buat jawab), kalau ada. Return list string singkat,
    bisa kosong kalau Gemini gak pakai fitur search buat jawaban ini."""
    meta = kandidat.get("groundingMetadata") or {}
    chunks = meta.get("groundingChunks") or []
    hasil = []
    for chunk in chunks:
        web = chunk.get("web") or {}
        judul = web.get("title") or web.get("uri")
        if judul and judul not in hasil:
            hasil.append(judul)
        if len(hasil) >= maksimal:
            break
    return hasil


def _panggil_generate(payload, model=None, timeout=15, percobaan_503_per_model=2):
    """Inti pemanggilan generateContent yang dipakai bareng sama
    tanya_gemini() (mode teks biasa) dan panggil_gemini_agen() (mode agen
    coding pakai function calling) -- biar logic rotasi-model & penanganan
    401/404/429/503 gak dobel ditulis di 2 tempat.

    `payload` = body request LENGKAP (contents + tools dll, apa pun yang mau
    dikirim), fungsi ini cuma nambahin/ganti-ganti target model-nya.
    Return (kandidat_dict, error) -- kandidat_dict = candidates[0] dari
    response (punya "content", bisa "groundingMetadata", dll), None kalau
    gagal total (dan `error` isinya alasan gagal, format ramah manusia)."""
    body = json.dumps(payload).encode("utf-8")

    kandidat_model = [model] if model else _daftar_kandidat_model(timeout=timeout)
    if not kandidat_model:
        kandidat_model = ["gemini-flash-latest"]

    error_terakhir = None
    sudah_refresh_404 = False

    for idx, nama_model in enumerate(kandidat_model):
        for percobaan in range(percobaan_503_per_model):
            url = _GEMINI_URL.format(model=nama_model)
            req = urllib.request.Request(url, data=body, method="POST", headers=_headers())
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                kandidat_list = data.get("candidates") or []
                if not kandidat_list:
                    return None, "Gemini gak ngasih jawaban (kemungkinan kena filter safety)"
                return kandidat_list[0], None
            except urllib.error.HTTPError as e:
                pesan = e.read().decode("utf-8", errors="ignore")[:200]

                if e.code == 401:
                    return None, (
                        "401 -- API key ditolak. Cek lagi apakah key di tools_privat.py "
                        "masih aktif (belum dihapus/expired) di https://aistudio.google.com/apikey"
                    )

                if e.code == 429:
                    error_terakhir = f"{nama_model}: kuota abis (429)"
                    break  # kuota per-model bisa beda, coba model lain, gak ada gunanya diulang di model yang sama

                if e.code == 503:
                    error_terakhir = f"{nama_model}: server sibuk (503)"
                    if percobaan < percobaan_503_per_model - 1:
                        time.sleep(1 + percobaan)  # jeda singkat sebelum coba model yang sama sekali lagi
                        continue
                    break  # model ini nyerah, lanjut ke model berikutnya

                if e.code == 404:
                    error_terakhir = f"{nama_model}: gak ketemu (404)"
                    if model is None and not sudah_refresh_404:
                        sudah_refresh_404 = True
                        kandidat_model = _daftar_kandidat_model(timeout=timeout, paksa_refresh=True)
                    break  # lanjut ke model berikutnya di daftar

                error_terakhir = f"{nama_model}: HTTP {e.code}: {pesan}"
                break  # error lain -- gak ada gunanya diulang, lanjut model berikutnya
            except urllib.error.URLError as e:
                return None, f"gagal konek ({e.reason})"
            except (TimeoutError, OSError) as e:
                return None, f"timeout/error ({e})"

    if error_terakhir and "kuota abis (429)" in error_terakhir:
        return None, (
            f"{error_terakhir} -- semua model kena limit kuota gratis harian. "
            "Kuota reset otomatis tiap hari (Pacific Time), atau upgrade billing "
            "kalau butuh lebih longgar: https://ai.google.dev/gemini-api/docs/rate-limits"
        )
    return None, error_terakhir or "semua model Gemini lagi sibuk/gak tersedia, coba lagi beberapa saat lagi"



def gemini_siap():
    """True kalau GEMINI_API_KEY di sini udah diisi API key ASLI (bukan
    placeholder). Dipakai chat_loop buat ngecek cepat apakah fitur Gemini
    boleh dicoba, tanpa perlu manggil API-nya dulu."""
    return bool(GEMINI_API_KEY) and not GEMINI_API_KEY.startswith("MASUKKAN_")


def tanya_gemini(prompt, timeout=15, model=None, system=None, percobaan_503_per_model=2, gunakan_search=False):
    """Tanya ke Gemini API. Kalau `model` gak dikasih, otomatis coba
    beberapa model gantian (dari yang paling live tersedia) -- soalnya
    free tier Gemini gampang 503 (server sibuk) kalau satu model lagi
    rame dipakai orang lain, jadi model lain kemungkinan masih longgar.
    Kalau `gunakan_search` True, Gemini dikasih akses "Grounding with Google
    Search" -- dia BOLEH nyari di internet dulu sebelum jawab kalau memang
    butuh info terbaru. DEFAULT-nya False (jangan asal nyalain tiap request!)
    soalnya kuota gratis buat fitur search ini jauh lebih kecil (~500
    request/hari, KE SEMUA model) dibanding kuota generateContent biasa --
    kalau tiap chat biasa ikut nyalain tools search, kuota abis dalam
    hitungan menit terus muncul error 429 "exceeded your current quota".
    Pakai helper butuh_info_terbaru(teks) buat mutusin kapan perlu dinyalain.
    Return (teks_jawaban, error). Sukses: (jawaban, None). Gagal: (None, "pesan error")."""
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("MASUKKAN_"):
        return None, "API key Gemini belum diisi (tools_privat.py)"

    isi_prompt = f"{system}\n\n{prompt}" if system else prompt
    payload = {"contents": [{"parts": [{"text": isi_prompt}]}]}
    if gunakan_search:
        payload["tools"] = [{"google_search": {}}]

    kandidat, error = _panggil_generate(payload, model=model, timeout=timeout, percobaan_503_per_model=percobaan_503_per_model)
    if kandidat is None:
        return None, error

    bagian = kandidat.get("content", {}).get("parts") or []
    teks = "".join(p.get("text", "") for p in bagian).strip()
    if not teks:
        return None, "Gemini balikin jawaban kosong"
    sumber = _ekstrak_sumber_search(kandidat)
    if sumber:
        teks += "\n\nSumber: " + ", ".join(sumber)
    return teks, None


def panggil_gemini_agen(contents, deklarasi_tools, timeout=45, model=None, percobaan_503_per_model=2):
    """Satu putaran percakapan multi-turn ke Gemini dengan FUNCTION CALLING
    aktif -- ini yang dipakai buat mode agen coding (/agen), beda dari
    tanya_gemini() yang cuma 1 kali tanya-jawab teks biasa.

    `contents` = riwayat percakapan lengkap format Gemini, list of
    {"role": "user"/"model"/"function", "parts": [...]}. Caller (yang manggil
    fungsi ini) yang nyimpen & nambahin riwayatnya turn demi turn -- fungsi
    ini cuma ngirim 1 request dan balikin 1 response mentah.

    `deklarasi_tools` = list of function declaration dict, format:
    {"name": ..., "description": ..., "parameters": {"type": "object",
    "properties": {...}, "required": [...]}}.

    Return (kandidat_dict, error). kandidat_dict punya "content" berisi
    "parts" -- tiap part bisa {"text": "..."} (jawaban akhir/penjelasan)
    atau {"functionCall": {"name": ..., "args": {...}}} (Gemini minta
    caller jalanin sebuah tool)."""
    payload = {
        "contents": contents,
        "tools": [{"functionDeclarations": deklarasi_tools}],
    }
    return _panggil_generate(payload, model=model, timeout=timeout, percobaan_503_per_model=percobaan_503_per_model)



if __name__ == "__main__":
    for expr in ["12*5", "10/0", "3+4*2", "sqrt(4)"]:
        print(expr, "->", hitung(expr))


# Override GEMINI_API_KEY pakai key asli dari tools_privat.py kalau file itu
# ada (buat testing lokal) -- tools_privat.py gak ikut ter-track git, jadi
# key asli gak pernah ke-push ke GitHub.
try:
    from tools_privat import GEMINI_API_KEY as _GEMINI_API_KEY_LOKAL
    if _GEMINI_API_KEY_LOKAL and not _GEMINI_API_KEY_LOKAL.startswith("MASUKKAN_"):
        GEMINI_API_KEY = _GEMINI_API_KEY_LOKAL
except ImportError:
    pass  # tools_privat.py gak ada -- normal kalau baru clone dari GitHub
