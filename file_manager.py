"""Fitur kelola folder buat Akai: rapihin file, cari file, cari duplikat.

Semua fungsi yang NGUBAH isi disk (rapihin, hapus duplikat) WAJIB dicek
dulu lewat folder_diizinkan() sebelum jalan -- ini lapisan perizinan biar
Akai gak bisa ngutak-atik folder mana pun di komputer selain yang udah
diizinkan user secara eksplisit (lewat perintah /izinkan).

File ini sengaja gak punya print()/input() sama sekali -- murni logika,
biar gampang dites dan gak nyampur sama tampilan chat di pytorch_chatbot.py.
"""

import hashlib
import os
import shutil
import subprocess
import time


KATEGORI_DEFAULT = {
    "gambar": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico", ".heic"],
    "dokumen": [".pdf", ".doc", ".docx", ".txt", ".xlsx", ".xls", ".ppt", ".pptx", ".csv", ".odt", ".rtf"],
    "video": [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"],
    "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"],
    "arsip": [".zip", ".rar", ".7z", ".tar", ".gz"],
    "program": [".exe", ".msi", ".apk", ".bat"],
}


def folder_diizinkan(path, allowed_folders):
    """True kalau `path` ada di dalam salah satu folder yang diizinkan user."""
    if not path or not allowed_folders:
        return False
    try:
        target = os.path.realpath(path)
    except OSError:
        return False
    for folder in allowed_folders:
        try:
            base = os.path.realpath(folder)
        except OSError:
            continue
        if target == base or target.startswith(base + os.sep):
            return True
    return False


def _kategori_untuk(ext, kategori_map):
    ext = ext.lower()
    for nama, daftar_ext in kategori_map.items():
        if ext in daftar_ext:
            return nama
    return "lainnya"


def rencana_rapihin(folder, kategori_map=None):
    """Scan file yang LANGSUNG ada di `folder` (gak masuk subfolder), bikin
    rencana pemindahan tiap file ke subfolder sesuai kategorinya.

    Return list of dict: {"nama", "asal", "kategori", "tujuan"}.
    Belum mindahin apa-apa -- ini cuma rencana, dieksekusi lewat
    jalankan_rapihin().
    """
    kategori_map = kategori_map or KATEGORI_DEFAULT
    rencana = []
    for nama in sorted(os.listdir(folder)):
        asal = os.path.join(folder, nama)
        if not os.path.isfile(asal):
            continue
        _, ext = os.path.splitext(nama)
        kategori = _kategori_untuk(ext, kategori_map)
        tujuan = os.path.join(folder, kategori, nama)
        rencana.append({"nama": nama, "asal": asal, "kategori": kategori, "tujuan": tujuan})
    return rencana


def ringkas_rencana(rencana):
    """Kelompokin rencana per kategori buat ditampilin ringkas.
    Return dict {kategori: jumlah_file}."""
    ringkasan = {}
    for item in rencana:
        ringkasan[item["kategori"]] = ringkasan.get(item["kategori"], 0) + 1
    return ringkasan


def jalankan_rapihin(rencana):
    """Eksekusi rencana dari rencana_rapihin(). Gak pernah menimpa file yang
    udah ada -- kalau nama bentrok di folder tujuan, ditambahin _1, _2, dst.
    Return (jumlah_sukses, list_pesan_error)."""
    sukses = 0
    error = []
    for item in rencana:
        asal, tujuan, nama = item["asal"], item["tujuan"], item["nama"]
        try:
            os.makedirs(os.path.dirname(tujuan), exist_ok=True)
            tujuan_final = tujuan
            if os.path.exists(tujuan_final):
                base, ext = os.path.splitext(nama)
                i = 1
                folder_tujuan = os.path.dirname(tujuan)
                while os.path.exists(tujuan_final):
                    tujuan_final = os.path.join(folder_tujuan, f"{base}_{i}{ext}")
                    i += 1
            shutil.move(asal, tujuan_final)
            sukses += 1
        except OSError as e:
            error.append(f"{nama}: {e}")
    return sukses, error


def cari_file(keyword, allowed_folders, maksimal=30):
    """Cari file yang namanya mengandung `keyword` (case-insensitive) di
    semua folder yang diizinkan, rekursif ke dalam subfolder.
    Return list of (path, ukuran_bytes), maksimal `maksimal` hasil."""
    keyword = (keyword or "").lower().strip()
    hasil = []
    if not keyword:
        return hasil
    for folder in allowed_folders or []:
        if not os.path.isdir(folder):
            continue
        for root, _dirs, files in os.walk(folder):
            for nama in files:
                if keyword in nama.lower():
                    path = os.path.join(root, nama)
                    try:
                        ukuran = os.path.getsize(path)
                    except OSError:
                        ukuran = 0
                    hasil.append((path, ukuran))
                    if len(hasil) >= maksimal:
                        return hasil
    return hasil


def _hash_file(path, blok=65536):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                data = f.read(blok)
                if not data:
                    break
                h.update(data)
        return h.hexdigest()
    except OSError:
        return None


def cari_duplikat(folder, minimal_ukuran=1024):
    """Cari file yang isinya identik persis di dalam `folder` (rekursif).
    Return list of list[path] -- tiap grup isinya path2 file yang isinya
    sama persis (cuma grup dengan lebih dari 1 anggota yang di-return)."""
    ukuran_map = {}
    for root, _dirs, files in os.walk(folder):
        for nama in files:
            path = os.path.join(root, nama)
            try:
                ukuran = os.path.getsize(path)
            except OSError:
                continue
            if ukuran < minimal_ukuran:
                continue
            ukuran_map.setdefault(ukuran, []).append(path)

    grup_duplikat = []
    for _ukuran, paths in ukuran_map.items():
        if len(paths) < 2:
            continue
        hash_map = {}
        for path in paths:
            h = _hash_file(path)
            if h:
                hash_map.setdefault(h, []).append(path)
        for group in hash_map.values():
            if len(group) > 1:
                grup_duplikat.append(group)
    return grup_duplikat


def format_ukuran(ukuran_bytes):
    """Format ukuran file jadi string gampang dibaca (KB/MB/GB)."""
    ukuran = float(ukuran_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if ukuran < 1024 or unit == "GB":
            return f"{ukuran:.1f}{unit}" if unit != "B" else f"{int(ukuran)}{unit}"
        ukuran /= 1024
    return f"{ukuran:.1f}GB"


EKSTENSI_SAMPAH = (
    ".tmp", ".temp", ".log", ".bak", ".old", ".crdownload", ".part", ".cache",
)

_POLA_NAMA_SAMPAH = ("cache",)
_PREFIX_NAMA_SAMPAH = ("~$",)


def _kenapa_gak_kepake(path, nama, hari_lama):
    """Cek satu file, return alasan (string) kalau dianggap gak kepake,
    atau None kalau file ini masih dianggap kepake."""
    ext = os.path.splitext(nama)[1].lower()
    nama_lower = nama.lower()

    if ext in EKSTENSI_SAMPAH:
        return f"file sampah/sementara (ekstensi {ext})"
    if nama_lower.startswith(_PREFIX_NAMA_SAMPAH):
        return "file sampah/sementara (file temporer aplikasi)"
    for pola in _POLA_NAMA_SAMPAH:
        if pola in nama_lower:
            return f"file sampah/sementara (nama mengandung '{pola}')"

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    umur_hari = (time.time() - mtime) / 86400
    if umur_hari >= hari_lama:
        return f"file lama, gak disentuh {int(umur_hari)} hari"
    return None


def cari_file_gak_kepake(folder, hari_lama=180, minimal_ukuran=0):
    """Scan rekursif `folder`, cari file yang dianggap 'gak kepake':
    - file lama yang gak disentuh (mtime lebih tua dari `hari_lama` hari), ATAU
    - file sampah/sementara (ekstensi/nama khas file temporer).

    TIDAK termasuk file duplikat -- itu urusan cari_duplikat()/`/duplikat`.

    Return list of dict: {"path", "ukuran", "alasan"}.
    Ini cuma nyari & ngelaporin, belum ngehapus apa pun."""
    hasil = []
    if not os.path.isdir(folder):
        return hasil
    for root, _dirs, files in os.walk(folder):
        for nama in files:
            path = os.path.join(root, nama)
            try:
                ukuran = os.path.getsize(path)
            except OSError:
                continue
            if ukuran < minimal_ukuran:
                continue
            alasan = _kenapa_gak_kepake(path, nama, hari_lama)
            if alasan:
                hasil.append({"path": path, "ukuran": ukuran, "alasan": alasan})
    return hasil


def baca_isi_file(path, batas_karakter=20000):
    """Baca isi file teks buat dikasih ke AI (mode agen coding). Kalau
    kepanjangan, dipotong sampai `batas_karakter` biar gak boros token.
    Return (isi, dipotong) -- dipotong=True kalau isi asli lebih panjang
    dari batas. Return (None, None) kalau gagal baca."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            isi = f.read()
    except OSError:
        return None, None
    if len(isi) > batas_karakter:
        return isi[:batas_karakter], True
    return isi, False


def tulis_isi_file(path, isi_baru, backup=True):
    """Tulis/timpa `path` dengan `isi_baru`. Kalau file udah ada sebelumnya
    dan backup=True, bikin salinan `.bak` DULU sebelum ditimpa -- biar isi
    lama gak ilang kalau AI-nya salah nulis.
    Return (sukses, info): sukses True -> info = path backup (atau None
    kalau ini file baru, gak ada yang di-backup); sukses False -> info =
    pesan error."""
    backup_path = None
    try:
        if backup and os.path.exists(path):
            backup_path = path + ".bak"
            shutil.copy2(path, backup_path)
        folder_induk = os.path.dirname(path)
        if folder_induk:
            os.makedirs(folder_induk, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(isi_baru)
        return True, backup_path
    except OSError as e:
        return False, str(e)


def jalankan_command_agen(command, cwd, timeout=60):
    """Jalanin 1 command shell di dalam folder `cwd` (dipakai mode agen
    coding). Output stdout/stderr dipotong biar gak boros token kalau
    kepanjangan.
    Return dict {"exit_code", "stdout", "stderr"} -- exit_code None kalau
    timeout atau gagal dijalanin sama sekali."""
    try:
        hasil = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "exit_code": hasil.returncode,
            "stdout": hasil.stdout[-4000:] if hasil.stdout else "",
            "stderr": hasil.stderr[-4000:] if hasil.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "stdout": "", "stderr": f"command timeout setelah {timeout} detik"}
    except OSError as e:
        return {"exit_code": None, "stdout": "", "stderr": str(e)}



def hapus_file_list(daftar):
    """Hapus semua file di `daftar` (list of path, atau list of dict yang
    punya key "path"). Return (jumlah_sukses, list_pesan_error)."""
    sukses = 0
    error = []
    for item in daftar:
        path = item["path"] if isinstance(item, dict) else item
        try:
            os.remove(path)
            sukses += 1
        except OSError as e:
            error.append(f"{path}: {e}")
    return sukses, error
