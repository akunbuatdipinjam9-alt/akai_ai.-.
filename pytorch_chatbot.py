import argparse
import difflib
import json
import os
import pickle
import random
import shutil
import time
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from tokenizer import tokenize
from file_manager import (
    baca_isi_file,
    cari_duplikat,
    cari_file,
    cari_file_gak_kepake,
    folder_diizinkan,
    format_ukuran,
    hapus_file_list,
    jalankan_command_agen,
    jalankan_rapihin,
    rencana_rapihin,
    ringkas_rencana,
    tulis_isi_file,
)
from tools_publik import (
    ambil_hadits,
    ayat_quran,
    butuh_info_terbaru,
    cuaca_kota,
    format_todo_list,
    gemini_siap,
    generate_gambar,
    hitung,
    jadwal_sholat,
    kurs_mata_uang,
    panggil_gemini_agen,
    pollinations_fallback,
    tanya_gemini,
    terjemah_teks,
    todo_hapus,
    todo_list,
    todo_selesai,
    todo_tambah,
    waktu_kota,
    wiki_ringkas,
)


DEFAULT_CONFIG = {
    "model": {
        "d_model": 96,
        "n_heads": 6,
        "n_layers": 4,
        "d_ff": 256,
        "dropout": 0.1,
        "max_len": 24,
    },
    "training": {
        "learning_rate": 0.002,
        "epochs": 800,
        "batch_size": 16,
        "log_every": 25,
        "grad_clip": 1.0,
    },
    "generation": {
        "default_temperature": 0.7,
        "max_new_tokens": 12,
    },
    "paths": {
        "data_file": "data/percakapan.json",
        "model_path": "chatbot_model.pt",
        "vocab_path": "chatbot_vocab.pkl",
    },
    "memory": {
        "enabled": True,
        "log_file": "logs/riwayat_chat.jsonl",
        "show_last": 5,
    },
    "tools": {
        "todo_file": "data/todo.json",
        "request_timeout": 10,
        "pollinations_enabled": True,
        "pollinations_models": ["openai-fast"],
        "pollinations_timeout": 15,
        "pollinations_system": "Kamu adalah Akai, asisten AI yang santai dan ramah, jawab dalam bahasa Indonesia.",
        "gambar_model": "flux",
        "gambar_width": 1024,
        "gambar_height": 1024,
        "gambar_timeout": 60,
        "gambar_folder": "gambar",
    },
    "telegram": {
        "bot_token": "MASUKKAN_TOKEN_BOT_TELEGRAM_KAMU_DI_SINI",
        "telegram_allowed_chat_ids": [],
    },
    "file_manager": {
        "allowed_folders": [],
    },
}


def load_config(path):
    """Muat config.json dan gabungkan dengan nilai default (kalau
    key tertentu tidak ada di file, dipakai nilai default)."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        for section, values in user_cfg.items():
            cfg.setdefault(section, {}).update(values)
    return cfg


def simpan_config(cfg, path):
    """Simpan `cfg` balik ke file config.json (dipakai waktu user
    /izinkan atau /cabut folder, biar izinnya nempel permanen)."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except OSError:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Chatbot transformer dari nol (bahasa Indonesia)")
    parser.add_argument("--config", default="config.json", help="Path file konfigurasi JSON")
    parser.add_argument("--data", default=None, help="Override path dataset percakapan (json)")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch training")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override ukuran batch")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature default saat chat")
    parser.add_argument("--retrain", action="store_true", help="Paksa training ulang meski model sudah tersimpan")
    parser.add_argument("--no-chat", action="store_true", help="Lewati mode chat interaktif (berguna untuk testing)")
    parser.add_argument("--no-memory", action="store_true", help="Matikan penyimpanan riwayat chat ke file untuk sesi ini")
    return parser.parse_args()


def apply_overrides(cfg, args):
    if args.data:
        cfg["paths"]["data_file"] = args.data
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.temperature is not None:
        cfg["generation"]["default_temperature"] = args.temperature
    if getattr(args, "no_memory", False):
        cfg["memory"]["enabled"] = False
    return cfg


def load_percakapan(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [tuple(pair) for pair in data]


def build_vocab(percakapan, min_freq=1):
    """Bangun vocab, buang kata yang muncul kurang dari min_freq."""
    counter = Counter()
    for p, j in percakapan:
        counter.update(tokenize(p))
        counter.update(tokenize(j))

    semua_kata = {k for k, v in counter.items() if v >= min_freq}
    semua_kata.update(["|", "<pad>", "<eos>", "<unk>"])

    kata_ke_id = {k: i for i, k in enumerate(sorted(semua_kata))}
    id_ke_kata = {i: k for k, i in kata_ke_id.items()}

    print(f"Kata unik total    : {len(counter)}")
    print(f"Kata dipakai (>={min_freq}x): {len(semua_kata)}")
    print(f"Kata dibuang       : {len(counter) - len(semua_kata) + 4}")

    return kata_ke_id, id_ke_kata


def build_samples(percakapan, kata_ke_id, max_len=None):
    unk_id = kata_ke_id.get("<unk>", kata_ke_id.get("<pad>"))
    samples = []
    dipotong = 0
    for p, j in percakapan:
        full = tokenize(p) + ["|"] + tokenize(j) + ["<eos>"]
        if max_len is not None and len(full) - 1 > max_len:
            full = full[-(max_len + 1):]
            dipotong += 1
        inp = [kata_ke_id.get(k, unk_id) for k in full[:-1]]
        tgt = [kata_ke_id.get(k, unk_id) for k in full[1:]]
        samples.append((inp, tgt))
    if dipotong:
        print(f"Peringatan: {dipotong} contoh percakapan dipotong karena melebihi max_len={max_len} (bagian awal pertanyaan yang dibuang, biar training tidak crash).")
    return samples


def make_batches(samples, batch_size, pad_id):
    idxs = list(range(len(samples)))
    random.shuffle(idxs)
    batches = []
    for i in range(0, len(idxs), batch_size):
        chunk = [samples[j] for j in idxs[i:i + batch_size]]
        inps = [torch.tensor(s[0], dtype=torch.long) for s in chunk]
        tgts = [torch.tensor(s[1], dtype=torch.long) for s in chunk]
        inp_pad = pad_sequence(inps, batch_first=True, padding_value=pad_id)
        tgt_pad = pad_sequence(tgts, batch_first=True, padding_value=pad_id)
        pad_mask = (inp_pad == pad_id)
        batches.append((inp_pad, tgt_pad, pad_mask))
    return batches


class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, model_cfg):
        super().__init__()
        self.max_len = model_cfg["max_len"]
        self.embed = nn.Embedding(vocab_size, model_cfg["d_model"])
        self.pos = nn.Embedding(self.max_len, model_cfg["d_model"])
        layer = nn.TransformerEncoderLayer(
            d_model=model_cfg["d_model"], nhead=model_cfg["n_heads"],
            dim_feedforward=model_cfg["d_ff"], dropout=model_cfg["dropout"],
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=model_cfg["n_layers"])
        self.out = nn.Linear(model_cfg["d_model"], vocab_size)

    def forward(self, x, pad_mask=None):
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.out(h)


def progress_bar(current, total, width=25):
    filled = int(width * current / total)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def generate(model, pertanyaan, kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg, temperature=None):
    model.eval()
    if temperature is None:
        temperature = gen_cfg["default_temperature"]
    max_new = gen_cfg["max_new_tokens"]
    max_len = model_cfg["max_len"]
    pad_id = kata_ke_id.get("<pad>")

    tokens = tokenize(pertanyaan) + ["|"]
    unk_id = kata_ke_id.get("<unk>", pad_id)
    ids = [kata_ke_id.get(k, unk_id) for k in tokens]

    hasil = []
    with torch.no_grad():
        for _ in range(max_new):
            konteks = ids[-max_len:]
            x = torch.tensor([konteks], dtype=torch.long, device=device)
            logits = model(x)[0, -1]

            if temperature <= 0.01:
                next_id = torch.argmax(logits).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

            next_kata = id_ke_kata[next_id]
            if next_kata in ("|", "<eos>", "<pad>"):
                break
            hasil.append(next_kata)
            ids.append(next_id)
            if len(ids) >= max_len:
                break
    return " ".join(hasil) if hasil else "(kosong)"


def train(percakapan, kata_ke_id, id_ke_kata, cfg, device):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    gen_cfg = cfg["generation"]
    paths = cfg["paths"]

    V = len(kata_ke_id)
    pad_id = kata_ke_id["<pad>"]
    samples = build_samples(percakapan, kata_ke_id, max_len=model_cfg["max_len"])
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]

    print("\n" + "=" * 70)
    print(f"Mulai training — {epochs} epoch, {len(samples)} contoh percakapan, vocab {V} kata")
    print("=" * 70)

    model = TransformerChatbot(V, model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []
    start = time.time()

    for epoch in range(epochs):
        model.train()
        batches = make_batches(samples, batch_size, pad_id)
        total_loss = 0.0

        for inp, tgt, pad_mask in batches:
            inp = inp.to(device)
            tgt = tgt.to(device)
            pad_mask = pad_mask.to(device)

            logits = model(inp, pad_mask=pad_mask)
            loss = criterion(logits.reshape(-1, V), tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(batches)
        loss_history.append(avg_loss)

        if epoch % train_cfg["log_every"] == 0 or epoch == epochs - 1:
            bar = progress_bar(epoch + 1, epochs)
            contoh = generate(model, "halo", kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg, temperature=0.01)
            print(f"Epoch {epoch:4d}/{epochs} | {bar} | Loss: {avg_loss:.4f} | '{contoh}'")

    elapsed = time.time() - start
    print("=" * 70)
    print(f"Selesai dalam {elapsed:.1f} detik")
    print("=" * 70)

    torch.save(model.state_dict(), paths["model_path"])
    with open(paths["vocab_path"], "wb") as f:
        pickle.dump({
            "kata_ke_id": kata_ke_id,
            "id_ke_kata": id_ke_kata,
            "model_cfg": model_cfg,
        }, f)
    print(f"Model disimpan: {paths['model_path']}")

    return model, loss_history


def plot_loss(loss_history):
    print("\n=== Grafik Loss ===")
    max_l = max(loss_history)
    min_l = min(loss_history)
    h = 10
    w = 50
    step = max(1, len(loss_history) // w)
    sampled = loss_history[::step][:w]

    for row in range(h, 0, -1):
        thr = min_l + (max_l - min_l) * (row / h)
        line = "".join("█" if v >= thr else " " for v in sampled)
        label = f"{thr:.3f}" if row in (h, h // 2, 1) else " " * 5
        print(f"{label:>6} | {line}")
    print("       +" + "-" * len(sampled))
    print(f"        {'awal':<20}{'akhir':>30}")


def log_percakapan(memory_cfg, user_text, bot_text):
    if not memory_cfg or not memory_cfg.get("enabled"):
        return
    log_path = memory_cfg.get("log_file", "logs/riwayat_chat.jsonl")
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    entry = {
        "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_text,
        "bot": bot_text,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"(gagal menyimpan riwayat: {e})")


def tampilkan_riwayat(memory_cfg, jumlah=None):
    if not memory_cfg or not memory_cfg.get("enabled"):
        print("Penyimpanan riwayat sedang dimatikan untuk sesi ini.")
        return

    log_path = memory_cfg.get("log_file", "logs/riwayat_chat.jsonl")
    if not os.path.exists(log_path):
        print("Belum ada riwayat obrolan tersimpan.")
        return

    if jumlah is None:
        jumlah = memory_cfg.get("show_last", 5)

    with open(log_path, "r", encoding="utf-8") as f:
        baris = [line for line in f if line.strip()]

    if not baris:
        print("Belum ada riwayat obrolan tersimpan.")
        return

    dipilih = baris[-jumlah:]
    print(f"\n--- {len(dipilih)} obrolan terakhir (dari {len(baris)} total) ---")
    for line in dipilih:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        print(f"[{entry.get('waktu', '?')}]")
        print(f"  Kamu: {entry.get('user', '')}")
        print(f"  Bot : {entry.get('bot', '')}")
    print("---\n")


def model_tidak_yakin(jawaban):
    """Deteksi apakah jawaban model lokal patut dianggap 'gak bisa jawab':
    kosong, atau isinya kebanyakan token <unk> (kata yang gak ada di vocab)."""
    if not jawaban or jawaban.strip() == "(kosong)":
        return True
    kata = jawaban.split()
    if not kata:
        return True
    jumlah_unk = sum(1 for k in kata if k == "<unk>")
    return (jumlah_unk / len(kata)) >= 0.4


def jawab_dengan_fallback(model, user_text, kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg, tools_cfg, temperature=None):
    """Jawab pakai model lokal dulu; kalau model lokal gak yakin, backup ke
    pollinations.ai (gantian beberapa AI). Kalau pollinations juga gagal
    (down/rate limit/dll) dan tools_privat.py ada API key-nya, lanjut coba
    Gemini sebagai lapis backup terakhir."""
    jawaban = generate(model, user_text, kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg, temperature=temperature)

    tools_cfg = tools_cfg or {}
    if not model_tidak_yakin(jawaban):
        return jawaban, "lokal"

    if not tools_cfg.get("pollinations_enabled", True):
        return jawaban, "lokal"

    print("Bot : (model lokal belum yakin, coba tanya AI backup...)")
    backup_models = tools_cfg.get("pollinations_models")
    timeout = tools_cfg.get("pollinations_timeout", 15)
    system = tools_cfg.get("pollinations_system")
    hasil, model_dipakai, err = pollinations_fallback(user_text, models=backup_models, timeout=timeout, system=system)

    if hasil:
        return hasil, model_dipakai

    if gemini_siap():
        print(f"Bot : (pollinations gagal ({err}), coba Gemini...)")
        hasil_gemini, err_gemini = tanya_gemini(user_text, system=system, gunakan_search=butuh_info_terbaru(user_text))
        if hasil_gemini:
            return hasil_gemini, "gemini"
        return jawaban, f"lokal (pollinations gagal: {err}; gemini gagal: {err_gemini})"

    return jawaban, "lokal (backup gagal: " + str(err) + ")"


def proses_perintah_tools(user, tools_cfg):
    """Cek apakah `user` adalah salah satu perintah tools (/hitung, /todo,
    /cuaca, dst). Gak print/input apa-apa di sini (biar bisa dipakai bareng
    dari CLI maupun bot Telegram) — cuma balikin hasilnya.
    Return (ditangani, teks_balasan, path_gambar):
      - ditangani: True kalau ini memang perintah tools (walau gagal/salah format)
      - teks_balasan: teks yang mau ditampilkan/dikirim ke user (bisa None)
      - path_gambar: path file gambar hasil /gambar (cuma diisi kalau berhasil)
    """
    tools_cfg = tools_cfg or {}
    todo_path = tools_cfg.get("todo_file", "data/todo.json")
    timeout = tools_cfg.get("request_timeout", 10)

    if user.startswith("/hitung"):
        ekspresi = user[len("/hitung"):].strip()
        if not ekspresi:
            return True, "Format: /hitung 12*5", None
        hasil, err = hitung(ekspresi)
        if err:
            return True, f"gak bisa hitung itu ({err})", None
        return True, f"{ekspresi} = {hasil}", None

    if user.startswith("/todo"):
        sisa = user[len("/todo"):].strip()
        if sisa.startswith("tambah "):
            tugas = sisa[len("tambah "):].strip()
            if tugas:
                new_id = todo_tambah(todo_path, tugas)
                return True, f"oke, tugas #{new_id} ditambahkan: {tugas}", None
            return True, "Format: /todo tambah <tugas>", None
        if sisa in ("list", ""):
            return True, format_todo_list(todo_list(todo_path)), None
        if sisa.startswith("selesai "):
            try:
                item_id = int(sisa.split()[1])
                ok = todo_selesai(todo_path, item_id)
                return True, f"tugas #{item_id} " + ("ditandai selesai" if ok else "tidak ditemukan"), None
            except (IndexError, ValueError):
                return True, "Format: /todo selesai <nomor>", None
        if sisa.startswith("hapus "):
            try:
                item_id = int(sisa.split()[1])
                ok = todo_hapus(todo_path, item_id)
                return True, f"tugas #{item_id} " + ("dihapus" if ok else "tidak ditemukan"), None
            except (IndexError, ValueError):
                return True, "Format: /todo hapus <nomor>", None
        return True, "Format: /todo tambah <tugas> | /todo list | /todo selesai <nomor> | /todo hapus <nomor>", None

    if user.startswith("/cuaca"):
        kota = user[len("/cuaca"):].strip()
        if not kota:
            return True, "Format: /cuaca <nama kota>", None
        teks, err = cuaca_kota(kota, timeout=timeout)
        return True, (f"gagal ambil cuaca ({err})" if err else teks), None

    if user.startswith("/sholat"):
        kota = user[len("/sholat"):].strip()
        if not kota:
            return True, "Format: /sholat <kota> (atau /sholat <kota>,<negara>)", None
        teks, err = jadwal_sholat(kota, timeout=timeout)
        return True, (f"gagal ambil jadwal sholat ({err})" if err else teks), None

    if user.startswith("/waktu"):
        kota = user[len("/waktu"):].strip()
        if not kota:
            return True, "Format: /waktu <kota>", None
        teks, err = waktu_kota(kota, timeout=timeout)
        return True, (f"gagal cek waktu ({err})" if err else teks), None

    if user.startswith("/terjemah"):
        sisa = user[len("/terjemah"):].strip()
        if not sisa:
            return True, "Format: /terjemah <teks>  (default id->en)\natau: /terjemah id <teks>  (en->id)", None
        potongan = sisa.split(maxsplit=1)
        if len(potongan) == 2 and potongan[0].lower() in ("id", "en"):
            target, teks_input = potongan[0].lower(), potongan[1]
        else:
            target, teks_input = "en", sisa
        hasil, err = terjemah_teks(teks_input, target=target, timeout=timeout)
        return True, (f"gagal menerjemahkan ({err})" if err else hasil), None

    if user.startswith("/kurs"):
        sisa = user[len("/kurs"):].strip().split()
        if len(sisa) < 2:
            return True, "Format: /kurs <dari> <ke> [jumlah]  (contoh: /kurs USD IDR 100)", None
        dari, ke = sisa[0], sisa[1]
        try:
            jumlah = float(sisa[2].replace(",", ".")) if len(sisa) > 2 else 1.0
        except ValueError:
            return True, "Format: /kurs <dari> <ke> [jumlah]  (contoh: /kurs USD IDR 100)", None
        teks, err = kurs_mata_uang(dari, ke, jumlah=jumlah, timeout=timeout)
        return True, (f"gagal cek kurs ({err})" if err else teks), None

    if user.startswith("/wiki"):
        topik = user[len("/wiki"):].strip()
        if not topik:
            return True, "Format: /wiki <topik>", None
        teks, err = wiki_ringkas(topik, timeout=timeout)
        return True, (f"gagal cari di Wikipedia ({err})" if err else teks), None

    if user.startswith("/alquran"):
        sisa = user[len("/alquran"):].strip()
        if ":" not in sisa:
            return True, "Format: /alquran <nomor surat>:<nomor ayat>  (contoh: /alquran 2:255)", None
        bagian = sisa.split(":", 1)
        surat_input, ayat_input = bagian[0].strip(), bagian[1].strip()
        teks, err = ayat_quran(surat_input, ayat_input, timeout=timeout)
        return True, (f"gagal ambil ayat ({err})" if err else teks), None

    if user.startswith("/hadits"):
        sisa = user[len("/hadits"):].strip().split()
        kitab = sisa[0] if len(sisa) > 0 else None
        nomor = sisa[1] if len(sisa) > 1 else None
        teks, err = ambil_hadits(kitab, nomor, timeout=timeout)
        return True, (f"gagal ambil hadits ({err})" if err else teks), None

    if user.startswith("/gambar"):
        deskripsi = user[len("/gambar"):].strip()
        if not deskripsi:
            return True, "Format: /gambar <deskripsi gambar>", None
        model_gbr = tools_cfg.get("gambar_model", "flux")
        lebar = tools_cfg.get("gambar_width", 1024)
        tinggi = tools_cfg.get("gambar_height", 1024)
        gambar_timeout = tools_cfg.get("gambar_timeout", 60)
        folder = tools_cfg.get("gambar_folder", "gambar")
        path_file, url_gambar, err = generate_gambar(
            deskripsi, model=model_gbr, width=lebar, height=tinggi, folder_simpan=folder, timeout=gambar_timeout,
        )
        if err and not path_file:
            return True, f"gagal bikin gambar ({err})", None
        teks = f"gambar disimpan di {path_file}" + (f" (catatan: {err})" if err else "")
        return True, teks, path_file

    if user.startswith("/tanyaaigemini"):
        pertanyaan = user[len("/tanyaaigemini"):].strip()
        if not pertanyaan:
            return True, "Format: /tanyaaigemini <pertanyaan>", None
        if not gemini_siap():
            return True, "Gemini belum di-setup (file tools_privat.py gak ada / belum diisi API key)", None
        system = tools_cfg.get("pollinations_system")
        hasil_gemini, err_gemini = tanya_gemini(pertanyaan, system=system, gunakan_search=butuh_info_terbaru(pertanyaan))
        if hasil_gemini:
            return True, f"[gemini] {hasil_gemini}", None
        return True, f"gagal tanya Gemini ({err_gemini})", None

    if user.startswith("/tanyaai"):
        pertanyaan = user[len("/tanyaai"):].strip()
        if not pertanyaan:
            return True, "Format: /tanyaai <pertanyaan>", None
        backup_models = tools_cfg.get("pollinations_models")
        p_timeout = tools_cfg.get("pollinations_timeout", 15)
        system = tools_cfg.get("pollinations_system")
        hasil, model_dipakai, err = pollinations_fallback(pertanyaan, models=backup_models, timeout=p_timeout, system=system)
        if hasil:
            return True, f"[{model_dipakai}] {hasil}", None
        if gemini_siap():
            hasil_gemini, err_gemini = tanya_gemini(pertanyaan, system=system, gunakan_search=butuh_info_terbaru(pertanyaan))
            if hasil_gemini:
                return True, f"[gemini] {hasil_gemini}", None
            return True, f"gagal tanya AI backup (pollinations: {err}; gemini: {err_gemini})", None
        return True, f"gagal tanya AI backup ({err})", None

    return False, None, None


def _daftar_folder_izin(cfg):
    return cfg.setdefault("file_manager", {}).setdefault("allowed_folders", [])


def tangani_perintah_folder(user, cfg, config_path):
    """Tangani perintah kelola folder (/izinkan, /cabut, /rapihin, /cari,
    /duplikat). Sengaja interaktif (print/input langsung) karena butuh
    konfirmasi user sebelum ubah/hapus file beneran di disk -- makanya
    fitur ini cuma jalan di chat CLI, bukan lewat bot Telegram.
    Return True kalau perintah ini yang nangani (chat_loop tinggal `continue`)."""

    allowed = _daftar_folder_izin(cfg)

    if user.strip().lower() in ("/izinkan", "/izinkan list"):
        if not allowed:
            print("Bot : Belum ada folder yang diizinkan. Pakai: /izinkan <path folder>\n")
        else:
            print("Bot : Folder yang diizinkan Akai kelola:")
            for f in allowed:
                print(f"      - {f}")
            print()
        return True

    if user.startswith("/izinkan "):
        target = user[len("/izinkan "):].strip().strip('"')
        if not os.path.isdir(target):
            print(f"Bot : Folder gak ditemukan: {target}\n")
            return True
        target_real = os.path.realpath(target)
        if any(os.path.realpath(f) == target_real for f in allowed):
            print(f"Bot : Folder itu udah diizinkan sebelumnya: {target_real}\n")
            return True
        konfirmasi = input(f"Izinkan Akai buka/pindah/hapus file di \'{target_real}\'? (y/n): ").strip().lower()
        if konfirmasi != "y":
            print("Bot : Oke, gak jadi.\n")
            return True
        allowed.append(target_real)
        if simpan_config(cfg, config_path):
            print(f"Bot : Sip, folder \'{target_real}\' sekarang diizinkan.\n")
        else:
            print("Bot : Izin dikasih buat sesi ini, tapi gagal nyimpen ke config.json.\n")
        return True

    if user.startswith("/cabut "):
        target = user[len("/cabut "):].strip().strip('"')
        target_real = os.path.realpath(target)
        sebelum = len(allowed)
        allowed[:] = [f for f in allowed if os.path.realpath(f) != target_real]
        if len(allowed) == sebelum:
            print(f"Bot : Folder itu emang belum ada di daftar izin: {target_real}\n")
        else:
            simpan_config(cfg, config_path)
            print(f"Bot : Izin buat folder \'{target_real}\' dicabut.\n")
        return True

    if user.startswith("/rapihin"):
        target = user[len("/rapihin"):].strip().strip('"')
        if not target:
            print("Bot : Format: /rapihin <path folder>\n")
            return True
        target_real = os.path.realpath(target)
        if not os.path.isdir(target_real):
            print(f"Bot : Folder gak ditemukan: {target_real}\n")
            return True
        if not folder_diizinkan(target_real, allowed):
            print(f"Bot : Akai belum diizinkan buka folder itu. Pakai \'/izinkan {target_real}\' dulu.\n")
            return True
        rencana = rencana_rapihin(target_real)
        if not rencana:
            print("Bot : Gak ada file buat dirapihin di situ (mungkin udah rapi atau isinya folder semua).\n")
            return True
        ringkasan = ringkas_rencana(rencana)
        print(f"Bot : Rencana rapihin \'{target_real}\' ({len(rencana)} file):")
        for kategori, jumlah in sorted(ringkasan.items(), key=lambda x: -x[1]):
            print(f"      - {jumlah} file -> {kategori}/")
        konfirmasi = input("Lanjutkan? (y/n): ").strip().lower()
        if konfirmasi != "y":
            print("Bot : Oke, gak jadi dirapihin.\n")
            return True
        sukses, error = jalankan_rapihin(rencana)
        print(f"Bot : Selesai. {sukses} file dipindah.")
        if error:
            print(f"      {len(error)} gagal, contoh: {error[0]}")
        print()
        return True

    if user.startswith("/cari "):
        keyword = user[len("/cari "):].strip()
        if not allowed:
            print("Bot : Belum ada folder yang diizinkan buat dicari. Pakai /izinkan <path folder> dulu.\n")
            return True
        hasil = cari_file(keyword, allowed)
        if not hasil:
            print(f"Bot : Gak ketemu file yang mengandung \'{keyword}\'.\n")
            return True
        print(f"Bot : Ketemu {len(hasil)} file:")
        for path, ukuran in hasil:
            print(f"      - {path} ({format_ukuran(ukuran)})")
        print()
        return True

    if user.startswith("/duplikat"):
        target = user[len("/duplikat"):].strip().strip('"')
        if not target:
            print("Bot : Format: /duplikat <path folder>\n")
            return True
        target_real = os.path.realpath(target)
        if not os.path.isdir(target_real):
            print(f"Bot : Folder gak ditemukan: {target_real}\n")
            return True
        if not folder_diizinkan(target_real, allowed):
            print(f"Bot : Akai belum diizinkan buka folder itu. Pakai \'/izinkan {target_real}\' dulu.\n")
            return True
        grup = cari_duplikat(target_real)
        if not grup:
            print("Bot : Gak ketemu file duplikat di situ.\n")
            return True
        print(f"Bot : Ketemu {len(grup)} grup file duplikat:")
        for idx, group in enumerate(grup, start=1):
            print(f"\n  Grup {idx}:")
            for i, path in enumerate(group, start=1):
                ukuran = os.path.getsize(path) if os.path.exists(path) else 0
                print(f"    {i}. {path} ({format_ukuran(ukuran)})")
            pilihan = input("  Hapus yang mana? (nomor dipisah koma, atau Enter buat skip): ").strip()
            if not pilihan:
                continue
            try:
                nomor_hapus = {int(x.strip()) for x in pilihan.split(",") if x.strip()}
            except ValueError:
                print("  Format nomor salah, grup ini di-skip.")
                continue
            for i, path in enumerate(group, start=1):
                if i in nomor_hapus:
                    try:
                        os.remove(path)
                        print(f"  Dihapus: {path}")
                    except OSError as e:
                        print(f"  Gagal hapus {path}: {e}")
        print()
        return True

    return False



def tangani_perintah_beresin(user, cfg, config_path):
    """/beresin -- rapihin SEMUA folder yang diizinkan sekaligus, plus deteksi
    duplikat per folder (dilaporin doang, GAK dihapus otomatis -- pakai
    /duplikat <folder> kalau mau hapus). Cuma SATU konfirmasi buat semua
    folder sekaligus, bukan konfirmasi per folder/per file."""
    allowed = _daftar_folder_izin(cfg)
    if not allowed:
        print("Bot : Belum ada folder yang diizinkan. Pakai /izinkan <path folder> dulu.\n")
        return True

    rencana_per_folder = {}
    duplikat_per_folder = {}
    total_file = 0
    for folder in allowed:
        if not os.path.isdir(folder):
            continue
        rencana = rencana_rapihin(folder)
        if rencana:
            rencana_per_folder[folder] = rencana
            total_file += len(rencana)
        grup = cari_duplikat(folder)
        if grup:
            duplikat_per_folder[folder] = grup

    if not rencana_per_folder and not duplikat_per_folder:
        print("Bot : Semua folder yang diizinkan udah rapi, gak ada file duplikat juga.\n")
        return True

    print(f"Bot : Ringkasan /beresin buat {len(allowed)} folder yang diizinkan:\n")
    for folder, rencana in rencana_per_folder.items():
        ringkasan = ringkas_rencana(rencana)
        print(f"  [{folder}]")
        print(f"    {len(rencana)} file bisa dirapihin:")
        for kategori, jumlah in sorted(ringkasan.items(), key=lambda x: -x[1]):
            print(f"      - {jumlah} file -> {kategori}/")
    for folder, grup in duplikat_per_folder.items():
        print(f"  [{folder}]")
        print(f"    {len(grup)} grup file duplikat ditemukan (laporan doang, pakai /duplikat {folder} buat hapus)")
    print()

    if not rencana_per_folder:
        print("Bot : (gak ada yang perlu dirapihin -- duplikat di atas cuma laporan, pakai /duplikat <folder> buat hapus)\n")
        return True

    konfirmasi = input(
        f"Rapihin semua ({total_file} file total di {len(rencana_per_folder)} folder) sekaligus? (y/n): "
    ).strip().lower()
    if konfirmasi != "y":
        print("Bot : Oke, gak jadi dirapihin.\n")
        return True

    total_sukses = 0
    total_error = 0
    for folder, rencana in rencana_per_folder.items():
        sukses, error = jalankan_rapihin(rencana)
        total_sukses += sukses
        total_error += len(error)

    print(f"Bot : Selesai beresin semua folder. Total {total_sukses} file dipindah", end="")
    print(f", {total_error} gagal." if total_error else ".")
    if duplikat_per_folder:
        total_grup = sum(len(g) for g in duplikat_per_folder.values())
        print(f"      Ditemukan juga {total_grup} grup file duplikat (belum dihapus) -- pakai /duplikat <folder> buat review & hapus.")
    print()
    return True


def tangani_perintah_bersihkan(user, cfg, config_path):
    """/bersihkan <path folder> [jumlah hari] -- cari file lama/gak disentuh
    atau file sampah/sementara, lalu hapus SEKALIGUS dengan SATU konfirmasi
    buat seluruh batch (gak nanya satu-satu per file). Duplikat TIDAK termasuk
    di sini -- itu urusan /duplikat."""
    sisa = user[len("/bersihkan"):].strip()
    if not sisa:
        print("Bot : Format: /bersihkan <path folder> [jumlah hari, default 180]\n")
        return True

    hari_lama = 180
    target = sisa
    bagian = sisa.rsplit(" ", 1)
    if len(bagian) == 2 and bagian[1].isdigit():
        target, hari_str = bagian
        hari_lama = int(hari_str)
    target = target.strip().strip('"')

    target_real = os.path.realpath(target)
    if not os.path.isdir(target_real):
        print(f"Bot : Folder gak ditemukan: {target_real}\n")
        return True

    allowed = _daftar_folder_izin(cfg)
    if not folder_diizinkan(target_real, allowed):
        print(f"Bot : Akai belum diizinkan buka folder itu. Pakai \'/izinkan {target_real}\' dulu.\n")
        return True

    daftar = cari_file_gak_kepake(target_real, hari_lama=hari_lama)
    if not daftar:
        print(
            f"Bot : Gak ketemu file gak kepake di \'{target_real}\' "
            f"(kriteria: lebih tua dari {hari_lama} hari, atau file sampah/sementara).\n"
        )
        return True

    total_ukuran = sum(item["ukuran"] for item in daftar)
    print(f"Bot : Ketemu {len(daftar)} file gak kepake di \'{target_real}\' (total {format_ukuran(total_ukuran)}):")
    for item in daftar[:50]:
        print(f"      - {item['path']} ({format_ukuran(item['ukuran'])}) -- {item['alasan']}")
    if len(daftar) > 50:
        print(f"      ... dan {len(daftar) - 50} file lainnya")
    print("\n  PERINGATAN: file yang dihapus langsung hilang dari disk (gak masuk Recycle Bin).")

    konfirmasi = input(f"Hapus SEMUA {len(daftar)} file ini sekaligus? (y/n): ").strip().lower()
    if konfirmasi != "y":
        print("Bot : Oke, gak jadi dihapus.\n")
        return True

    sukses, error = hapus_file_list(daftar)
    print(f"Bot : Selesai. {sukses} file dihapus (total {format_ukuran(total_ukuran)}).")
    if error:
        print(f"      {len(error)} gagal, contoh: {error[0]}")
    print()
    return True


def tangani_perintah_editfile(user, cfg, config_path):
    """/editfile <path file> -- Gemini nulis ulang isi file sesuai instruksi
    bahasa natural dari user. Nunjukin preview diff dulu, minta konfirmasi,
    lalu bikin backup .bak SEBELUM nimpa file aslinya."""
    if not gemini_siap():
        print("Bot : Gemini belum di-setup (tools_privat.py gak ada / API key kosong), fitur ini butuh Gemini.\n")
        return True

    target = user[len("/editfile"):].strip().strip('"')
    if not target:
        print("Bot : Format: /editfile <path file>\n")
        return True

    target_real = os.path.realpath(target)
    if not os.path.isfile(target_real):
        print(f"Bot : File gak ditemukan: {target_real}\n")
        return True

    allowed = _daftar_folder_izin(cfg)
    folder_induk = os.path.dirname(target_real)
    if not folder_diizinkan(folder_induk, allowed):
        print(f"Bot : Akai belum diizinkan buka folder itu. Pakai \'/izinkan {folder_induk}\' dulu.\n")
        return True

    try:
        with open(target_real, "r", encoding="utf-8", errors="replace") as f:
            isi_lama = f.read()
    except OSError as e:
        print(f"Bot : Gagal baca file: {e}\n")
        return True

    batas_karakter = 20000
    if len(isi_lama) > batas_karakter:
        print(f"Bot : File terlalu besar buat diedit AI ({len(isi_lama)} karakter, batas {batas_karakter}).\n")
        return True

    instruksi = input("Mau diedit gimana? (jelasin dalam bahasa biasa): ").strip()
    if not instruksi:
        print("Bot : Gak ada instruksi, dibatalin.\n")
        return True

    prompt = (
        "Kamu editor file. Di bawah ini isi lengkap sebuah file, lalu instruksi "
        "edit dari user. Tulis ULANG isi file itu LENGKAP setelah diedit sesuai "
        "instruksi. Balas HANYA isi file barunya, TANPA markdown code fence, TANPA "
        "penjelasan atau komentar tambahan apa pun.\n\n"
        f"=== ISI FILE SAAT INI ({os.path.basename(target_real)}) ===\n{isi_lama}\n"
        "=== AKHIR ISI FILE ===\n\n"
        f"Instruksi edit dari user: {instruksi}\n\n"
        "Ingat: balas cuma isi file baru, dari baris pertama sampai terakhir, gak pakai ```."
    )

    hasil, err = tanya_gemini(prompt, timeout=30)
    if not hasil:
        print(f"Bot : Gagal minta Gemini edit file ({err})\n")
        return True

    isi_baru = hasil.strip()
    if isi_baru.startswith("```"):
        isi_baru = isi_baru.strip("`")
        if "\n" in isi_baru:
            isi_baru = isi_baru.split("\n", 1)[1]
        isi_baru = isi_baru.rsplit("```", 1)[0]

    if isi_baru.strip() == isi_lama.strip():
        print("Bot : Gemini gak ngubah apa-apa (atau hasilnya sama persis). Dibatalin.\n")
        return True

    diff = list(difflib.unified_diff(
        isi_lama.splitlines(keepends=True),
        isi_baru.splitlines(keepends=True),
        fromfile=f"{os.path.basename(target_real)} (lama)",
        tofile=f"{os.path.basename(target_real)} (baru)",
    ))
    print(f"Bot : Preview perubahan buat \'{target_real}\':\n")
    for baris in diff[:200]:
        print("  " + baris.rstrip("\n"))
    if len(diff) > 200:
        print(f"  ... ({len(diff) - 200} baris diff lainnya gak ditampilin)")
    print()

    konfirmasi = input("Terapkan perubahan ini? (y/n): ").strip().lower()
    if konfirmasi != "y":
        print("Bot : Oke, gak jadi diedit.\n")
        return True

    backup_path = target_real + ".bak"
    try:
        shutil.copy2(target_real, backup_path)
    except OSError as e:
        print(f"Bot : Gagal bikin backup, edit dibatalin demi keamanan ({e}).\n")
        return True

    try:
        with open(target_real, "w", encoding="utf-8") as f:
            f.write(isi_baru)
    except OSError as e:
        print(f"Bot : Gagal nulis file baru: {e}\n")
        return True

    print(f"Bot : File berhasil diedit. Backup asli disimpan di \'{backup_path}\'.\n")
    return True


_AGEN_MAKS_LANGKAH = 15

_AGEN_TOOLS = [
    {
        "name": "baca_file",
        "description": (
            "Baca isi sebuah file teks/kode di dalam folder proyek. Pakai ini "
            "buat liat isi file SEBELUM ngedit atau buat ngerti struktur project "
            "-- jangan pernah nebak isi file tanpa baca dulu."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path file yang mau dibaca, relatif ke folder proyek (atau path lengkap kalau perlu).",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "tulis_file",
        "description": (
            "Tulis/timpa ISI LENGKAP sebuah file dengan konten baru (bikin file "
            "baru kalau belum ada). Selalu bikin backup .bak otomatis kalau file "
            "itu udah ada sebelumnya. Kirim isi lengkap file, bukan cuma bagian "
            "yang berubah."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path file yang mau ditulis, relatif ke folder proyek (atau path lengkap).",
                },
                "isi": {
                    "type": "string",
                    "description": "Isi LENGKAP file setelah ditulis, dari baris pertama sampai terakhir.",
                },
            },
            "required": ["path", "isi"],
        },
    },
    {
        "name": "jalankan_command",
        "description": (
            "Jalanin 1 command di terminal/command prompt di dalam folder "
            "proyek (misal: python script.py, pip install nama_paket, "
            "npm install, dir, git status)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command shell lengkap yang mau dijalanin.",
                }
            },
            "required": ["command"],
        },
    },
]


def _path_dalam_proyek(path, folder_proyek):
    """Resolve path (relatif/absolut) yang diminta Gemini ke path absolut,
    lalu pastikan hasilnya masih di dalam folder_proyek yang diizinkan.
    Return None kalau path-nya nyeleweng keluar folder proyek."""
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(folder_proyek, path)
    path_real = os.path.realpath(path)
    if not folder_diizinkan(path_real, [folder_proyek]):
        return None
    return path_real


def _eksekusi_tool_agen(nama, args, folder_proyek):
    """Tampilin ke user apa yang mau dilakuin Gemini, minta konfirmasi y/n
    (KECUALI buat baca_file yang cuma baca, gak ngubah apa-apa jadi gak
    perlu konfirmasi), baru eksekusi kalau di-ACC. Return dict hasil yang
    bakal dikirim balik ke Gemini sebagai functionResponse."""
    if nama == "baca_file":
        path = _path_dalam_proyek(args.get("path"), folder_proyek)
        if not path:
            return {"error": "path di luar folder proyek yang diizinkan, atau path gak valid"}
        print(f"Bot : [agen] baca file: {path}")
        isi, dipotong = baca_isi_file(path)
        if isi is None:
            return {"error": f"gagal baca file {path} (gak ada / gak bisa dibuka)"}
        hasil = {"isi": isi}
        if dipotong:
            hasil["catatan"] = "isi dipotong karena kepanjangan"
        return hasil

    if nama == "tulis_file":
        path = _path_dalam_proyek(args.get("path"), folder_proyek)
        isi_baru = args.get("isi", "")
        if not path:
            return {"error": "path di luar folder proyek yang diizinkan, atau path gak valid"}
        print(f"Bot : [agen] Gemini mau nulis/ngedit file: {path} ({len(isi_baru)} karakter)")
        konfirmasi = input("      Izinkan? (y/n): ").strip().lower()
        if konfirmasi != "y":
            print("      Ditolak.\n")
            return {"error": "user menolak aksi ini, coba cara lain atau berhenti"}
        sukses, info = tulis_isi_file(path, isi_baru, backup=True)
        if not sukses:
            return {"error": f"gagal nulis file: {info}"}
        print(f"      Ditulis. Backup: {info or '(file baru, gak ada backup)'}\n")
        return {"sukses": True, "backup": info or "tidak ada (file baru)"}

    if nama == "jalankan_command":
        command = args.get("command", "")
        print(f"Bot : [agen] Gemini mau jalanin command di \'{folder_proyek}\':")
        print(f"      $ {command}")
        konfirmasi = input("      Izinkan? (y/n): ").strip().lower()
        if konfirmasi != "y":
            print("      Ditolak.\n")
            return {"error": "user menolak aksi ini, coba cara lain atau berhenti"}
        hasil = jalankan_command_agen(command, cwd=folder_proyek, timeout=60)
        print(f"      exit_code={hasil['exit_code']}")
        if hasil["stdout"]:
            print(f"      stdout: {hasil['stdout'][:500]}")
        if hasil["stderr"]:
            print(f"      stderr: {hasil['stderr'][:500]}")
        print()
        return hasil

    return {"error": f"tool gak dikenal: {nama}"}


def tangani_perintah_agen(user, cfg, config_path):
    """/agen <path folder proyek> <instruksi> -- mode agen coding otonom:
    Gemini bisa baca file, nulis/ngedit file, dan jalanin command terminal
    sendiri (looping, function calling) buat nyelesain instruksi kamu.
    Tiap aksi yang NGUBAH sesuatu (tulis file / jalanin command) WAJIB
    dikonfirmasi user dulu (y/n) -- baca file doang gak perlu konfirmasi
    soalnya gak ngubah apa-apa. Cuma boleh nyentuh file di DALAM folder
    proyek yang udah diizinkan lewat /izinkan."""
    if not gemini_siap():
        print("Bot : Gemini belum di-setup (tools_privat.py gak ada / API key kosong), fitur ini butuh Gemini.\n")
        return True

    sisa = user[len("/agen"):].strip()
    bagian = sisa.split(" ", 1)
    if len(bagian) < 2 or not bagian[1].strip():
        print("Bot : Format: /agen <path folder proyek> <instruksi tugas>\n")
        print("      Contoh: /agen C:\\project\\bot-ku tolong tambahin komentar di setiap fungsi\n")
        return True

    target, instruksi = bagian[0].strip().strip('"'), bagian[1].strip()
    target_real = os.path.realpath(target)
    if not os.path.isdir(target_real):
        print(f"Bot : Folder gak ditemukan: {target_real}\n")
        return True

    allowed = _daftar_folder_izin(cfg)
    if not folder_diizinkan(target_real, allowed):
        print(f"Bot : Akai belum diizinkan buka folder itu. Pakai \'/izinkan {target_real}\' dulu.\n")
        return True

    system_prompt = (
        "Kamu Akai, asisten coding otonom yang kerja di dalam SATU folder "
        f"proyek: \'{target_real}\'. Kamu punya tools baca_file, tulis_file, dan "
        "jalankan_command buat nyelesain tugas dari user. Selalu pakai path "
        "RELATIF ke folder proyek itu kecuali user minta path lain secara "
        "eksplisit. Kerja step-by-step: baca dulu file yang relevan sebelum "
        "ngedit, jangan asal nebak isi file. Kalau salah satu aksi kamu "
        "ditolak user (error \'user menolak aksi ini\'), coba cara lain atau "
        "jelasin ke user kenapa kamu berhenti -- jangan ngulang aksi yang "
        "sama persis. Kalau tugas udah selesai (atau kamu udah mentok), "
        "JAWAB DENGAN TEKS BIASA (TANPA manggil tool lagi) yang ngejelasin "
        "apa yang udah/belum dilakuin -- itu tandanya kamu berhenti."
    )

    contents = [{"role": "user", "parts": [{"text": f"{system_prompt}\n\nTugas dari user: {instruksi}"}]}]

    print(f"Bot : (mode agen aktif di \'{target_real}\', mikir...)\n")

    for langkah in range(1, _AGEN_MAKS_LANGKAH + 1):
        kandidat, error = panggil_gemini_agen(contents, _AGEN_TOOLS, timeout=45)
        if kandidat is None:
            print(f"Bot : Gagal manggil Gemini ({error}), agen berhenti.\n")
            return True

        parts = kandidat.get("content", {}).get("parts") or []
        function_calls = [p["functionCall"] for p in parts if "functionCall" in p]
        teks_parts = [p.get("text", "") for p in parts if p.get("text")]

        contents.append({"role": "model", "parts": parts})

        if not function_calls:
            jawaban = "\n".join(teks_parts).strip()
            pesan_selesai = jawaban or "(gak ada penjelasan tambahan dari Gemini)"
            print(f"Bot : [agen selesai, {langkah} langkah] {pesan_selesai}\n")
            return True

        function_responses = []
        for call in function_calls:
            nama = call.get("name")
            args = call.get("args") or {}
            hasil_response = _eksekusi_tool_agen(nama, args, target_real)
            function_responses.append({"functionResponse": {"name": nama, "response": hasil_response}})

        contents.append({"role": "function", "parts": function_responses})

    print(
        f"Bot : (agen berhenti otomatis setelah {_AGEN_MAKS_LANGKAH} langkah -- "
        "kemungkinan tugasnya kepanjangan atau muter-muter. Coba pecah jadi "
        "instruksi yang lebih kecil.)\n"
    )
    return True



def tangani_perintah_lanjutan(user, cfg, config_path):
    """Command tambahan level 'agen': /beresin, /bersihkan, /editfile, /agen.
    Return True kalau perintah ini yang nangani."""
    if user.strip().lower() == "/beresin":
        return tangani_perintah_beresin(user, cfg, config_path)
    if user.startswith("/bersihkan"):
        return tangani_perintah_bersihkan(user, cfg, config_path)
    if user.startswith("/editfile"):
        return tangani_perintah_editfile(user, cfg, config_path)
    if user.startswith("/agen"):
        return tangani_perintah_agen(user, cfg, config_path)
    return False


_KATA_KUNCI_FOLDER = ("folder", "rapihin", "rapikan", "duplikat", "dupikat", "beresin", "bersihin", "bersihkan", "sampah", "gak kepake", "nggak kepake")
_KATA_KERJA_FILE = ("cari", "hapus", "bersihin", "beresin", "sortir", "pindah", "rapihin")


def _sepertinya_soal_folder(user):
    """Pre-filter murah (tanpa panggil Gemini) buat nebak apakah pesan biasa
    (bukan command /) kemungkinan soal kelola folder, biar gak asal panggil
    Gemini di tiap pesan chat biasa."""
    teks = user.lower()
    if any(k in teks for k in _KATA_KUNCI_FOLDER):
        return True
    if "file" in teks and any(v in teks for v in _KATA_KERJA_FILE):
        return True
    return False


def _cocokkan_folder(target, allowed):
    """Cocokkan nama folder yang disebut user (bisa cuma nama pendek kayak
    'download') ke salah satu folder yang udah diizinkan (folder_diizinkan),
    atau ke path asli kalau target udah berupa path folder yang valid."""
    if not target:
        return None
    if os.path.isdir(target):
        return os.path.realpath(target)
    target_lower = target.lower()
    for f in allowed:
        basename = os.path.basename(f.rstrip("\\/")).lower()
        if basename == target_lower or basename in target_lower or target_lower in basename:
            return f
    return None


def tangani_perintah_folder_alami(user, cfg, config_path):
    """Coba pahami perintah kelola folder dalam bahasa natural pakai Gemini
    (misal: 'tolong rapihin folder download aku', 'cariin file laporan'),
    lalu delegasikan ke tangani_perintah_folder() yang udah ada -- jadi tetap
    minta konfirmasi user sebelum eksekusi beneran, izin folder juga tetap
    dicek sama seperti command /rapihin dkk. Return True kalau berhasil
    dipahami & ditangani sebagai perintah folder."""
    if not gemini_siap():
        return False

    allowed = _daftar_folder_izin(cfg)
    daftar_teks = "\n".join(f"- {f}" for f in allowed) if allowed else "(belum ada folder yang diizinkan)"

    prompt = (
        "Kamu parser perintah buat asisten pengelola folder. Ada 5 aksi:\n"
        "- rapihin: kelompokkan file dalam 1 folder ke subfolder per tipe\n"
        "- cari: cari file berdasar kata kunci nama file\n"
        "- duplikat: cari file yang isinya identik dalam 1 folder\n"
        "- beresin_semua: rapihin SEMUA folder yang diizinkan sekaligus (gak butuh target folder spesifik)\n"
        "- bersihkan: hapus file lama/gak kepake atau file sampah/sementara dalam 1 folder\n\n"
        f"Folder yang sudah diizinkan user:\n{daftar_teks}\n\n"
        "Balas HANYA JSON, tanpa teks lain, format persis:\n"
        '{"aksi": "rapihin/cari/duplikat/beresin_semua/bersihkan/tidak_relevan", "target": "..."}\n\n'
        'Kalau pesan user BUKAN permintaan kelola folder, aksi harus "tidak_relevan".\n'
        "Kalau aksi rapihin/duplikat/bersihkan, target isinya path/nama folder yang dimaksud "
        "(cocokkan ke salah satu folder yang diizinkan di atas kalau user nyebut nama pendek).\n"
        "Kalau aksi cari, target isinya kata kunci pencarian.\n"
        "Kalau aksi beresin_semua, target boleh dikosongin (\"\").\n\n"
        f'Pesan user: "{user}"'
    )

    hasil, _err = tanya_gemini(prompt, timeout=15)
    if not hasil:
        return False

    teks = hasil.strip()
    if teks.startswith("```"):
        teks = teks.strip("`")
        if "\n" in teks:
            teks = teks.split("\n", 1)[1]
        teks = teks.rsplit("```", 1)[0].strip()

    try:
        data = json.loads(teks)
    except (ValueError, TypeError):
        return False

    aksi = str(data.get("aksi") or "").strip().lower()
    target = str(data.get("target") or "").strip()

    if aksi not in ("rapihin", "cari", "duplikat", "beresin_semua", "bersihkan"):
        return False

    if aksi == "cari":
        if not target:
            return False
        print(f"Bot : (paham maksudnya: cari file '{target}')")
        return tangani_perintah_folder(f"/cari {target}", cfg, config_path)

    if aksi == "beresin_semua":
        print("Bot : (paham maksudnya: beresin SEMUA folder yang diizinkan)")
        return tangani_perintah_lanjutan("/beresin", cfg, config_path)

    target_resolve = _cocokkan_folder(target, allowed)
    if not target_resolve:
        print(
            f"Bot : Kayaknya kamu mau {aksi} folder, tapi aku gak yakin folder mana yang "
            f"dimaksud. Coba lebih spesifik, misal: /{aksi} <path folder lengkap>\n"
        )
        return True

    print(f"Bot : (paham maksudnya: {aksi} folder '{target_resolve}')")
    if aksi == "bersihkan":
        return tangani_perintah_lanjutan(f"/bersihkan {target_resolve}", cfg, config_path)
    return tangani_perintah_folder(f"/{aksi} {target_resolve}", cfg, config_path)


def _adalah_perintah_tools(user):
    prefiks = (
        "/hitung", "/todo", "/cuaca", "/sholat", "/waktu", "/terjemah",
        "/kurs", "/wiki", "/alquran", "/gambar", "/hadits",
        "/tanyaaigemini", "/tanyaai",
    )
    return user.startswith(prefiks)


def chat_loop(model, kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg, temperature=None, memory_cfg=None, tools_cfg=None, cfg=None, config_path=None):
    if temperature is None:
        temperature = gen_cfg["default_temperature"]

    memori_aktif = bool(memory_cfg and memory_cfg.get("enabled"))
    mode_gemini = False

    print("\n" + "=" * 70)
    print("MODE CHAT — ketik 'exit' untuk keluar")
    print(f"Temperature: {temperature} (ubah dengan /temp 0.5)")
    if memori_aktif:
        print(f"Riwayat chat disimpan ke: {memory_cfg.get('log_file', 'logs/riwayat_chat.jsonl')}")
        print("Ketik /riwayat untuk lihat obrolan sebelumnya")
    else:
        print("Riwayat chat TIDAK disimpan (memori dimatikan)")
    print("Tools: /hitung 12*5 | /todo tambah|list|selesai|hapus | /cuaca <kota> | /tanyaai <pertanyaan>")
    print("       /tanyaaigemini [pertanyaan]  (masuk mode Gemini terus-terusan sampai /stopgemini)")
    print("       /sholat <kota> | /waktu <kota> | /terjemah <teks> | /kurs <dari> <ke> [jumlah]")
    print("       /wiki <topik> | /alquran [surat:ayat] | /gambar <deskripsi>")
    print("       /hadits [kitab] [nomor]  (kitab: bukhari, muslim, tirmidzi, nasai, abu-daud, ibnu-majah, ahmad, darimi, malik)")
    print("Kelola folder: /izinkan <path> | /izinkan list | /cabut <path>")
    print("               /rapihin <path> | /cari <kata kunci> | /duplikat <path>")
    print("               /beresin  (rapihin SEMUA folder yang diizinkan + laporan duplikat)")
    print("               /bersihkan <path> [hari]  (hapus file lama/sampah, 1 konfirmasi buat semua)")
    print("               /editfile <path file>  (Gemini nulis ulang file sesuai instruksi kamu, ada backup .bak)")
    print("               /agen <path folder proyek> <instruksi>  (mode agen coding: baca/tulis file + jalanin command, tiap aksi dikonfirmasi)")
    print("               (atau ngomong biasa: \"tolong rapihin folder download aku\" -- butuh Gemini aktif)")
    print("=" * 70 + "\n")

    while True:
        try:
            user = input("Kamu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user:
            continue
        if user.lower() in ("exit", "quit", "keluar"):
            print("Bot : bye")
            break
        if user.startswith("/temp "):
            try:
                temperature = float(user.split()[1])
                print(f"Temperature → {temperature}")
            except (IndexError, ValueError):
                print("Format: /temp 0.5")
            continue
        if user.lower() in ("/riwayat", "/history"):
            tampilkan_riwayat(memory_cfg)
            continue
        if user.startswith("/riwayat "):
            try:
                n = int(user.split()[1])
                tampilkan_riwayat(memory_cfg, jumlah=n)
            except (IndexError, ValueError):
                print("Format: /riwayat 10")
            continue
        if user.lower() == "/stopgemini":
            if mode_gemini:
                mode_gemini = False
                print("Bot : Oke, balik ke mode biasa (model lokal + fallback normal).\n")
            else:
                print("Bot : Mode Gemini emang lagi gak aktif.\n")
            continue

        if user.startswith("/tanyaaigemini"):
            if not gemini_siap():
                print("Bot : Gemini belum di-setup (tools_privat.py gak ada / API key kosong)\n")
                continue
            pertanyaan = user[len("/tanyaaigemini"):].strip()
            mode_gemini = True
            if pertanyaan:
                system = (tools_cfg or {}).get("pollinations_system")
                hasil_gemini, err_gemini = tanya_gemini(pertanyaan, system=system, gunakan_search=butuh_info_terbaru(pertanyaan))
                if hasil_gemini:
                    print(f"Bot : [gemini] {hasil_gemini}\n")
                    log_percakapan(memory_cfg, pertanyaan, hasil_gemini)
                else:
                    print(f"Bot : gagal tanya Gemini ({err_gemini})\n")
            print("Bot : (mode Gemini AKTIF -- semua chat berikutnya langsung ke Gemini, ketik /stopgemini buat balik normal)\n")
            continue

        # Perintah kelola folder (eksplisit /rapihin dkk ATAU bahasa natural)
        # dicek DULUAN, sebelum mode_gemini nyantol -- biar pas lagi mode
        # Gemini aktif, "tolong rapihin folder download aku" tetap beneran
        # motong file, bukan cuma jadi obrolan basa-basi ke Gemini.
        if user.startswith((
            "/izinkan", "/cabut", "/rapihin", "/cari ", "/duplikat",
            "/beresin", "/bersihkan", "/editfile", "/agen",
        )) and cfg is not None:
            if tangani_perintah_folder(user, cfg, config_path):
                continue
            if tangani_perintah_lanjutan(user, cfg, config_path):
                continue

        if (
            not user.startswith("/")
            and cfg is not None
            and gemini_siap()
            and _sepertinya_soal_folder(user)
        ):
            if tangani_perintah_folder_alami(user, cfg, config_path):
                continue

        if mode_gemini and not user.startswith("/"):
            system = (tools_cfg or {}).get("pollinations_system")
            hasil_gemini, err_gemini = tanya_gemini(user, system=system, gunakan_search=butuh_info_terbaru(user))
            if hasil_gemini:
                print(f"Bot : [gemini] {hasil_gemini}\n")
                log_percakapan(memory_cfg, user, hasil_gemini)
            else:
                print(f"Bot : gagal tanya Gemini ({err_gemini})\n")
            continue

        if _adalah_perintah_tools(user):
            _, teks, path_gambar = proses_perintah_tools(user, tools_cfg)
            if teks:
                print(f"Bot : {teks}\n")
            if path_gambar:
                try:
                    os.startfile(os.path.abspath(path_gambar))
                except (AttributeError, OSError):
                    pass
            continue

        jawaban, sumber = jawab_dengan_fallback(model, user, kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg, tools_cfg, temperature=temperature)
        if sumber == "lokal":
            print(f"Bot : {jawaban}\n")
        else:
            print(f"Bot : [{sumber}] {jawaban}\n")
        log_percakapan(memory_cfg, user, jawaban)


def muat_model_terlatih(cfg, device):
    """Load model yang udah dilatih dari disk (dipakai bareng oleh CLI dan
    bot Telegram). Raise RuntimeError kalau model belum ada/belum dilatih."""
    paths = cfg["paths"]
    model_ready = os.path.exists(paths["model_path"]) and os.path.exists(paths["vocab_path"])
    if not model_ready:
        raise RuntimeError(
            "Model belum dilatih. Jalankan dulu 'python pytorch_chatbot.py' "
            "di terminal sampai proses training selesai, baru jalankan bot Telegram-nya."
        )

    with open(paths["vocab_path"], "rb") as f:
        saved = pickle.load(f)
    kata_ke_id = saved["kata_ke_id"]
    id_ke_kata = saved["id_ke_kata"]
    model_cfg = saved.get("model_cfg", cfg["model"])

    model = TransformerChatbot(len(kata_ke_id), model_cfg).to(device)
    model.load_state_dict(torch.load(paths["model_path"], map_location=device))
    model.eval()
    return model, kata_ke_id, id_ke_kata, model_cfg


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    paths = cfg["paths"]
    percakapan = load_percakapan(paths["data_file"])
    print(f"Jumlah percakapan: {len(percakapan)}")

    model_ready = os.path.exists(paths["model_path"]) and os.path.exists(paths["vocab_path"])
    model = None
    kata_ke_id = None
    id_ke_kata = None
    model_cfg = cfg["model"]
    loaded_ok = False

    if model_ready and not args.retrain:
        print("Model ditemukan, load dari disk...")
        try:
            with open(paths["vocab_path"], "rb") as f:
                saved = pickle.load(f)
            kata_ke_id = saved["kata_ke_id"]
            id_ke_kata = saved["id_ke_kata"]
            model_cfg = saved.get("model_cfg", cfg["model"])

            model = TransformerChatbot(len(kata_ke_id), model_cfg).to(device)
            model.load_state_dict(torch.load(paths["model_path"], map_location=device))
            print("Model berhasil di-load.\n")
            loaded_ok = True
        except Exception as e:
            print(f"Model tersimpan tidak cocok lagi ({e}), latih ulang dari nol...\n")

    if loaded_ok:
        retrain_now = False
        if not args.no_chat:
            pilihan = input("Mau training ulang? (y/n): ").strip().lower()
            retrain_now = pilihan == "y"

        if retrain_now:
            kata_ke_id, id_ke_kata = build_vocab(percakapan)
            model_cfg = cfg["model"]
            model, loss_history = train(percakapan, kata_ke_id, id_ke_kata, cfg, device)
            plot_loss(loss_history)
    else:
        print("Mulai training dari nol...")
        kata_ke_id, id_ke_kata = build_vocab(percakapan)
        model_cfg = cfg["model"]
        model, loss_history = train(percakapan, kata_ke_id, id_ke_kata, cfg, device)
        plot_loss(loss_history)

    gen_cfg = cfg["generation"]

    print("\n=== Test Cepat ===")
    for p in ["halo", "terima kasih", "siapa kamu", "bye"]:
        print(f"Kamu: {p}")
        print(f"Bot : {generate(model, p, kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg, temperature=0.01)}\n")

    if not args.no_chat:
        chat_loop(model, kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg,
                  temperature=gen_cfg["default_temperature"], memory_cfg=cfg.get("memory"),
                  tools_cfg=cfg.get("tools"), cfg=cfg, config_path=args.config)


if __name__ == "__main__":
    main()
