# 🤖 Akai — Chatbot Transformer Bikinan Sendiri (dari NOL, gak pake library siap pakai)

> Iya bener, dari nol. Bukan fine-tune GPT, bukan panggil API doang terus diklaim "bikin AI sendiri". Ini beneran neuron, backprop, attention mechanism ditulis manual biar ngerti gimana cara kerja AI dari akarnya — baru abis itu dirakit jadi chatbot yang bisa dipake sehari-hari.

Proyek solo iseng-iseng belajar deep learning sambil bikin sesuatu yang beneran kepake: chatbot bahasa Indonesia bernama **Akai**, jalan di terminal, punya otak transformer sendiri, plus segudang fitur tambahan (tools, integrasi Gemini, manajemen file, sampe mode agen coding otonom kayak Claude Code / Gemini CLI).

---

## ✨ Apa aja yang bisa dilakuin Akai

### 🧠 Otak sendiri
Model transformer (`nn.TransformerEncoder`) yang di-training dari dataset percakapan sendiri (`data/percakapan.json`), tokenisasi word-level custom, config-driven lewat `config.json` — bukan nyomot model orang.

### 💬 Chat gak gampang nyerah
Kalau model lokal masih bego / gak yakin jawabannya, otomatis lempar ke backup:
1. Model lokal duluan
2. Kalau gak yakin → pollinations.ai
3. Kalau itu juga gagal → Gemini API

Jadi Akai jarang banget bilang "gak ngerti" doang.

### 🛠️ Tools bawaan
- `/hitung` — kalkulator ekspresi matematika
- `/todo` — to-do list (tambah/list/selesai/hapus)
- `/cuaca <kota>` — cuaca real-time
- `/sholat <kota>` — jadwal sholat
- `/waktu <kota>` — jam di kota lain
- `/terjemah <teks>` — translate
- `/kurs <dari> <ke> [jumlah]` — kurs mata uang
- `/wiki <topik>` — ringkasan Wikipedia
- `/alquran [surat:ayat]` — ambil ayat Al-Qur'an
- `/hadits [kitab] [nomor]` — ambil hadits (Bukhari, Muslim, Tirmidzi, dll)
- `/gambar <deskripsi>` — generate gambar AI (via pollinations)
- `/tanyaai <pertanyaan>` — tanya AI backup langsung

### ✨ Integrasi Gemini
- `/tanyaaigemini [pertanyaan]` — masuk mode ngobrol nonstop sama Gemini sampe ketik `/stopgemini`
- Otomatis rotasi model kalau satu model lagi sibuk (503) atau kena limit kuota (429) — gak nyerah cuma karena 1 model penuh
- **Grounding with Google Search** otomatis nyala pas emang butuh info terbaru (berita, cuaca, harga, dll) — biar kuota gratisnya gak boros dipake basa-basi doang

### 📁 Manajemen folder (dengan sistem izin!)
Akai gak bisa asal utak-atik file di komputer kamu. Harus di-`/izinkan` dulu folder mana aja yang boleh dia sentuh:

| Command | Fungsi |
|---|---|
| `/izinkan <path>` | Kasih izin folder ke Akai |
| `/izinkan list` | Liat folder yang udah diizinkan |
| `/cabut <path>` | Cabut izin folder |
| `/rapihin <path>` | Kelompokkan file berantakan ke subfolder per tipe |
| `/cari <kata kunci>` | Cari file di semua folder yang diizinkan |
| `/duplikat <path>` | Cari & hapus file kembar (isinya identik) |
| `/beresin` | Rapihin **SEMUA** folder yang diizinkan sekaligus + laporan duplikat, cuma 1 konfirmasi |
| `/bersihkan <path> [hari]` | Hapus file lama/sampah (`.tmp`, `.log`, dll) otomatis, 1 konfirmasi buat semua |
| `/editfile <path file>` | Gemini nulis ulang isi file sesuai instruksi kamu, backup `.bak` otomatis |

Bonus: semua ini juga bisa dipicu pakai bahasa natural kalau Gemini aktif — tinggal ngomong biasa kayak *"tolong rapihin folder download aku"*, Gemini yang nerjemahin ke command yang bener (tapi tetep lewat sistem izin & konfirmasi yang sama, gak ada jalan pintas).

### 🕶️ Mode Agen Coding (kayak Claude Code / Gemini CLI, tapi versi Akai)
```
/agen <path folder proyek> <instruksi tugas>
```
Kasih instruksi, Akai bakal:
1. Baca file yang relevan sendiri
2. Mikir & rencana
3. Minta izin (y/n) tiap kali mau nulis/edit file atau jalanin command terminal
4. Ulang sampe tugas kelar (max 15 langkah biar gak muter-muter)

Semua tetap dibatasin folder yang udah `/izinkan`, dan setiap file yang ditimpa otomatis di-backup `.bak` dulu.

---

## 🧗 Perjalanan belajarnya (level 0 → 4)

File-file `level*.py` itu bukan sampah, itu jejak belajar dari nol sebelum nyampe ke transformer beneran:

| File | Isinya |
|---|---|
| `level0_neuron.py` | 1 neuron doang, belajar `y = 2x + 1` |
| `level1_xor.py` | Neural network kecil belajar XOR |
| `level2_word_embedding.py` | Prediksi kata berikutnya pake word embedding |
| `level3_attention.py` | Demo attention mechanism ditulis manual |
| `level4_transformer.py` | Mini transformer PAKE NUMPY DOANG, gak pake pytorch |
| `pytorch_chatbot.py` | Versi produksi — transformer beneran pake PyTorch + semua fitur di atas |

Kalau penasaran gimana attention/transformer kerja tanpa baca 100 paper dulu, urut aja baca dari `level0` sampe `level4`.

---

## 🚀 Cara jalanin

### 1. Clone & masuk folder
```bash
git clone <url-repo-kamu>
cd chatbot-from-scratch
```

### 2. Bikin virtual environment & install dependency
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# atau: source venv/bin/activate   (Linux/Mac)

pip install torch numpy
```
> Butuh internet buat fitur tools (cuaca, kurs, Gemini, dll) — kalau offline, chat model lokal tetep jalan.

### 3. Siapin config
`config.json` udah ada default-nya, tinggal disesuain kalau perlu (path dataset, folder yang mau diizinkan Akai, dll).

### 4. (Opsional tapi direkomendasiin) Aktifin fitur Gemini
```bash
copy tools_privat.example.py tools_privat.py    # Windows
# atau: cp tools_privat.example.py tools_privat.py
```
Buka `tools_privat.py`, ganti `MASUKKAN_API_KEY_GEMINI_KAMU_DI_SINI` sama API key asli kamu. Dapetin gratis di [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

> ⚠️ `tools_privat.py` udah masuk `.gitignore` — JANGAN pernah hapus baris itu, dan jangan pernah paste API key asli ke file lain yang bakal di-push ke GitHub. Semua fitur/logic Gemini udah aman ada di `tools_publik.py` (isinya cuma placeholder), jadi aman di-share ke publik apa adanya.

### 5. Training (otomatis kalau model belum ada)
```bash
python pytorch_chatbot.py
```
Pertama kali jalan, kalau `chatbot_model.pt` belum ada, dia bakal training dulu pake dataset di `data/percakapan.json`. Abis itu langsung masuk mode chat.

Opsi tambahan:
```bash
python pytorch_chatbot.py --retrain              # paksa training ulang
python pytorch_chatbot.py --epochs 500 --lr 0.001
python pytorch_chatbot.py --temperature 0.9      # lebih random/kreatif
python pytorch_chatbot.py --no-memory            # gak nyimpen riwayat chat
```

### 6. Ngobrol
```
Kamu: halo akai
Bot : halo juga! ada yang bisa dibantu?
```
Ketik `exit`, `quit`, atau `keluar` buat berhenti.

---

## 📂 Struktur project

```
chatbot-from-scratch/
├── pytorch_chatbot.py       # otak utama + chat loop + semua fitur
├── tokenizer.py             # tokenisasi word-level custom
├── file_manager.py          # logika manajemen file (murni logic, no I/O interaktif)
├── tools_publik.py          # semua tools + integrasi Gemini (AMAN di-push, placeholder key)
├── tools_privat.py          # cuma API key asli (GITIGNORED, jangan di-push!)
├── tools_privat.example.py  # template buat bikin tools_privat.py sendiri
├── telegram_bot.py          # eksperimen bot Telegram (belum aktif dipakai)
├── config.json              # semua konfigurasi (model, training, tools, izin folder)
├── data/percakapan.json     # dataset training
├── level0_neuron.py ... level4_transformer.py   # jejak belajar dari nol
└── logs/, gambar/           # hasil runtime (riwayat chat, gambar generated)
```

---

## 🔒 Soal keamanan

- **Jangan pernah** commit `tools_privat.py` — udah di-`.gitignore`, tapi tetep double-check sebelum `git push` kalau paranoid.
- API key Gemini di `tools_privat.py` cuma buat testing lokal kamu sendiri.
- Fitur manajemen folder & mode agen (`/agen`) **selalu** dibatesin folder yang udah eksplisit di-`/izinkan` — gak ada cara buat AI-nya nyolong akses ke folder lain, termasuk lewat trik path (`../../dst`) yang udah dicek & diblok.
- Command yang ngubah/hapus/jalanin sesuatu **selalu** minta konfirmasi dulu (kecuali `/beresin` & `/bersihkan` yang emang sengaja dibikin 1-konfirmasi-buat-semua biar gak capek nge-`y` satu-satu, tapi tetep nunjukin dulu apa yang bakal kejadian).

---

## 🗺️ Roadmap / masih pengen ditambahin

- [ ] Fine-tune dataset biar model lokal makin gak plonga-plongo
- [ ] Web UI (biar gak cuma CLI)
- [ ] Voice mode?
- [ ] Aktifin lagi integrasi Telegram (`telegram_bot.py` udah setengah jadi, tinggal disambung)

PR & ide selalu diterima (kalau ini udah publik ya 😄).

---

## 📜 Lisensi

Belum dipikirin serius, anggep aja MIT — pake, modif, share bebas, tapi jangan ngaku-ngaku bikin dari nol kalau cuma nyomot 😝

---

<p align="center">Dibikin dengan kopi, keringat, dan banyak error 503 dari Gemini API 🙃</p>
