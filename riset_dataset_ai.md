# Riset: sumber dataset/knowledge base untuk chatbot mini

Catatan hasil pencarian sumber dataset percakapan/pengetahuan bahasa Indonesia
yang bisa dipakai untuk memperluas `data/percakapan.json` pada proyek
chatbot transformer buatan sendiri ini. Ditulis supaya bisa dicek ulang atau
dipakai lagi nanti tanpa perlu riset ulang dari nol.

## Paling relevan (format mirip percakapan.json — pattern → response)

- **[sarahlintang/Simple-Deep-Learning-Indonesia-Chatbot — intents.json](https://github.com/sarahlintang/Simple-Deep-Learning-Indonesia-Chatbot/blob/main/intents.json)**
  16 intent (salam, pamit, apa kabar, perkenalan, salam islami, dll), total
  sekitar 60+ variasi kalimat pattern dengan beberapa pilihan response per
  intent.
  ⚠️ **Repo ini tidak punya file LICENSE.** Artinya secara hukum belum jelas
  boleh dipakai ulang bebas (default hak cipta ada di pemiliknya). Aman untuk
  dipelajari gaya/pola kalimatnya, tapi jangan copy-paste isi datanya mentah
  ke proyek ini tanpa izin dari pembuatnya. Data tambahan yang sudah
  dimasukkan ke `percakapan.json` ditulis ulang dengan kalimat sendiri,
  bukan hasil salin dari repo ini.

## Kumpulan resource NLP Indonesia (hub, bukan satu dataset siap pakai)

- **[louisowen6/NLP_bahasa_resources](https://github.com/louisowen6/NLP_bahasa_resources)**
  Daftar terkurasi dataset & library NLP Bahasa Indonesia (korpus, sentimen,
  dsb). Berguna kalau nanti mau nambah topik di luar obrolan santai.
- **[Wikidepia/indonesian_datasets](https://github.com/Wikidepia/indonesian_datasets)**
  Kumpulan dataset NLP Indonesia lain, ukurannya besar-besar, perlu difilter
  dulu kalau mau dipakai.
- **[indonlp/indonlu (Hugging Face)](https://huggingface.co/datasets/indonlp/indonlu)**
  Benchmark NLP Indonesia resmi (IndoNLU), tapi formatnya untuk klasifikasi
  teks/NER, bukan pasangan tanya-jawab langsung — butuh konversi format kalau
  mau dipakai di chatbot ini.

## Kurang cocok untuk model kecil ini

- **[binsarjr/chatbot-indonesia](https://github.com/binsarjr/chatbot-indonesia)**
  Isinya campuran (daftar singkatan, data sentimen), bukan pasangan obrolan
  bersih siap pakai.
- **[DwikiWitman/Chatbot-Bahasa-Indonesia](https://github.com/DwikiWitman/Chatbot-Bahasa-Indonesia)**
  Memakai dataset OpenSubtitles2018 (jutaan baris subtitle film) — terlalu
  besar dan berisik untuk model transformer kecil yang tokenisasinya per-kata
  seperti punya kita (vocab bisa meledak dan makna kalimatnya sering
  terpotong/ambigu karena berasal dari dialog film).

## Kesimpulan & yang sudah dilakukan

Karena model di proyek ini kecil (tokenisasi per-kata, vocab ratusan kata,
tanpa BPE/subword) dan dataset publik yang ditemukan rata-rata punya masalah
lisensi tidak jelas atau format/skala yang tidak cocok, langkah yang diambil:

1. Tidak mengimpor dataset orang lain secara mentah ke proyek ini.
2. `data/percakapan.json` ditambah dengan pasangan tanya-jawab baru yang
   ditulis sendiri (bukan hasil salin), terinspirasi dari kategori umum yang
   biasa ada di dataset chatbot (salam, kabar, identitas bot, perasaan,
   pengetahuan umum sederhana, matematika dasar, dll) — total sekarang 313
   pasang (dari 195 sebelumnya), vocab ~570 kata.
3. Kalau suatu saat mau pakai dataset di atas secara resmi, hubungi/lihat
   izin pemiliknya dulu (terutama yang belum ada file LICENSE-nya), atau
   pakai yang sudah jelas open (mis. IndoNLU) dengan proses konversi format.
