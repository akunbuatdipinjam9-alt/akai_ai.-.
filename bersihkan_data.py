import json
import re

# ============ LOAD BACKUP ============
with open("data/percakapan_backup.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Data awal (backup): {len(data)}")

# ============ FILTER ============
def valid(item):
    """Return True kalau item layak disimpan."""
    p, j = item[0].strip(), item[1].strip()
    
    if not p or not j:
        return False
    
    # Buang kalau SALAH SATU mengandung URL panjang / domain
    if re.search(r'\b\w+\.(com|id|net|org|io|co)\b', p + ' ' + j):
        return False
    
    # Buang kalau ada kata super panjang (> 25 karakter)
    for teks in [p, j]:
        for kata in teks.split():
            if len(kata) > 25:
                return False
    
    # Buang kalau baris CUMA angka/simbol (bukan kalimat)
    # Contoh: "1 + 1 = 2", "{ Frekuensi = ... }", "$15"
    if re.match(r'^[\d\s\+\-\*\/\=\{\}\(\)\$\%\.\,]+$', p):
        return False
    if re.match(r'^[\d\s\+\-\*\/\=\{\}\(\)\$\%\.\,]+$', j):
        return False
    
    # Buang kalau ada karakter aneh berlebihan
    aneh = sum(1 for c in p + j if c in '{}[]<>@#^&*~`|\\')
    if aneh > 3:
        return False
    
    return True

data_bersih = []
dibuang = []

for item in data:
    if valid(item):
        data_bersih.append([item[0].strip(), item[1].strip()])
    else:
        dibuang.append(item)

print(f"Dibuang: {len(dibuang)}")
print(f"Sisa: {len(data_bersih)}")

# ============ VOCAB STATS ============
from collections import Counter
counter = Counter()
for p, j in data_bersih:
    counter.update(p.split())
    counter.update(j.split())

print(f"\nKata unik: {len(counter)}")
print(f"Kata muncul 1x: {sum(1 for v in counter.values() if v == 1)}")
print(f"Kata muncul >=2x: {sum(1 for v in counter.values() if v >= 2)}")

# ============ SIMPAN ============
with open("data/percakapan.json", "w", encoding="utf-8") as f:
    json.dump(data_bersih, f, ensure_ascii=False, indent=2)

print(f"\nData bersih: data/percakapan.json")

# Contoh yang dibuang
print("\n=== Contoh 10 yang dibuang ===")
for item in dibuang[:10]:
    print(f"  {item[0][:60]} → {item[1][:60]}")