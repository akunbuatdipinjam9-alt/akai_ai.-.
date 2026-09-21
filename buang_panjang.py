import json
from collections import Counter

# ============ LOAD ============
with open("data/percakapan.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Data awal: {len(data)}")

# ============ FILTER ============
MAX_LEN = 32
data_bersih = []
dibuang = []

for p, j in data:
    total = len(p.split()) + len(j.split())
    if total <= MAX_LEN:
        data_bersih.append([p.strip(), j.strip()])
    else:
        dibuang.append([p, j, total])

print(f"Dibuang (> {MAX_LEN} token): {len(dibuang)}")
print(f"Sisa: {len(data_bersih)}")

# ============ VOCAB STATS ============
counter = Counter()
for p, j in data_bersih:
    counter.update(p.split())
    counter.update(j.split())

print(f"\nKata unik: {len(counter)}")
print(f"Kata muncul 1x: {sum(1 for v in counter.values() if v == 1)}")
print(f"Kata muncul >=2x: {sum(1 for v in counter.values() if v >= 2)}")

# ============ BACKUP ============
with open("data/percakapan_backup_panjang.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# ============ SIMPAN ============
with open("data/percakapan.json", "w", encoding="utf-8") as f:
    json.dump(data_bersih, f, ensure_ascii=False, indent=2)

print(f"\nData bersih: data/percakapan.json")
print(f"Backup: data/percakapan_backup_panjang.json")

# ============ CONTOH YANG DIBUANG ============
print("\n=== 10 yang dibuang ===")
for p, j, n in sorted(dibuang, key=lambda x: -x[2])[:10]:
    print(f"  [{n} token] {p[:60]}...")