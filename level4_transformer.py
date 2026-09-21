import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Level 4: transformer mini murni numpy (tanpa pytorch)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate awal")
    parser.add_argument("--epochs", type=int, default=400, help="Jumlah epoch training")
    parser.add_argument("--d-model", type=int, default=16, help="Dimensi model")
    parser.add_argument("--d-ff", type=int, default=32, help="Dimensi feed-forward")
    parser.add_argument("--n-layers", type=int, default=2, help="Jumlah layer transformer")
    parser.add_argument("--max-len", type=int, default=10, help="Panjang maksimum sekuens")
    parser.add_argument("--log-every", type=int, default=50, help="Cetak progress tiap berapa epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)


def relu(x):
    return np.maximum(0, x)


def drelu(x):
    return (x > 0).astype(np.float32)


def clip(g, max_norm=1.0):
    n = np.linalg.norm(g)
    if n > max_norm:
        return g * (max_norm / (n + 1e-9))
    return g


def init(shape, scale=0.1):
    return (np.random.randn(*shape) * scale).astype(np.float32)


def build_params(V, d_model, d_ff, n_layers, max_len):
    return {
        "E": init((V, d_model)),
        "PE": init((max_len, d_model)),
        "Wq": [init((d_model, d_model)) for _ in range(n_layers)],
        "Wk": [init((d_model, d_model)) for _ in range(n_layers)],
        "Wv": [init((d_model, d_model)) for _ in range(n_layers)],
        "Wo": [init((d_model, d_model)) for _ in range(n_layers)],
        "W1": [init((d_model, d_ff)) for _ in range(n_layers)],
        "b1": [np.zeros(d_ff, dtype=np.float32) for _ in range(n_layers)],
        "W2": [init((d_ff, d_model)) for _ in range(n_layers)],
        "b2": [np.zeros(d_model, dtype=np.float32) for _ in range(n_layers)],
        "Wout": init((d_model, V)),
        "bout": np.zeros(V, dtype=np.float32),
    }


def forward(ids, p, n_layers, d_model):
    ids = np.array(ids, dtype=np.int64)
    T = len(ids)

    x0 = p["E"][ids] + p["PE"][:T]
    mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

    cache = {"ids": ids, "x0": x0, "layers": []}
    x = x0

    for li in range(n_layers):
        lc = {}
        lc["res1"] = x
        lc["xin"] = x

        Q = x @ p["Wq"][li]
        K = x @ p["Wk"][li]
        Vv = x @ p["Wv"][li]

        scores = Q @ K.T / np.sqrt(d_model) + mask
        attn = softmax(scores)
        combined = attn @ Vv
        out_attn = combined @ p["Wo"][li]

        x = lc["res1"] + out_attn
        lc["Q"], lc["K"], lc["V"], lc["attn"], lc["combined"] = Q, K, Vv, attn, combined

        lc["res2"] = x
        lc["xin_ff"] = x

        h_pre = x @ p["W1"][li] + p["b1"][li]
        h = relu(h_pre)
        out_ff = h @ p["W2"][li] + p["b2"][li]

        x = lc["res2"] + out_ff
        lc["h_pre"], lc["h"] = h_pre, h

        cache["layers"].append(lc)

    cache["x_final"] = x
    logits = x @ p["Wout"] + p["bout"]
    return logits, cache


def backward(targets, logits, cache, p, n_layers, d_model):
    T = len(targets)
    probs = softmax(logits)
    loss = -np.mean(np.log(probs[np.arange(T), targets] + 1e-9))

    g = {
        "E": np.zeros_like(p["E"]),
        "PE": np.zeros_like(p["PE"]),
        "Wq": [None] * n_layers, "Wk": [None] * n_layers,
        "Wv": [None] * n_layers, "Wo": [None] * n_layers,
        "W1": [None] * n_layers, "b1": [None] * n_layers,
        "W2": [None] * n_layers, "b2": [None] * n_layers,
    }

    dlogits = probs.copy()
    dlogits[np.arange(T), targets] -= 1
    dlogits /= T

    g["Wout"] = cache["x_final"].T @ dlogits
    g["bout"] = np.sum(dlogits, axis=0)
    dx = dlogits @ p["Wout"].T

    for li in reversed(range(n_layers)):
        lc = cache["layers"][li]

        d_res2 = dx.copy()
        g["W2"][li] = lc["h"].T @ dx
        g["b2"][li] = np.sum(dx, axis=0)

        dh = dx @ p["W2"][li].T
        dh_pre = dh * drelu(lc["h_pre"])
        g["W1"][li] = lc["xin_ff"].T @ dh_pre
        g["b1"][li] = np.sum(dh_pre, axis=0)

        dx_attn_out = dh_pre @ p["W1"][li].T
        d_res2 = d_res2 + dx_attn_out

        d_attn_out = d_res2
        g["Wo"][li] = lc["combined"].T @ d_attn_out
        d_combined = d_attn_out @ p["Wo"][li].T

        d_attn = d_combined @ lc["V"].T
        dV = lc["attn"].T @ d_combined

        d_scores = lc["attn"] * (d_attn - np.sum(d_attn * lc["attn"], axis=-1, keepdims=True))
        d_scores = d_scores / np.sqrt(d_model)

        dQ = d_scores @ lc["K"]
        dK = d_scores.T @ lc["Q"]

        g["Wq"][li] = lc["xin"].T @ dQ
        g["Wk"][li] = lc["xin"].T @ dK
        g["Wv"][li] = lc["xin"].T @ dV

        dx_in = dQ @ p["Wq"][li].T + dK @ p["Wk"][li].T + dV @ p["Wv"][li].T
        dx = d_res2 + dx_in

    for i in range(T):
        g["E"][cache["ids"][i]] += dx[i]
        g["PE"][i] = dx[i]

    return loss, g


def apply_update(p, g, lr, n_layers):
    p["E"] -= lr * clip(g["E"])
    p["PE"] -= lr * clip(g["PE"])
    p["Wout"] -= lr * clip(g["Wout"])
    p["bout"] -= lr * clip(g["bout"])
    for li in range(n_layers):
        p["Wq"][li] -= lr * clip(g["Wq"][li])
        p["Wk"][li] -= lr * clip(g["Wk"][li])
        p["Wv"][li] -= lr * clip(g["Wv"][li])
        p["Wo"][li] -= lr * clip(g["Wo"][li])
        p["W1"][li] -= lr * clip(g["W1"][li])
        p["b1"][li] -= lr * clip(g["b1"][li])
        p["W2"][li] -= lr * clip(g["W2"][li])
        p["b2"][li] -= lr * clip(g["b2"][li])


def generate(pertanyaan, kata_ke_id, id_ke_kata, p, n_layers, d_model, max_len, max_new=6):
    tokens = pertanyaan.split() + ["|"]
    ids = [kata_ke_id.get(k, 0) for k in tokens]
    hasil = []
    for _ in range(max_new):
        logits, _ = forward(ids, p, n_layers, d_model)
        probs = softmax(logits[-1])
        next_id = int(np.argmax(probs))
        next_kata = id_ke_kata[next_id]
        if next_kata == "|":
            break
        hasil.append(next_kata)
        ids.append(next_id)
        if len(ids) >= max_len:
            break
    return " ".join(hasil) if hasil else "(kosong)"


def main():
    args = parse_args()
    np.random.seed(args.seed)

    percakapan = [
        ("halo", "hai"),
        ("hai", "halo"),
        ("apa kabar", "baik"),
        ("kabar baik", "syukur"),
        ("siapa kamu", "bot"),
        ("siapa namamu", "bot"),
        ("terima kasih", "sama-sama"),
        ("makasih", "sama-sama"),
        ("selamat pagi", "pagi"),
        ("selamat malam", "malam"),
        ("kamu pintar", "terima kasih"),
        ("bisa bantu", "tentu"),
        ("tolong bantu", "siap"),
        ("sampai jumpa", "sampai jumpa"),
        ("bye", "bye"),
        ("kamu suka apa", "belajar"),
        ("hobi kamu", "belajar"),
        ("berapa umurmu", "baru lahir"),
        ("kamu dari mana", "dari kode"),
        ("asal kamu", "dari kode"),
    ]

    semua_kata = set()
    for p, j in percakapan:
        semua_kata.update(p.split())
        semua_kata.update(j.split())
    semua_kata.add("|")

    kata_ke_id = {k: i for i, k in enumerate(sorted(semua_kata))}
    id_ke_kata = {i: k for k, i in kata_ke_id.items()}
    V = len(kata_ke_id)
    print(f"Vocab size: {V}")

    samples = []
    for p, j in percakapan:
        full = p.split() + ["|"] + j.split()
        inp = [kata_ke_id[k] for k in full[:-1]]
        tgt = [kata_ke_id[k] for k in full[1:]]
        samples.append((inp, tgt))

    d_model = args.d_model
    d_ff = args.d_ff
    n_layers = args.n_layers
    max_len = args.max_len

    P = build_params(V, d_model, d_ff, n_layers, max_len)

    print("\n=== Training ===")
    for epoch in range(args.epochs):
        lr = args.lr * (0.95 ** (epoch / 50))
        total, count = 0.0, 0
        for inp, tgt in samples:
            logits, cache = forward(inp, P, n_layers, d_model)
            loss, g = backward(np.array(tgt, dtype=np.int64), logits, cache, P, n_layers, d_model)
            if np.isnan(loss) or np.isinf(loss):
                continue
            apply_update(P, g, lr, n_layers)
            total += loss
            count += 1
        if epoch % args.log_every == 0 and count > 0:
            print(f"Epoch {epoch:4d} | LR: {lr:.5f} | Loss: {total / count:.4f}")

    print("\n=== Test Chatbot ===")
    for p in ["halo", "apa kabar", "siapa kamu", "siapa namamu",
              "terima kasih", "selamat pagi", "bye", "kamu pintar"]:
        print(f"Kamu: {p}")
        print(f"Bot : {generate(p, kata_ke_id, id_ke_kata, P, n_layers, d_model, max_len)}\n")


if __name__ == "__main__":
    main()
