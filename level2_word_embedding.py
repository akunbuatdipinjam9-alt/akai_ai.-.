import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Level 2: prediksi kata berikutnya dengan word embedding")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="Jumlah epoch training")
    parser.add_argument("--embedding-dim", type=int, default=8, help="Dimensi vektor embedding tiap kata")
    parser.add_argument("--log-every", type=int, default=200, help="Cetak progress tiap berapa epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(probs, targets):
    n = probs.shape[0]
    eps = 1e-7
    return -np.mean(np.log(probs[np.arange(n), targets] + eps))


def main():
    args = parse_args()
    np.random.seed(args.seed)

    kalimat = [
        "saya suka makan nasi",
        "saya suka makan mie",
        "saya suka minum kopi",
        "saya suka minum teh",
        "kamu suka makan nasi",
        "kamu suka makan mie",
        "kamu suka minum kopi",
        "dia suka makan nasi",
        "dia suka minum teh",
        "mereka suka makan nasi",
    ]

    semua_kata = set()
    for k in kalimat:
        for kata in k.split():
            semua_kata.add(kata)

    kata_ke_id = {kata: i for i, kata in enumerate(sorted(semua_kata))}
    id_ke_kata = {i: kata for kata, i in kata_ke_id.items()}
    vocab_size = len(kata_ke_id)

    print(f"Jumlah kata unik (vocab): {vocab_size}")
    print(f"Contoh mapping: {list(kata_ke_id.items())[:5]}")

    X_data = []
    Y_data = []
    for k in kalimat:
        kata = k.split()
        X_data.append([kata_ke_id[kata[0]], kata_ke_id[kata[1]], kata_ke_id[kata[2]]])
        Y_data.append(kata_ke_id[kata[3]])

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    embedding_dim = args.embedding_dim
    embedding = np.random.randn(vocab_size, embedding_dim) * 0.1
    W_out = np.random.randn(embedding_dim, vocab_size) * np.sqrt(2.0 / embedding_dim)
    b_out = np.zeros((1, vocab_size))

    for epoch in range(args.epochs):
        emb = embedding[X_data]
        avg_emb = np.mean(emb, axis=1)
        logits = avg_emb @ W_out + b_out
        probs = softmax(logits)

        loss = cross_entropy_loss(probs, Y_data)

        dlogits = probs.copy()
        dlogits[np.arange(len(Y_data)), Y_data] -= 1
        dlogits /= len(Y_data)

        dW_out = avg_emb.T @ dlogits
        db_out = np.sum(dlogits, axis=0, keepdims=True)

        davg_emb = dlogits @ W_out.T

        demb = np.repeat(davg_emb[:, None, :], 3, axis=1) / 3

        embedding_grad = np.zeros_like(embedding)
        for i in range(len(X_data)):
            for j in range(3):
                embedding_grad[X_data[i, j]] += demb[i, j]

        embedding -= args.lr * embedding_grad
        W_out -= args.lr * dW_out
        b_out -= args.lr * db_out

        if epoch % args.log_every == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")

    print("\n--- Test prediksi ---")
    test_kalimat = [
        "saya suka makan ???",
        "kamu suka minum ???",
        "dia suka makan ???",
    ]

    for tk in test_kalimat:
        kata = tk.replace("???", "").split()
        ids = np.array([[kata_ke_id[k] for k in kata]])

        emb = embedding[ids]
        avg_emb = np.mean(emb, axis=1)
        logits = avg_emb @ W_out + b_out
        probs = softmax(logits)[0]

        top3 = np.argsort(probs)[-3:][::-1]
        print(f"\n{tk}")
        for idx in top3:
            print(f"  {id_ke_kata[idx]:<10} ({probs[idx] * 100:.1f}%)")


if __name__ == "__main__":
    main()
