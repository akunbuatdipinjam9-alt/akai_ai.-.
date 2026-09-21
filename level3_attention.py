import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Level 3: demo attention mechanism dari nol")
    parser.add_argument("--embedding-dim", type=int, default=8, help="Dimensi vektor embedding tiap kata")
    parser.add_argument("--vocab-size", type=int, default=20, help="Ukuran vocab pura-pura untuk demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    kalimat = ["saya", "suka", "makan", "nasi"]
    vocab_size = args.vocab_size
    embedding_dim = args.embedding_dim

    embedding = np.random.randn(vocab_size, embedding_dim) * 0.5

    kata_ids = [3, 7, 12, 5]
    X = embedding[kata_ids]

    print(f"=== Input Embedding (4 kata x {embedding_dim} dimensi) ===")
    print(X)
    print()

    W_q = np.random.randn(embedding_dim, embedding_dim) * 0.5
    W_k = np.random.randn(embedding_dim, embedding_dim) * 0.5
    W_v = np.random.randn(embedding_dim, embedding_dim) * 0.5

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    print("=== Query (yang 'bertanya') ===")
    print(Q)
    print()

    scores = Q @ K.T

    print("=== Attention Scores (sebelum scaling) ===")
    print(np.round(scores, 3))
    print()
    print("Baris = kata yang bertanya (Query)")
    print("Kolom = kata yang ditanya (Key)")
    print()

    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    print(f"=== Setelah Scaling (bagi √{d_k}={np.sqrt(d_k):.2f}) ===")
    print(np.round(scaled_scores, 3))
    print()

    attention_weights = softmax(scaled_scores)

    print("=== Attention Weights (setelah softmax) ===")
    print(np.round(attention_weights, 3))
    print()
    print("Setiap baris = bobot perhatian satu kata ke semua kata")
    print("Jumlah tiap baris = 1.0")
    print()

    print("=== Interpretasi ===")
    for i, kata in enumerate(kalimat):
        print(f"Kata '{kata}' memperhatikan:")
        for j, kata_lain in enumerate(kalimat):
            bobot = attention_weights[i, j]
            bar = "█" * int(bobot * 30)
            print(f"  {kata_lain:<8} {bobot:.3f} {bar}")
        print()

    output = attention_weights @ V

    print(f"=== Output Attention (4 kata x {embedding_dim} dimensi) ===")
    print(np.round(output, 3))
    print()
    print("Setiap kata sekarang 'mengandung informasi' dari kata lain")
    print("yang relevan, sesuai bobot attention.")


if __name__ == "__main__":
    main()
