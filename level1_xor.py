import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Level 1: neural network kecil belajar XOR")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50000, help="Jumlah epoch training")
    parser.add_argument("--hidden", type=int, default=8, help="Jumlah neuron di hidden layer")
    parser.add_argument("--log-every", type=int, default=5000, help="Cetak progress tiap berapa epoch")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(np.float32)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def main():
    args = parse_args()
    np.random.seed(args.seed)

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=np.float32)

    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    hidden = args.hidden
    W1 = np.random.randn(2, hidden) * np.sqrt(2.0 / 2)
    b1 = np.zeros((1, hidden))

    W2 = np.random.randn(hidden, 1) * np.sqrt(2.0 / hidden)
    b2 = np.zeros((1, 1))

    loss = 0.0
    for epoch in range(args.epochs):
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)

        eps = 1e-7
        loss = -np.mean(Y * np.log(a2 + eps) + (1 - Y) * np.log(1 - a2 + eps))

        dL_dz2 = (a2 - Y) / Y.shape[0]
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * relu_derivative(z1)
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        W2 -= args.lr * dL_dW2
        b2 -= args.lr * dL_db2
        W1 -= args.lr * dL_dW1
        b1 -= args.lr * dL_db1

        if epoch % args.log_every == 0:
            print(f"Epoch {epoch:6d} | Loss: {loss:.6f}")

    print(f"\nLoss akhir: {loss:.6f}")

    print("\n--- Hasil prediksi ---")
    print(f"{'Input':<12} {'Target':<10} {'Prediksi':<12} {'Keputusan'}")
    benar = 0
    for i in range(len(X)):
        z1 = X[i:i + 1] @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        pred = a2[0, 0]
        keputusan = 1 if pred > 0.5 else 0
        if keputusan == int(Y[i, 0]):
            benar += 1
        print(f"{str(X[i]):<12} {Y[i, 0]:<10} {pred:<12.4f} {keputusan}")

    print(f"\nBenar: {benar}/4")


if __name__ == "__main__":
    main()
