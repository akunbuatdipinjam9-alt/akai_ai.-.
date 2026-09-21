import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Level 0: neuron tunggal belajar y = 2x + 1")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Jumlah epoch training")
    parser.add_argument("--log-every", type=int, default=100, help="Cetak progress tiap berapa epoch")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (opsional, untuk hasil konsisten)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    Y = np.array([3, 5, 7, 9, 11], dtype=np.float32)

    w = np.random.randn()
    b = np.random.randn()

    print(f"Sebelum training: w={w:.4f}, b={b:.4f}")

    for epoch in range(args.epochs):
        y_pred = w * X + b

        loss = np.mean((y_pred - Y) ** 2)

        error = y_pred - Y
        dw = np.mean(2 * error * X)
        db = np.mean(2 * error)

        w = w - args.lr * dw
        b = b - args.lr * db

        if epoch % args.log_every == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | w={w:.4f}, b={b:.4f}")

    print(f"\nSetelah training: w={w:.4f}, b={b:.4f}")
    print(f"Target sebenarnya: w=2.0, b=1.0")

    print("\n--- Test prediksi ---")
    for x in [6, 7, 10]:
        y_pred = w * x + b
        y_benar = 2 * x + 1
        print(f"x={x} → prediksi={y_pred:.2f}, seharusnya={y_benar}")


if __name__ == "__main__":
    main()
