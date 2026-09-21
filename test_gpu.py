import argparse
import time

import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark training loop singkat di CPU/GPU")
    parser.add_argument("--input-dim", type=int, default=100, help="Dimensi input dummy")
    parser.add_argument("--hidden", type=int, default=512, help="Ukuran hidden layer")
    parser.add_argument("--output-dim", type=int, default=10, help="Jumlah kelas output dummy")
    parser.add_argument("--batch-size", type=int, default=64, help="Ukuran batch dummy")
    parser.add_argument("--steps", type=int, default=500, help="Jumlah step training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--log-every", type=int, default=100, help="Cetak progress tiap berapa step")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU tidak terdeteksi, training di CPU (lebih lambat)")

    model = nn.Sequential(
        nn.Linear(args.input_dim, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.output_dim),
    ).to(device)

    x = torch.randn(args.batch_size, args.input_dim).to(device)
    y = torch.randint(0, args.output_dim, (args.batch_size,)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for step in range(args.steps):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

    elapsed = time.time() - start
    print(f"\n{args.steps} step selesai dalam {elapsed:.2f} detik")
    print(f"Training loop jalan di {device}!")


if __name__ == "__main__":
    main()
