import argparse

import torch

from pytorch_chatbot import (
    apply_overrides,
    build_vocab,
    generate,
    load_config,
    load_percakapan,
    plot_loss,
    train,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Latih chatbot sambil melihat progres & grafik loss")
    parser.add_argument("--config", default="config.json", help="Path file konfigurasi JSON")
    parser.add_argument("--data", default=None, help="Override path dataset percakapan (json)")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch training")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override ukuran batch")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    paths = cfg["paths"]
    percakapan = load_percakapan(paths["data_file"])
    print(f"Jumlah percakapan: {len(percakapan)}")

    kata_ke_id, id_ke_kata = build_vocab(percakapan)
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]

    print("\n" + "=" * 90)
    print(f"TRAINING VISUAL — {cfg['training']['epochs']} epoch")
    print("=" * 90)

    model, loss_history = train(percakapan, kata_ke_id, id_ke_kata, cfg, device)
    plot_loss(loss_history)

    print("\n" + "=" * 90)
    print("TEST AKHIR — Greedy (pasti) vs Sampling (variatif)")
    print("=" * 90)

    test_prompts = [
        "halo", "apa kabar", "siapa kamu", "siapa namamu",
        "terima kasih", "selamat pagi", "bye", "kamu pintar",
    ]

    for p in test_prompts:
        greedy = generate(model, p, kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg, temperature=0.01)
        sampled = generate(model, p, kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg, temperature=0.8)
        print(f"Kamu   : {p}")
        print(f"Greedy : {greedy}")
        print(f"Sample : {sampled}")
        print()

    print("=" * 90)


if __name__ == "__main__":
    main()
