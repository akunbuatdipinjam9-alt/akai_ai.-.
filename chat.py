import argparse
import os
import pickle
import sys

import torch

from pytorch_chatbot import TransformerChatbot, chat_loop, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Ngobrol di terminal dengan chatbot yang sudah dilatih")
    parser.add_argument("--config", default="config.json", help="Path file konfigurasi JSON")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature default saat chat")
    parser.add_argument("--no-memory", action="store_true", help="Matikan penyimpanan riwayat chat ke file untuk sesi ini")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.temperature is not None:
        cfg["generation"]["default_temperature"] = args.temperature
    if args.no_memory:
        cfg["memory"]["enabled"] = False

    paths = cfg["paths"]
    if not (os.path.exists(paths["model_path"]) and os.path.exists(paths["vocab_path"])):
        print("Model belum ditemukan.")
        print(f"  ({paths['model_path']} / {paths['vocab_path']} tidak ada)")
        print("Latih dulu modelnya dengan salah satu dari ini:")
        print("  python pytorch_chatbot.py")
        print("  python train_visual.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        with open(paths["vocab_path"], "rb") as f:
            saved = pickle.load(f)
        kata_ke_id = saved["kata_ke_id"]
        id_ke_kata = saved["id_ke_kata"]
        model_cfg = saved.get("model_cfg", cfg["model"])

        model = TransformerChatbot(len(kata_ke_id), model_cfg).to(device)
        model.load_state_dict(torch.load(paths["model_path"], map_location=device))
        model.eval()
    except Exception as e:
        print(f"Gagal load model ({e}).")
        print("Kemungkinan dataset/config sudah berubah sejak model terakhir dilatih.")
        print("Latih ulang dengan: python pytorch_chatbot.py --retrain")
        sys.exit(1)

    print("Model berhasil dimuat.\n")

    gen_cfg = cfg["generation"]
    chat_loop(model, kata_ke_id, id_ke_kata, device, model_cfg, gen_cfg,
              temperature=gen_cfg["default_temperature"], memory_cfg=cfg.get("memory"),
              tools_cfg=cfg.get("tools"))


if __name__ == "__main__":
    main()
