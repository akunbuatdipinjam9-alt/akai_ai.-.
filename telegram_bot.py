import logging
import os
import re

import torch
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from pytorch_chatbot import (
    _adalah_perintah_tools,
    jawab_dengan_fallback,
    load_config,
    log_percakapan,
    muat_model_terlatih,
    proses_perintah_tools,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("akai-telegram")

PESAN_SELAMAT_DATANG = (
    "Halo, aku Akai! 👋\n"
    "Chat langsung aja buat ngobrol biasa, atau pakai perintah:\n"
    "/hitung 12*5 | /todo tambah|list|selesai|hapus | /cuaca <kota>\n"
    "/sholat <kota> | /waktu <kota> | /terjemah <teks> | /kurs <dari> <ke> [jumlah]\n"
    "/wiki <topik> | /alquran <surat>:<ayat> | /gambar <deskripsi>\n"
    "/hadits [kitab] [nomor] | /tanyaai <pertanyaan>"
)


def _bersihkan_teks(teks):
    # Buang suffix "@NamaBot" yang suka ditempel Telegram di command grup, misal "/hitung@AkaiBot 1+1"
    return re.sub(r"^(/[a-zA-Z_]+)@\w+", r"\1", teks or "").strip()


def _boleh_akses(chat_id, telegram_cfg):
    allowlist = (telegram_cfg or {}).get("telegram_allowed_chat_ids") or []
    if not allowlist:
        return True
    return chat_id in allowlist


class AkaiTelegramBot:
    def __init__(self, cfg, model, kata_ke_id, id_ke_kata, device, model_cfg):
        self.cfg = cfg
        self.model = model
        self.kata_ke_id = kata_ke_id
        self.id_ke_kata = id_ke_kata
        self.device = device
        self.model_cfg = model_cfg
        self.gen_cfg = cfg["generation"]
        self.tools_cfg = cfg.get("tools", {})
        self.telegram_cfg = cfg.get("telegram", {})
        self.memory_cfg = cfg.get("memory")

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pesan = update.message
        if not pesan or not pesan.text:
            return

        chat_id = update.effective_chat.id
        if not _boleh_akses(chat_id, self.telegram_cfg):
            await pesan.reply_text("Maaf, kamu belum diizinkan pakai bot ini.")
            return

        teks = _bersihkan_teks(pesan.text)
        if not teks:
            return

        if teks.lower() == "/start":
            await pesan.reply_text(PESAN_SELAMAT_DATANG)
            return

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        if _adalah_perintah_tools(teks):
            _, balasan, path_gambar = proses_perintah_tools(teks, self.tools_cfg)
            if path_gambar and os.path.exists(path_gambar):
                try:
                    with open(path_gambar, "rb") as foto:
                        await pesan.reply_photo(photo=foto, caption=balasan or "")
                    return
                except Exception as e:
                    log.warning("gagal kirim foto: %s", e)
            if balasan:
                await pesan.reply_text(balasan)
            return

        jawaban, sumber = jawab_dengan_fallback(
            self.model, teks, self.kata_ke_id, self.id_ke_kata, self.device,
            self.model_cfg, self.gen_cfg, self.tools_cfg,
        )
        teks_balasan = jawaban if sumber == "lokal" else f"[{sumber}] {jawaban}"
        await pesan.reply_text(teks_balasan)
        log_percakapan(self.memory_cfg, teks, jawaban)


def main():
    cfg = load_config("config.json")
    telegram_cfg = cfg.get("telegram", {})
    token = telegram_cfg.get("bot_token")

    if not token or token == "MASUKKAN_TOKEN_BOT_TELEGRAM_KAMU_DI_SINI":
        raise SystemExit(
            "Token bot Telegram belum diisi. Bikin bot lewat @BotFather di Telegram, "
            "terus taruh token-nya di config.json bagian telegram.bot_token"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model, kata_ke_id, id_ke_kata, model_cfg = muat_model_terlatih(cfg, device)
    log.info("Model berhasil di-load, vocab %d kata.", len(kata_ke_id))

    bot = AkaiTelegramBot(cfg, model, kata_ke_id, id_ke_kata, device, model_cfg)

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT, bot.on_message))

    log.info("Bot Telegram jalan, tunggu pesan masuk... (Ctrl+C buat berhenti)")
    app.run_polling()


if __name__ == "__main__":
    main()
