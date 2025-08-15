# app_webhook.py — запуск бота через webhook (Render)
import os, logging
from telegram.ext import Application, CommandHandler

from app import start, ping, sp500, sp5005, sp50010, etf, mix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TOKEN = os.getenv("BOT_TOKEN")
PORT = int(os.getenv("PORT", "10000"))         # Render выдаёт PORT
PUBLIC_URL = os.getenv("PUBLIC_URL")           # например, https://ai-rank-bot-1.onrender.com

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN не задан")
    if not PUBLIC_URL:
        raise RuntimeError("PUBLIC_URL не задан")

    app = Application.builder().token(TOKEN).build()

    # Регистрируем команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("sp500", sp500))
    app.add_handler(CommandHandler("sp5005", sp5005))
    app.add_handler(CommandHandler("sp50010", sp50010))
    app.add_handler(CommandHandler("etf", etf))
    app.add_handler(CommandHandler("mix", mix))

    url = PUBLIC_URL.rstrip("/") + f"/{TOKEN}"
    logging.info("Starting webhook on %s", url)

    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=TOKEN,
        webhook_url=url,
    )

if __name__ == "__main__":
    main()
