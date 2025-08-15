import os, logging
from telegram.ext import Application, CommandHandler
from app import start, strong, ping   # <-- только эти три

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TOKEN = os.getenv("BOT_TOKEN")
PORT = int(os.getenv("PORT", "10000"))
PUBLIC_URL = os.getenv("PUBLIC_URL")  # например, https://ai-rank-bot-1.onrender.com

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN не задан")
    if not PUBLIC_URL:
        raise RuntimeError("PUBLIC_URL не задан")

    app = Application.builder().token(TOKEN).build()

    # Регистрируем только существующие команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("strong", strong))

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
