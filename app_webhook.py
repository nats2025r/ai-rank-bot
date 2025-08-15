# app_webhook.py — запуск PTB через вебхук на Render
import os, logging
from telegram.ext import Application, CommandHandler

# импортируем хэндлеры из app.py
from app import start, ping, sp500, sp5005, sp50010, etf, mix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TOKEN = os.getenv("BOT_TOKEN")
PORT = int(os.getenv("PORT", "10000"))
PUBLIC_URL = os.getenv("PUBLIC_URL")  # например, https://ai-rank-bot.onrender.com

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN не задан")
    if not PUBLIC_URL:
        raise RuntimeError("PUBLIC_URL не задан")

    app = Application.builder().token(TOKEN).build()

    # регистрируем команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))

    app.add_handler(CommandHandler("sp500", sp500))
    app.add_handler(CommandHandler("sp5005", sp5005))
    app.add_handler(CommandHandler("sp50010", sp50010))

    # ETF: универсальная и шорткаты
    app.add_handler(CommandHandler("etf", etf))
    # шорткаты: /etfcore, /etfall, /etfx2x3
    app.add_handler(CommandHandler("etfcore", lambda u, c: etf(u, type("ctx", (), {"args": ["core"]}) )))  # etf core
    app.add_handler(CommandHandler("etfall",  lambda u, c: etf(u, type("ctx", (), {"args": ["all"]}) )))
    app.add_handler(CommandHandler("etfx2x3", lambda u, c: etf(u, type("ctx", (), {"args": ["x2x3"]}) )))

    app.add_handler(CommandHandler("mix", mix))

    # на вебхуке
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

