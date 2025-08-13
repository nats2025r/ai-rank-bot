import os, asyncio, yfinance as yf, pandas as pd, numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")  # токен из переменных окружения Render

# простой список ETF (можно распаковать позже на S&P500)
TICKERS = ["SPY","VOO","IVV","QQQ","DIA"]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот с рейтингом акций/ETF. Команды:\n"
        "/strong — сильные идеи по простому скору (30д доходность / волатильность)."
    )

async def strong(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = []
    for t in TICKERS:
        try:
            df = yf.download(t, period="6mo", interval="1d", auto_adjust=True, progress=False, threads=False)
            if df is None or df.empty or len(df) < 30:
                continue
            ret30 = df["Close"].pct_change(30).iloc[-1]
            vol   = df["Close"].pct_change().std()
            score = (ret30 / vol) if vol and vol != 0 else 0.0
            rows.append((t, float(ret30), float(vol), float(score)))
        except Exception:
            continue
    if not rows:
        await update.message.reply_text("Не удалось получить данные сейчас, попробуй позже.")
        return
    rows.sort(key=lambda x: x[3], reverse=True)
    txt = "📊 Сильные идеи (наивный скор):\n" + "\n".join([f"{i+1}. {t}: 30d={r:.2%}, score={s:.2f}" for i,(t,r,v,s) in enumerate(rows)])
    await update.message.reply_text(txt)

def make_app() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is empty. Set it in Render → Environment Variables.")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("strong", strong))
    return app

if __name__ == "__main__":
    # Параметры вебхука
    PORT = int(os.getenv("PORT", "10000"))  # Render передаст свой порт
    SECRET = os.getenv("WEBHOOK_SECRET", "secret-path-123")  # любой случайный путь
    BASE = os.getenv("WEBHOOK_BASE") or os.getenv("RENDER_EXTERNAL_URL")  # Render задаст после первого деплоя
    app = make_app()
    if BASE:
        url = f"{BASE.rstrip('/')}/{SECRET}"
        print(f"Setting webhook to: {url}")
        app.run_webhook(listen="0.0.0.0", port=PORT, url_path=SECRET, webhook_url=url)
    else:
        # если BASE ещё не известен (первый деплой) — просто поднимем сервер,
        # но вебхук станет активным после повторного деплоя, когда появится BASE
        print("WEBHOOK_BASE/RENDER_EXTERNAL_URL not set yet. Start HTTP server and set webhook on next deploy.")
        app.run_webhook(listen="0.0.0.0", port=PORT, url_path=SECRET)
