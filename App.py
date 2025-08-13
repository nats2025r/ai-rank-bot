app.py
import os
import yfinance as yf
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Загружаем переменные окружения
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

# Список тикеров S&P 500 ETF + сами индексы
TICKERS = ["SPY", "VOO", "IVV", "QQQ", "DIA"]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот AI-рейтинг акций. Напиши /strong чтобы получить сильные идеи.")

async def strong(update: Update, context: ContextTypes.DEFAULT_TYPE):
    results = []
    for ticker in TICKERS:
        data = yf.download(ticker, period="6mo", interval="1d")
        if len(data) < 30:
            continue

        # Momentum — % роста за 30 дней
        data["Return_30d"] = data["Adj Close"].pct_change(30)
        last_return = data["Return_30d"].iloc[-1]

        # Волатильность (чем ниже — тем лучше)
        volatility = data["Adj Close"].pct_change().std()

        # AI-скор (простая нормализация)
        score = (last_return / volatility) if volatility != 0 else 0
        results.append((ticker, last_return, volatility, score))

    df = pd.DataFrame(results, columns=["Ticker", "30d Return", "Volatility", "Score"])
    df = df.sort_values("Score", ascending=False)

    message = "📊 Сильные идеи (по AI-скор):\n"
    for _, row in df.iterrows():
        message += f"{row['Ticker']}: {row['30d Return']:.2%}, Score={row['Score']:.2f}\n"

    await update.message.reply_text(message)

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("strong", strong))
    app.run_polling()

if __name__ == "__main__":
    main()
