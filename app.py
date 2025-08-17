import os
import time
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np

from flask import Flask, request
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= ЛОГИ =========
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
log = logging.getLogger("ai-rank-bot")

# ========= ENV =========
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
URL   = os.getenv("WEBHOOK_URL")  # напр.: https://ai-rank-bot-1.onrender.com
if not TOKEN:
    raise RuntimeError("Не найден BOT_TOKEN")
if not URL:
    raise RuntimeError("Не найден WEBHOOK_URL")

# ========= Flask + PTB =========
app = Flask(__name__)
tg_app = Application.builder().token(TOKEN).build()

# ========= Константы =========
DAYS     = 21
BENCH    = "SPY"
PERIOD   = f"{DAYS*3}d"  # достаточная история для score

# ========= HTTP-сессия для надёжной загрузки yfinance =========
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})
retries = Retry(total=3, backoff_factor=0.7, status_forcelist=[429, 500, 502, 503, 504])
SESSION.mount("https://", HTTPAdapter(max_retries=retries))
SESSION.mount("http://",  HTTPAdapter(max_retries=retries))

# ========= Списки тикеров =========
# 1) Доп. акции со скринов — добавим к S&P топ-250
EXTRA_STOCKS = [
    "VNT","ODFL","GME","ORCL","GIL","MSI","CPRT","DASH","IBKR","JBL","KWEB","GS",
    "GE","CTAS","CMG","ACN","RTX","KLG","GSL","TSM","ON","NXPI","MU","MPWR","QLYS"
]

# 2) Все ETF со скринов (объединённый список)
ETF_ALL = sorted(set([
    # базовые индексы / секторные / тематические
    "SPY","QQQ","DIA","IWM","RSP","VOO","IVV","QQQM","SMH","IGV","XSD",
    "XLRE","XLC","XLE","XLK","XLV","XLF","XLY","XLI","XLP","XLB","XLU",
    "SPMO","SPYU","UDOW","SSO","SPUU","WANT","KWEB",
    # crypto-экспозиция
    "IBIT","BITX","BITO","ETHE",
    # из твоих прежних списков Up/Down/BTC/FAANG/FAA2
    "BOIL","UCO","SOXL","TSLL","WEBL","TNA","CWEB","FAS","NUGT","GUSH","LIT","TECL","USD",
    "TZA","KOLD","FAZ","DRIP","SOXS","SPXU","SCO","YANG","DUST","WEBS","SQQQ","SDOW",
    "BULZ","FNGS","FNGG","MAGX","GGLL","FBL","AMZU","FNGO","MSFU","MSFL","AAPU","NVDL","TSLT","AAPB","FNGA",
    "SPXL","UPRO","ARKK"
]))

# 3) Явный перечень x2/x3 ETF (для отдельной команды)
ETF_LEVERAGED = sorted(set([
    "SPXL","UPRO","SPXS","SPXU","SSO","SPUU","UDOW","SDOW",
    "TQQQ","SQQQ","QLD","WEBL","WEBS",
    "SOXL","SOXS","TECL","TNA","TZA",
    "FAS","FAZ","NUGT","DUST","GUSH","DRIP","UCO","SCO","BOIL","KOLD","YANG","WANT",
    "TSLL","TSLT","GGLL","FBL","AMZU","FNGG","FNGO","MSFU","MSFL","AAPU","AAPB","NVDL"
]))

# ========= Функции данных =========
def get_spx_top250_from_slickcharts() -> list[str]:
    """
    Тянем топ-250 S&P 500 по весу (соответствует крупнейшим по капитализации)
    со Slickcharts. Берём первые 250 символов.
    """
    url = "https://www.slickcharts.com/sp500"
    try:
        html = SESSION.get(url, timeout=15).text
        # Быстро распарсим таблицу через pandas
        tables = pd.read_html(html)
        # Обычно первая таблица — с колонками: #, Company, Symbol, Weight, ...
        df = None
        for t in tables:
            if {"Symbol"}.issubset(set(t.columns)):
                df = t
                break
        if df is None or df.empty:
            log.warning("Не удалось найти таблицу с колонкой Symbol на Slickcharts")
            return []
        symbols = df["Symbol"].astype(str).str.replace(" ", "", regex=False).tolist()
        symbols = [s.replace(".", "-") if s.count(".") == 1 else s for s in symbols]  # BRK.B -> BRK-B
        return symbols[:250]
    except Exception as e:
        log.error("Ошибка загрузки Slickcharts: %s", e)
        return []

def yf_download_safe(ticker: str, period: str) -> pd.DataFrame:
    """
    Надёжная загрузка OHLCV для одного тикера через yfinance.
    3 попытки + threads=False + session=SESSION.
    """
    for i in range(3):
        try:
            df = yf.download(
                ticker, period=period, progress=False, threads=False, session=SESSION
            )
            if df is not None and not df.empty:
                return df
            log.warning("Пустые данные для %s (попытка %d)", ticker, i + 1)
        except Exception as e:
            log.warning("yfinance error %s (try %d): %s", ticker, i + 1, e)
        time.sleep(1 + i * 2)
    return pd.DataFrame()

def calc_score(ticker: str, benchmark: str = BENCH, days: int = DAYS):
    """
    Тот же алгоритм score, что и у тебя:
    (ret_rel / vol) * vol_confirm
    """
    try:
        data = yf_download_safe(ticker, period=f"{days*3}d")
        if data.empty:
            return None
        close = data["Close"].dropna()
        volume = data["Volume"].dropna()
        if len(close) < days:
            return None

        # Доходность тикера
        ret = close.iloc[-1] / close.iloc[-days] - 1

        # Доходность бенчмарка
        bench = yf_download_safe(benchmark, period=f"{days*3}d")
        if bench.empty:
            return None
        bench_close = bench["Close"].dropna()
        if len(bench_close) < days:
            return None
        ret_bench = bench_close.iloc[-1] / bench_close.iloc[-days] - 1

        # Относительный ретёрн
        ret_rel = ret - ret_bench

        # Волатильность
        vol = close.pct_change().dropna().std()
        if not np.isfinite(vol) or vol == 0:
            return None

        # Подтверждение объёмом
        avg21 = float(volume.tail(days).mean())
        avg60 = float(volume.tail(days * 3).mean())
        vol_confirm = (avg21 / avg60) if (avg60 and avg60 > 0) else 1.0

        score = (ret_rel / (vol + 1e-9)) * vol_confirm
        return float(score)
    except Exception as e:
        log.error("Ошибка calc_score(%s): %s", ticker, e)
        return None

# ========= Инициализируем пул акций =========
SPX250_DYNAMIC = get_spx_top250_from_slickcharts()
if not SPX250_DYNAMIC or len(SPX250_DYNAMIC) < 200:
    log.warning("Slickcharts не вернул топ-250, будем работать только с EXTRA_STOCKS")
SPX_STOCKS = sorted(set(SPX250_DYNAMIC) | set(EXTRA_STOCKS))

# ========= Универсальный ответ ранжирования =========
async def _rank_and_reply(update: Update, tickers: list[str], n: int, title: str):
    await update.message.reply_text(f"Считаю {title}…")
    scores = {}
    for t in tickers:
        s = calc_score(t)
        if s is not None:
            scores[t] = s
    if not scores:
        await update.message.reply_text(f"{title}: нет данных ❌")
        return
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    lines = [f"📊 {title}:"]
    for t, s in top:
        lines.append(f"{t}: {s:.2f}")
    await update.message.reply_text("\n".join(lines))

# ========= Команды =========
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет 👋 Я AI Rank Bot.\n\n"
        "Команды:\n"
        "/sp500 — топ-10 акций (S&P топ-250 + твои)\n"
        "/sp5005 — топ-5 акций\n"
        "/etf — топ-10 ETF (все)\n"
        "/etf5 — топ-5 ETF\n"
        "/etfx — топ-10 ETF x2/x3\n"
        "/etfx5 — топ-5 ETF x2/x3\n"
        "/ping — проверить связь"
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅")

# Акции
async def cmd_sp500(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _rank_and_reply(update, SPX_STOCKS, 10, "Топ-10 акций")

async def cmd_sp5005(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _rank_and_reply(update, SPX_STOCKS, 5, "Топ-5 акций")

# ETF
async def cmd_etf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _rank_and_reply(update, ETF_ALL, 10, "Топ-10 ETF")

async def cmd_etf5(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _rank_and_reply(update, ETF_ALL, 5, "Топ-5 ETF")

# ETF x2/x3
async def cmd_etfx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tick = [t for t in ETF_LEVERAGED if t in ETF_ALL]
    await _rank_and_reply(update, tick, 10, "Топ-10 ETF x2/x3")

async def cmd_etfx5(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tick = [t for t in ETF_LEVERAGED if t in ETF_ALL]
    await _rank_and_reply(update, tick, 5, "Топ-5 ETF x2/x3")

# ========= Регистрация хэндлеров =========
tg_app.add_handler(CommandHandler("start",  cmd_start))
tg_app.add_handler(CommandHandler("ping",   cmd_ping))
tg_app.add_handler(CommandHandler("sp500",  cmd_sp500))
tg_app.add_handler(CommandHandler("sp5005", cmd_sp5005))
tg_app.add_handler(CommandHandler("etf",    cmd_etf))
tg_app.add_handler(CommandHandler("etf5",   cmd_etf5))
tg_app.add_handler(CommandHandler("etfx",   cmd_etfx))
tg_app.add_handler(CommandHandler("etfx5",  cmd_etfx5))

# ========= Flask webhook =========
@app.route(f"/webhook/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), tg_app.bot)
    tg_app.update_queue.put_nowait(update)
    return "ok"

# ========= Запуск =========
if __name__ == "__main__":
    import asyncio

    async def set_webhook():
        webhook_url = f"{URL}/webhook/{TOKEN}"
        await tg_app.bot.delete_webhook(drop_pending_updates=True)
        await tg_app.bot.set_webhook(webhook_url)
        log.info("Webhook установлен: %s", webhook_url)

    asyncio.get_event_loop().run_until_complete(set_webhook())
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
