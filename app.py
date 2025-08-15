# app.py — логика команд бота (без запуска)
import os
import time
import logging
from typing import List, Tuple, Optional

import pandas as pd
import yfinance as yf

# ===== базовая настройка логов =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BENCH = "SPY"

# ===== ETF-универсы =====
CORE_ETF = [
    "SPY","VOO","IVV","VTI","QQQ","DIA","IWM","IWB","IJH","IJR",
    "XLK","XLF","XLE","XLY","XLP","XLV","XLI","XLB","XLRE","XLU",
    "SMH","SOXX","IBB","ARKK","HYG","LQD","EEM","IEMG","GLD","SLV","USO"
]
LEVERAGED_ETF = [
    "SSO","SPXL","UPRO","QLD","TQQQ","UWM","TNA","DDM","UDOW",
    "TECL","FAS","SOXL","LABU","UCO","GUSH","EDC","YINN"
]
ALL_ETF = CORE_ETF + LEVERAGED_ETF

# ===== упрощённый «хард»-набор S&P500 (достаточно для ранжирования) =====
HARD_SP500 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","AVGO",
    "TSM","TSLA","WMT","JPM","XOM","V","UNH","PG","MA","ORCL","COST","HD",
    "JNJ","MRK","NFLX","ABBV","ADBE","CVX","PEP","KO","CRM","BAC","TMO",
    "AMD","ACN","LIN","MCD","CSCO","WFC","ABT","TXN","MS","INTU","PFE",
    "PM","IBM","CMCSA","QCOM","AMAT","DIS","DHR","UBER","RTX","GE","WELL",
    "CAT","BX","ISRG","AMGN","GS","SYK","LRCX","NOW","SPGI","BKNG","HON",
    "ETN","INTC","BLK","DE","PLTR","ELV","ADP","MDT","MU","GEV","MDLZ",
    "AXP","C","GILD","CVS","PGR","PANW","SCHW","REGN","T","COP","LLYVA",
    "ZTS","KLAC","MO","MAR","CI","VRTX","ANET","FI","CSX","LMT","PH","MMC",
    "ICE","SO","DUK","PNC","SBUX","EQIX","ADI","SHW","USB","FDX","NKE",
    "BDX","GM","F","HAL","NOC","TT","ROP","PSX","KMI","EA","ORLY","ROST",
    "CTAS","HUM","AEP","AIG","KHC","GIS","MNST","MCO","PCAR","CEG","MRNA",
    "PAYX","HSY","ADM","OXY","D","EXC","ED","BK","KDP","AFL","A","ALB",
    "CRWD","FTNT","SNPS","CDNS","MCHP","NXPI","APH"
]

# ========= утилиты загрузки/метрик =========
def download_batch(tickers: List[str], period: str = "6mo", tries: int = 3, pause: float = 1.2) -> Optional[pd.DataFrame]:
    """Скачиваем пачкой (дневки, 6 месяцев)"""
    for i in range(tries):
        try:
            df = yf.download(
                tickers, period=period, interval="1d",
                auto_adjust=False, group_by="ticker",
                progress=False, threads=True,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as e:
            logging.warning("Batch download failed (%s/%s): %s", i+1, tries, e)
        time.sleep(pause)
    return None

def get_series(df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        # предпочитаем Adj Close
        col = ("Adj Close" if (ticker, "Adj Close") in df.columns else
               "Close" if (ticker, "Close") in df.columns else None)
        if not col:
            return None
        s = df[(ticker, col)].dropna()
    else:
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if not col:
            return None
        s = df[col].dropna()
    s.name = ticker
    return s

def metric_21(series: Optional[pd.Series]) -> Optional[Tuple[float, float, float]]:
    """Возвращает (score, 21д доходность, 21д волатильность) где score=ret−vol"""
    if series is None or len(series) < 30:
        return None
    pct = series.pct_change()
    tail = pct.tail(21).dropna()
    if len(tail) < 10:
        return None
    ret21 = float(tail.sum())
    vol21 = float(tail.std())
    score = float(ret21 - vol21)
    return score, ret21, vol21

def pick_etf_universe(arg: Optional[str]):
    if not arg:
        return ALL_ETF, "all"
    a = arg.lower()
    if a == "core":
        return CORE_ETF, "core"
    if a in ("x2x3", "lev", "leveraged"):
        return LEVERAGED_ETF, "x2x3"
    return ALL_ETF, "all"

# ========= команды =========
from telegram import Update
from telegram.ext import ContextTypes

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Я на связи ✅\n\n"
        "Команды:\n"
        "/ping — проверить, что живой\n"
        "/sp500 [5|10] — топ акций S&P 500, обогнавших SPY (21д), по score\n"
        "/sp5005 и /sp50010 — быстрые варианты\n"
        "/etf [all|core|x2x3] [N] — ETF по score\n"
        "/etfcore — ядро ETF, /etfall — все ETF, /etfx2x3 — x2/x3\n"
        "/mix [N] — смешанный список (S&P500 + ETF)"
    )

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def sp500(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # сколько показать
    try:
        n = int(context.args[0]) if context.args else 5
        n = 10 if n >= 10 else 5
    except Exception:
        n = 5

    # бенчмарк
    dfb = download_batch([BENCH])
    sb = get_series(dfb, BENCH)
    mb = metric_21(sb)
    if not mb:
        await update.message.reply_text("Не удалось получить данные SPY.")
        return
    bench_ret = mb[1]

    rows = []
    batch_size = 20
    for i in range(0, len(HARD_SP500), batch_size):
        batch = HARD_SP500[i:i+batch_size]
        df = download_batch(batch)
        if df is None:
            time.sleep(0.7)
            continue
        for t in batch:
            m = metric_21(get_series(df, t))
            if not m:
                continue
            score, r21, vol = m
            # показываем только тех, кто обогнал SPY
            if r21 > bench_ret:
                rows.append((t, score, r21, vol))
        time.sleep(0.8)

    if not rows:
        await update.message.reply_text("Сейчас нет акций, обогнавших SPY за 21д.")
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:n]
    text = (
        f"Топ S&P 500, обогнавших SPY (21д), score=21д рост − вола (показываю {n}):\n" +
        "\n".join(
            f"{i+1:>2}) {t:<6} | score={s:.4f} | 21д={r:.2%} | vol={v:.4f}"
            for i, (t, s, r, v) in enumerate(top)
        )
    )
    await update.message.reply_text(text)

async def sp5005(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.args = ["5"]
    await sp500(update, context)

async def sp50010(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.args = ["10"]
    await sp500(update, context)

async def etf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # разбор аргументов
    uni_key = None
    n = 5
    if context.args:
        a0 = context.args[0].lower()
        if a0.isdigit():
            n = int(a0)
        else:
            uni_key = a0
        if len(context.args) >= 2:
            try:
                n = int(context.args[1])
            except Exception:
                pass
    n = max(1, min(n, 25))
    tickers, uni = pick_etf_universe(uni_key)

    rows = []
    batch_size = 15
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        df = download_batch(batch)
        if df is None:
            time.sleep(0.7)
            continue
        for t in batch:
            m = metric_21(get_series(df, t))
            if not m:
                continue
            sc, r, v = m
            rows.append((t, sc, r, v))
        time.sleep(0.8)

    if not rows:
        await update.message.reply_text("Нет данных по ETF.")
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:n]
    text = (
        f"ETF ({uni}) — топ по score = 21д рост − вола (показываю {n}):\n" +
        "\n".join(
            f"{i+1:>2}) {t:<6} | score={s:.4f} | 21д={r:.2%} | vol={v:.4f}"
            for i, (t, s, r, v) in enumerate(top)
        )
    )
    await update.message.reply_text(text)

async def mix(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        n = int(context.args[0]) if context.args else 10
    except Exception:
        n = 10
    n = max(1, min(n, 30))

    tickers = HARD_SP500 + ALL_ETF
    rows = []
    batch_size = 20

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        df = download_batch(batch)
        if df is None:
            time.sleep(0.7)
            continue
        for t in batch:
            m = metric_21(get_series(df, t))
            if not m:
                continue
            sc, r, v = m
            label = "ETF" if t in ALL_ETF else "Акция"
            rows.append((t, sc, r, v, label))
        time.sleep(0.8)

    if not rows:
        await update.message.reply_text("Нет данных для смешанного списка.")
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:n]
    text = (
        f"Смешанный топ {n} (S&P 500 + ETF), сортировка по score:\n" +
        "\n".join(
            f"{i+1:>2}) {t:<6} [{lbl}] | score={s:.4f} | 21д={r:.2%} | vol={v:.4f}"
            for i, (t, s, r, v, lbl) in enumerate(top)
        )
    )
    await update.message.reply_text(text)

