# app.py — логика бота (команды)
import os
import time
import logging
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
load_dotenv()

BENCH = "SPY"

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
    "ICE","SO","DUK","PNC","SBUX","EQIX","ATVI","AON","ADI","MMC","SHW",
    "USB","FDX","NKE","BDX","GM","F","HAL","NOC","TT","ROP","PSX","KMI",
    "EA","ORLY","ROST","CTAS","HUM","AEP","AIG","KHC","GIS","MNST","MCO",
    "PCAR","CEG","MRNA","PAYX","HSY","ADM","OXY","D","EXC","ED","BK","KDP",
    "AFL","A","ALB","CRWD","FTNT","SNPS","CDNS","MCHP","NXPI","APH"
]

def download_batch(tickers, period="6mo", tries=3, pause=1.2):
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

def get_series(df: pd.DataFrame, ticker: str):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if (ticker, "Adj Close") in df.columns:
            s = df[(ticker, "Adj Close")].dropna()
        elif (ticker, "Close") in df.columns:
            s = df[(ticker, "Close")].dropna()
        else:
            return None
        s.name = ticker
        return s
    if "Adj Close" in df.columns:
        return df["Adj Close"].dropna()
    if "Close" in df.columns:
        return df["Close"].dropna()
    return None

def metric_21(series: pd.Series):
    if series is None or len(series) < 21:
        return None
    pct = series.pct_change()
    tail = pct.tail(21).dropna()
    if len(tail) < 10:
        return None
    ret21 = float(tail.sum())
    vol21 = float(tail.std())
    score = float(ret21 - vol21)
    return score, ret21, vol21

def load_sp500_tickers():
    return HARD_SP500

# ===== Команды =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот AI-рейтинг акций. Напиши /strong чтобы получить сильные идеи.\n"
        "Команды: /ping /sp500 [5|10] /sp5005 /sp50010 /etf [all|core|x2x3] [N] /mix [N]"
    )

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def sp500(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        n = int(context.args[0]) if context.args else 5
        n = 10 if n >= 10 else 5
    except Exception:
        n = 5

    tickers = load_sp500_tickers()
    dfb = download_batch([BENCH])
    sb = get_series(dfb, BENCH)
    mb = metric_21(sb)
    if not mb:
        await update.message.reply_text("Не удалось получить данные SPY.")
        return
    bench_ret = mb[1]

    rows = []
    for i in range(0, len(tickers), 20):
        batch = tickers[i:i+20]
        df = download_batch(batch)
        if df is None:
            continue
        for t in batch:
            m = metric_21(get_series(df, t))
            if not m:
                continue
            score, r21, vol = m
            if r21 > bench_ret:
                rows.append((t, score, r21, vol))

    if not rows:
        await update.message.reply_text("Сейчас нет акций, обогнавших SPY за 21д.")
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:n]
    text = "Топ S&P 500, обогнавших SPY за 21д (score=21д рост − вола):\n" + "\n".join(
        f"{i+1:>2}) {t:<6} | score={s:.4f} | 21д={r:.2%} | vol={v:.4f}"
        for i, (t, s, r, v) in enumerate(top)
    )
    await update.message.reply_text(text)

async def sp5005(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.args = ["5"]; await sp500(update, context)

async def sp50010(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.args = ["10"]; await sp500(update, context)

def pick_etf_universe(arg: str | None):
    if not arg: return ALL_ETF, "all"
    a = arg.lower()
    if a == "core": return CORE_ETF, "core"
    if a in ("x2x3","lev","leveraged"): return LEVERAGED_ETF, "x2x3"
    return ALL_ETF, "all"

async def etf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uni = "all"; n = 5
    if context.args:
        a0 = context.args[0].lower()
        if a0.isdigit(): n = int(a0)
        else: _, uni = pick_etf_universe(a0)
        if len(context.args) >= 2:
            try: n = int(context.args[1])
            except: pass
    n = max(1, min(n, 25))
    tickers, uni = pick_etf_universe(uni)

    rows = []
    for i in range(0, len(tickers), 15):
        batch = tickers[i:i+15]
        df = download_batch(batch)
        if df is None:
            continue
        for t in batch:
            m = metric_21(get_series(df, t))
            if not m: continue
            sc, r, v = m
            rows.append((t, sc, r, v))

    if not rows:
        await update.message.reply_text("Нет данных по ETF.")
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:n]
    text = f"ETF ({uni}) — топ по score (21д рост − вола):\n" + "\n".join(
        f"{i+1:>2}) {t:<6} | score={s:.4f} | 21д={r:.2%} | vol={v:.4f}"
        for i, (t, s, r, v) in enumerate(top)
    )
    await update.message.reply_text(text)

async def mix(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try: n = int(context.args[0]) if context.args else 10
    except: n = 10
    n = max(1, min(n, 30))

    tickers = HARD_SP500 + ALL_ETF
    rows = []
    for i in range(0, len(tickers), 20):
        batch = tickers[i:i+20]
        df = download_batch(batch)
        if df is None: continue
        for t in batch:
            m = metric_21(get_series(df, t))
            if not m: continue
            sc, r, v = m
            label = "ETF" if t in ALL_ETF else "Акция"
            rows.append((t, sc, r, v, label))

    if not rows:
        await update.message.reply_text("Нет данных для смешанного списка.")
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:n]
    text = "Смешанный топ (S&P 500 + ETF) по score:\n" + "\n".join(
        f"{i+1:>2}) {t:<6} [{lbl}] | score={s:.4f} | 21д={r:.2%} | vol={v:.4f}"
        for i, (t, s, r, v, lbl) in enumerate(top)
    )
    await update.message.reply_text(text)
