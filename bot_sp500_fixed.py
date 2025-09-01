# bot_sp500_fixed.py
# Telegram-бот: AI-моментум + ConfirmScore для SP250/ETFALL/ETFX
# Зависимости: python-telegram-bot==20.7, yfinance==0.2.43, pandas==2.2.2, numpy==1.26.4

import os
import math
import asyncio
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ──────────────────────────────────────────────────────────────────────────────
# ENV (можно задать в Render → Environment)
# ──────────────────────────────────────────────────────────────────────────────
BOT_TOKEN     = os.getenv("TELEGRAM_TOKEN", "").strip()
AI_CSV_URL    = os.getenv("AI_CSV_URL", "none").strip()    # "none" или URL CSV: ticker,ai_score
W_AI          = float(os.getenv("W_AI", "0.0"))            # 0..1 — вес AI в композите
BENCH_DEFAULT = os.getenv("BENCH_DEFAULT", "SPY").strip().upper()
BENCH_MODE    = os.getenv("BENCH_MODE", "global").strip().lower()  # "global" | "smart"
BOT_TZ        = os.getenv("BOT_TZ", "Europe/Berlin")

# Кандидаты бенчмарков для SMART-режима (корреляционный выбор по тикеру)
BENCH_CANDIDATES = [
    "SPY","QQQ","IWM","DIA",
    "XLK","XLV","XLY","XLP","XLE","XLF","XLI","XLB","XLRE","XLU"
]

# ──────────────────────────────────────────────────────────────────────────────
# === UNIVERSES ===
# SP250 — фиксированный список акций (S&P 500 Top-250 по весу)
# ETFALL — 50 обычных ETF (без плеча/инверса)
# ETFX — расширенный список (плечевые/инверсные/спец)
# ──────────────────────────────────────────────────────────────────────────────

SP250 = [
    # 1..250 по кумулятивному весу (Slickcharts snapshot ~ Sep-2025)
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA",
    "JPM","WMT","V","LLY","ORCL","MA","NFLX","XOM","JNJ","COST",
    "HD","BAC","PLTR","ABBV","PG","CVX","KO","GE","TMUS","UNH",
    "CSCO","AMD","WFC","PM","CRM","MS","ABT","AXP","IBM","GS",
    "LIN","MCD","DIS","RTX","MRK","T","PEP","CAT","UBER","NOW",
    "VZ","INTU","TMO","TXN","BKNG","C","BA","BLK","SCHW","QCOM",
    "ANET","ISRG","SPGI","GEV","ACN","BSX","AMGN","TJX","ADBE","SYK",
    "NEE","DHR","COF","PGR","LOW","PFE","GILD","HON","ETN","BX",
    "MU","APH","UNP","DE","AMAT","PANW","LRCX","CMCSA","KKR","ADI",
    "COP","ADP","MDT","KLAC","NKE","MO","WELL","SNPS","CB","INTC",
    "LMT","CRWD","PLD","DASH","SO","MMC","ICE","VRTX","SBUX","RCL",
    "CEG","PH","CME","BMY","CDNS","AMT","DUK","HCA","CVS","TT",
    "MCO","SHW","WM","ORLY","GD","MCK","CTAS","NOC","MMM","DELL",
    "NEM","PNC","CI","ABNB","MDLZ","AON","TDG","MSI","ECL","COIN",
    "APO","AJG","ITW","EQIX","USB","FI","BK","EMR","UPS","RSG",
    "MAR","ELV","WMB","AZO","HWM","JCI","ZTS","EOG","CL","ADSK",
    "PYPL","APD","HLT","VST","FCX","NSC","WDAY","REGN","URI","TRV",
    "TEL","MNST","CSX","TFC","FTNT","KMI","AEP","NXPI","SPG","AXON",
    "GLW","DLR","AFL","FAST","ROP","COR","CMG","PWR","GM","CARR",
    "BDX","SLB","CMI","MPC","FDX","NDAQ","MET","PSX","SRE","O",
    "ALL","PCAR","LHX","IDXX","PSA","D","DHI","CTVA","PAYX","SQ",
    "AMP","GWW","CBRE","ROST"
]

# ETFALL — 50 обычных (без плеча/инверса), по объёму/ликвидности
ETFALL = [
    "SPY","VOO","IVV","VTI","QQQ","IWM","DIA","RSP",
    "EFA","IEFA","EEM","IEMG",
    "AGG","BND","LQD","HYG","TLT","IEF","SHY","TIP",
    "GLD","IAU","SLV","GDX",
    "XLK","XLV","XLY","XLP","XLF","XLE","XLI","XLB","XLRE","XLU",
    "SMH","SOXX","XBI","IBB","XOP","XME","KRE","XHB","IYR","VNQ",
    "ARKK","BITO","LIT","MAGS","VUG","VTV"
]

# ETFX — 27 плечевых/инверсных (включая твои добавки)
ETFX = [
    # Индексы
    "UPRO","SPXL","SSO","SDS","SPXS","UDOW","SDOW",
    "TQQQ","SQQQ","QLD","QID",
    "TNA","TZA",
    "WEBL","WEBS",
    # Сектора/тематические
    "SOXL","SOXS","TECL","TECS","FNGU","FNGD",
    "LABU","LABD","NUGT","DUST",
    # Коммодити/крипто
    "UCO","SCO","BITX","USD","TSLL"
]

# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────

# Приведение тикеров к формату Yahoo (BRK.B → BRK-B)
def to_yahoo(t: str) -> str:
    t = t.strip().upper()
    t = t.replace("/", "-")
    if "." in t:
        t = t.replace(".", "-")
    return t

def from_yahoo(t: str) -> str:
    return t.replace("-", ".")

# подгрузка AI-оценок: CSV с колонками ticker, ai_score (0..1)
def load_ai_map(url: str) -> dict:
    if not url or url.lower() == "none":
        return {}
    try:
        df = pd.read_csv(url)
        df["ticker"] = df["ticker"].astype(str).str.upper()
        m = {k: float(v) for k, v in zip(df["ticker"], df["ai_score"])}
        return m
    except Exception:
        return {}

AI_MAP = load_ai_map(AI_CSV_URL)

# батчевый загрузчик котировок (чтобы не упереться в лимиты)
async def fetch_history(tickers, period="400d", interval="1d") -> pd.DataFrame:
    tickers_yy = [to_yahoo(t) for t in tickers]
    # yfinance умеет пачкой:
    data = yf.download(
        tickers=" ".join(tickers_yy),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    # Приведём к единому DataFrame Close
    closes = {}
    vols   = {}
    for tt in tickers:
        yy = to_yahoo(tt)
        try:
            df = data[yy] if isinstance(data.columns, pd.MultiIndex) else data
            closes[tt] = df["Close"].rename(tt)
            vols[tt]   = df["Volume"].rename(tt)
        except Exception:
            # тикер не загрузился
            pass
    close_df = pd.concat(closes.values(), axis=1) if closes else pd.DataFrame()
    vol_df   = pd.concat(vols.values(), axis=1)   if vols   else pd.DataFrame()
    return close_df, vol_df

def pct_change(df: pd.DataFrame, n: int) -> pd.Series:
    return df.iloc[-1] / df.iloc[-n-1] - 1.0

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def zscore(x: float, mean: float, std: float) -> float:
    if std <= 1e-9: return 0.0
    return (x - mean)/std

# выбор бенча: global или smart-корреляционный
def pick_benchmark(returns_tbl: pd.DataFrame, ticker: str, mode: str) -> str:
    if mode != "smart":
        return BENCH_DEFAULT
    best_t, best_r = BENCH_DEFAULT, -2.0
    # берём последние ~63 дневных доходностей
    if ticker not in returns_tbl.columns:
        return BENCH_DEFAULT
    y = returns_tbl[ticker].dropna()
    if len(y) < 40:
        return BENCH_DEFAULT
    for b in BENCH_CANDIDATES:
        if b not in returns_tbl.columns: 
            continue
        x = returns_tbl[b].dropna()
        n = min(len(x), len(y))
        if n < 40: 
            continue
        r = np.corrcoef(x.values[-n:], y.values[-n:])[0,1]
        if np.isfinite(r) and r > best_r:
            best_r, best_t = r, b
    return best_t

# основной скоринг
def score_universe(close: pd.DataFrame, vol: pd.DataFrame, universe: list,
                   bench_mode: str) -> pd.DataFrame:
    # очистка: только доступные колонны
    u = [t for t in universe if t in close.columns]
    if not u:
        return pd.DataFrame()

    # добираем бэнчи для корреляции
    bench_set = set([BENCH_DEFAULT] + BENCH_CANDIDATES)
    need_bench = [b for b in bench_set if b not in close.columns]
    if need_bench:
        add_close, add_vol = asyncio.run(fetch_history(need_bench))
        close = pd.concat([close, add_close], axis=1)
    # доходности (daily)
    ret_daily = close.pct_change().dropna()

    rows = []
    for t in u:
        px = close[t].dropna()
        if len(px) < 210:   # минимум данных
            continue

        # бенч для тикера
        b = pick_benchmark(ret_daily, t, bench_mode)
        bx = close[b].dropna()
        if len(bx) < 210:
            b = BENCH_DEFAULT
            bx = close[b].dropna()

        # RS21/RS63: доходность тикера минус бенча
        try:
            rs21 = (px.iloc[-1]/px.iloc[-22]-1) - (bx.iloc[-1]/bx.iloc[-22]-1)
            rs63 = (px.iloc[-1]/px.iloc[-64]-1) - (bx.iloc[-1]/bx.iloc[-64]-1)
        except Exception:
            continue

        # тренд/подтверждение
        ema20  = ema(px, 20).iloc[-1]
        ema50  = ema(px, 50).iloc[-1]
        ema200 = ema(px, 200).iloc[-1]
        trend_ok = 1.0 if (px.iloc[-1] > ema20 > ema50 > ema200) else 0.0

        # близость к 252-дн. хай/лоу
        window = px.tail(252)
        if len(window) < 30: 
            continue
        hi = window.max(); lo = window.min()
        prox_hi = float(px.iloc[-1]/hi)  # чем ближе к 1, тем лучше
        # волатильность (стд-дев дневных)
        vol_t = float(window.pct_change().std())
        vol_b = float(bx.tail(len(window)).pct_change().std())
        vol_rel = 1.0 - min(1.0, max(0.0, (vol_t / (vol_b+1e-9))))  # меньше волы — лучше

        # ConfirmScore в [0..1]
        base = (0.30*trend_ok +
                0.30*max(0.0, min(1.0, (prox_hi-0.85)/0.15)) +  # если цена в верхних 15%
                0.20*max(0.0, min(1.0, rs21/0.10)) +            # 10% за 21д как 1.0
                0.20*vol_rel)

        # AI-оценка (если есть)
        ai_s = float(AI_MAP.get(t, AI_MAP.get(from_yahoo(t), 0.0)))
        final_score = (1.0 - W_AI)*base + W_AI*ai_s

        rows.append({
            "ticker": t, "bench": b,
            "score": final_score,
            "RS21": rs21, "RS63": rs63,
            "ConfirmScore": base,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # фильтр “зелёный светофор”: RS21>0 и RS63>0
    df = df[(df["RS21"] > 0) & (df["RS63"] > 0)]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df

def fmt_row(i, r):
    return (f"{i}) {r['ticker']} | RS21={r['RS21']:+.2%} | RS63={r['RS63']:+.2%} "
            f"| score={r['score']:.3f} | bench={r['bench']} | "
            f"Confirm={r['ConfirmScore']:.3f}")

async def run_rank(universe: list, topn: int = 10, bench_mode: str = None) -> str:
    bench_mode = bench_mode or BENCH_MODE
    txt_head = ""
    if universe is SP250:
        txt_head = "SP250"
    elif universe is ETFALL:
        txt_head = "ETFALL"
    elif universe is ETFX:
        txt_head = "ETFX"
    else:
        txt_head = "UNIVERSE"

    # Загрузка данных
    close, vol = await fetch_history(universe + [BENCH_DEFAULT] + BENCH_CANDIDATES)
    if close.empty:
        return "Не удалось загрузить котировки. Попробуй ещё раз."

    df = score_universe(close, vol, universe, bench_mode)
    if df.empty:
        return f"{txt_head}: подходящих бумаг не найдено (фильтр RS21/RS63>0)."

    df_top = df.head(topn)
    lines = [f"Топ {topn} — {txt_head} (режим бенча: {bench_mode})"]
    for i, r in enumerate(df_top.itertuples(index=False), start=1):
        lines.append(fmt_row(i, r._asdict()))
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# Telegram handlers
# ──────────────────────────────────────────────────────────────────────────────
async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Я на связи ✅")

async def sp5005_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Считаю SP250… это ~20–30 сек на первом запуске.")
    txt = await run_rank(SP250, topn=5)
    await update.message.reply_text(txt)

async def sp50010_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Считаю SP250… это ~20–30 сек на первом запуске.")
    txt = await run_rank(SP250, topn=10)
    await update.message.reply_text(txt)

async def etfall_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Считаю ETFALL…")
    txt = await run_rank(ETFALL, topn=10)
    await update.message.reply_text(txt)

async def etfx_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Считаю ETFX…")
    txt = await run_rank(ETFX, topn=10)
    await update.message.reply_text(txt)

# полный рейтинг по группе (осторожно — длинно)
async def etf_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = (context.args[0].lower() if context.args else "etfall")
    group = ETFALL if arg == "etfall" else (ETFX if arg == "etfx" else ETFALL)
    await update.message.reply_text(f"Считаю {arg.upper()} полный рейтинг…")
    close, vol = await fetch_history(group + [BENCH_DEFAULT] + BENCH_CANDIDATES)
    df = score_universe(close, vol, group, BENCH_MODE)
    if df.empty:
        await update.message.reply_text("Пусто по фильтру.")
        return
    lines = [f"Полный рейтинг {arg.upper()} (top {len(df)})"]
    for i, r in enumerate(df.itertuples(index=False), start=1):
        lines.append(fmt_row(i, r._asdict()))
        if i % 50 == 0:
            await update.message.reply_text("\n".join(lines))
            lines = []
    if lines:
        await update.message.reply_text("\n".join(lines))

# Настройки/сервис
async def benchmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BENCH_MODE
    if not context.args:
        await update.message.reply_text(f"Режим бенчмарка: {BENCH_MODE}")
        return
    m = context.args[0].strip().lower()
    if m not in ("global","smart"):
        await update.message.reply_text("Формат: /benchmode <global|smart>")
        return
    BENCH_MODE = m
    await update.message.reply_text(f"Режим бенча установлен: {BENCH_MODE}")

async def setbenchmark_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BENCH_DEFAULT
    if not context.args:
        await update.message.reply_text("Формат: /setbenchmark <TICKER>")
        return
    BENCH_DEFAULT = context.args[0].upper()
    await update.message.reply_text(f"Глобальный бенч: {BENCH_DEFAULT}")

async def benchcandidates_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Кандидаты SMART:\n" + " ".join(BENCH_CANDIDATES))

async def getbench_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Текущие настройки бенча:\nMODE={BENCH_MODE}\nDEFAULT={BENCH_DEFAULT}"
    )

async def setaisource_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_CSV_URL, AI_MAP
    if not context.args:
        await update.message.reply_text("Формат: /setaisource <csv|none> — источник AI скор (ticker,ai_score)")
        return
    AI_CSV_URL = context.args[0].strip()
    AI_MAP = load_ai_map(AI_CSV_URL)
    await update.message.reply_text(f"Источник AI: {AI_CSV_URL}")

async def setw_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global W_AI
    if not context.args:
        await update.message.reply_text(f"Текущий W_AI={W_AI}. Формат: /setw_ai <0..1>")
        return
    try:
        W_AI = max(0.0, min(1.0, float(context.args[0])))
        await update.message.reply_text(f"W_AI установлен: {W_AI}")
    except Exception:
        await update.message.reply_text("Ошибка: нужен числовой 0..1")

# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_TOKEN env var.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("ping",          ping_cmd))
    app.add_handler(CommandHandler("sp5005",        sp5005_cmd))
    app.add_handler(CommandHandler("sp50010",       sp50010_cmd))
    app.add_handler(CommandHandler("etfall",        etfall_cmd))
    app.add_handler(CommandHandler("etfx",          etfx_cmd))
    app.add_handler(CommandHandler("etf",           etf_cmd))

    app.add_handler(CommandHandler("benchmode",     benchmode_cmd))
    app.add_handler(CommandHandler("setbenchmark",  setbenchmark_cmd))
    app.add_handler(CommandHandler("benchcandidates", benchcandidates_cmd))
    app.add_handler(CommandHandler("getbench",      getbench_cmd))
    app.add_handler(CommandHandler("setaisource",   setaisource_cmd))
    app.add_handler(CommandHandler("setw_ai",       setw_ai_cmd))

    print("Bot started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
