# bot_sp500_fixed.py
# Telegram-бот: AI-моментум + ConfirmScore для SP250/ETFALL/ETFX
# ВАЖНО: здесь убраны «болтушки» про 20–30 сек и сделана надёжная загрузка цен
# Зависимости: python-telegram-bot==20.7, yfinance==0.2.43, pandas==2.2.2, numpy==1.26.4, lxml, bs4, requests

import os
import math
import time
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

# telegram-bot
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# -------------------- ENV --------------------
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
BENCH_DEFAULT = os.getenv("BENCH_DEFAULT", "SPY").strip().upper()
BENCH_MODE = os.getenv("BENCH_MODE", "smart").strip().lower()   # "global" | "smart"
BOT_TZ = os.getenv("BOT_TZ", "Europe/Berlin")

# -------------------- Параметры загрузки котировок --------------------
YF_OPTS = dict(period="180d", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)

# батч размер и паузы (мягко уходим от 429)
BATCH_SIZE = 80                 # сколько тикеров за один мультизапрос yf.download
BATCH_PAUSE = 1.2               # сек между мульти-батчами
FALLBACK_GAP = 0.35             # сек между одиночными фоллбэками
HTTP_TIMEOUT = 12

# при желании можно подменять «капризные» тикеры на аналоги
ALIAS = {
    # "BITX": "BITB",
    # "USD": "USD",
}

# кандидаты бенчмарков для SMART-режима
BENCH_CANDIDATES = [
    "SPY","QQQ","IWM","DIA",
    "XLK","XLY","XLF","XLP","XLV","XLU","XLB","XLC","XLE","XLI",
]

# -------------------- анти-дубли --------------------
RECENT_MSG_IDS = deque(maxlen=500)

def seen_message(update: Update) -> bool:
    mid = getattr(update.message, "message_id", None)
    if mid is None:
        return False
    if mid in RECENT_MSG_IDS:
        return True
    RECENT_MSG_IDS.append(mid)
    return False

# -------------------- котировки --------------------
def _parse_close_from_multi(df: pd.DataFrame, orig_tickers: list[str]) -> pd.DataFrame:
    """
    Из результата yf.download(list_of_tickers) вытащить матрицу Close.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance для мульти выдает столбцы вида ('Close','AAPL'), ('Close','MSFT'), ...
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            close = df['Close'].copy()
            # сохранить порядок тикеров насколько возможно
            cols = [c for c in orig_tickers if c in close.columns]
            close = close[cols] if cols else close
            return close
        else:
            return pd.DataFrame()
    else:
        # одиночный фрейм, а не мульти
        if 'Close' in df.columns:
            # имя колонки одно — подменим на исходный тикер, если он один
            out = df[['Close']].copy()
            if len(orig_tickers) == 1:
                out.columns = [orig_tickers[0]]
            return out
        return pd.DataFrame()

def _fallback_yahoo_csv(ticker: str) -> pd.Series | None:
    """
    Запасной путь: CSV /v7/finance/download с хедерами.
    """
    t = ALIAS.get(ticker, ticker)
    url = "https://query1.finance.yahoo.com/v7/finance/download/{}".format(t)
    params = {
        "period1": int(time.time()) - 210*24*3600,
        "period2": int(time.time()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true"
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv, */*;q=0.1",
        "Connection": "close",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
        if r.status_code != 200 or not r.text or "Timestamp" in r.text:
            return None
        df = pd.read_csv(StringIO(r.text))
        if df.empty or "Close" not in df.columns:
            return None
        s = pd.Series(df["Close"].values, index=pd.to_datetime(df["Date"]), name=ticker)
        return s
    except Exception as e:
        print(f"[yahoo-csv] fail {ticker}: {e}")
        return None

def _fetch_one_close_strict(ticker: str) -> pd.Series | None:
    """
    Одна попытка через yfinance.download; если пусто — CSV фоллбэк.
    """
    t = ALIAS.get(ticker, ticker)
    try:
        df = yf.download(t, **YF_OPTS)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.get_level_values(0):
                    s = df['Close']
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    return s.rename(ticker)
            elif 'Close' in df.columns:
                return df['Close'].rename(ticker)
    except Exception as e:
        print(f"[yfinance] single fail {ticker}: {e}")

    # fallback
    s = _fallback_yahoo_csv(t)
    return s

async def load_close_matrix(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    1) Пытаемся загрузить ВСЕ тикеры мультибатчами (минимум запросов)
    2) Для недогруженных — индивидуальный фоллбэк с паузой
    """
    need = [ALIAS.get(t, t) for t in tickers]
    got = []
    frames = []
    for i in range(0, len(need), BATCH_SIZE):
        batch = need[i:i+BATCH_SIZE]
        try:
            df = yf.download(batch, **YF_OPTS)
            close = _parse_close_from_multi(df, batch)
            if not close.empty:
                frames.append(close)
                got.extend(list(close.columns))
        except Exception as e:
            print(f"[yfinance-multi] fail batch {batch[:3]}..: {e}")
        await asyncio.sleep(BATCH_PAUSE)

    closes = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    closes = closes.loc[:, ~closes.columns.duplicated()] if not closes.empty else closes

    # кто не загрузился — пробуем по одному
    missing_orig = [t for t in tickers if ALIAS.get(t, t) not in got]
    truly_bad = []
    for tk in missing_orig:
        s = _fetch_one_close_strict(tk)
        if s is None or s.dropna().empty:
            truly_bad.append(tk)
        else:
            closes = s.to_frame() if closes.empty else closes.join(s, how="outer")
        await asyncio.sleep(FALLBACK_GAP)

    if closes.empty:
        return pd.DataFrame(), truly_bad
    closes = closes.sort_index().dropna(how="all").dropna(axis=1, how="all")
    return closes, truly_bad

# -------------------- Вспомогалки --------------------
def pct_change(series: pd.Series, periods: int) -> float:
    if len(series) < periods + 1:
        return np.nan
    a = float(series.iloc[-periods-1])
    b = float(series.iloc[-1])
    if a == 0 or math.isnan(a) or math.isnan(b):
        return np.nan
    return b / a - 1.0

def confirm_score(pr: pd.Series) -> float:
    if len(pr) < 200:
        return 0.0
    close = pr.iloc[-1]
    ma50  = pr.rolling(50).mean().iloc[-1]
    ma100 = pr.rolling(100).mean().iloc[-1]
    ma150 = pr.rolling(150).mean().iloc[-1]
    ma200 = pr.rolling(200).mean().iloc[-1]
    score = 0
    for ma in (ma50, ma100, ma150, ma200):
        if close > ma:
            score += 1
    return score / 4.0

def rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average")

def pick_benchmark_for(ticker: str, closes: pd.DataFrame) -> str:
    candidates = [b for b in BENCH_CANDIDATES if b in closes.columns]
    if not candidates:
        return BENCH_DEFAULT
    s = closes[ticker].pct_change().dropna()
    best = BENCH_DEFAULT
    best_corr = -2
    for b in candidates:
        sb = closes[b].pct_change().dropna()
        corr = s.corr(sb)
        if pd.notna(corr) and corr > best_corr:
            best_corr = corr
            best = b
    return best

def composite_score(rs21_rel: pd.Series, rs63_rel: pd.Series, conf: pd.Series) -> pd.Series:
    r1 = rank_pct(rs21_rel.fillna(-1))
    r2 = rank_pct(rs63_rel.fillna(-1))
    rc = conf.fillna(0)
    return 0.45*r1 + 0.35*r2 + 0.20*rc

def fmt_row(i, tk, row):
    return f"{i}) {tk} | score={row['score']:.4f} | RS21={row['rs21_rel']:+.2%} | RS63={row['rs63_rel']:+.2%} | conf={row['conf']:.2f}"

# -------------------- Ранжирование вселенной --------------------
async def universe_rank(update: Update, context: ContextTypes.DEFAULT_TYPE, universe_name: str, tickers: list[str], top_n: int | None):
    # анти-дубль
    if seen_message(update):
        return

    # добавим бенчмарки к загрузке
    bench_need = {BENCH_DEFAULT}
    if BENCH_MODE == "smart":
        bench_need.update(BENCH_CANDIDATES)
    use_tickers = sorted(set([t.upper() for t in tickers] + list(bench_need)))

    closes, bad = await load_close_matrix(use_tickers)

    # отдадим предупреждение, но не валимся
    miss_for_universe = [t for t in tickers if t not in closes.columns]
    bad_all = sorted(set(bad + miss_for_universe))
    if bad_all:
        await update.message.reply_text("⚠️ Пропущены тикеры без данных: " + ", ".join(bad_all))

    # реальная вселенная
    uni = [t for t in tickers if t in closes.columns]
    if not uni:
        await update.message.reply_text("Нет данных для ранжирования (источник котировок недоступен).")
        return

    rows = []
    for tk in uni:
        pr = closes[tk].dropna()
        if pr.empty:
            continue

        bench = pick_benchmark_for(tk, closes) if BENCH_MODE == "smart" else BENCH_DEFAULT
        if bench not in closes.columns:
            bench = BENCH_DEFAULT

        rs21 = pct_change(pr, 21)
        rs63 = pct_change(pr, 63)

        if bench in closes.columns:
            bpr = closes[bench].dropna()
            brs21 = pct_change(bpr, 21)
            brs63 = pct_change(bpr, 63)
            rs21_rel = (rs21 - brs21) if pd.notna(rs21) and pd.notna(brs21) else np.nan
            rs63_rel = (rs63 - brs63) if pd.notna(rs63) and pd.notna(brs63) else np.nan
        else:
            rs21_rel, rs63_rel = rs21, rs63

        conf = confirm_score(pr)

        rows.append({
            "ticker": tk,
            "bench": bench,
            "rs21_rel": rs21_rel,
            "rs63_rel": rs63_rel,
            "conf": conf,
        })

    df = pd.DataFrame(rows).set_index("ticker")
    if df.empty:
        await update.message.reply_text("Нет валидных данных после загрузки котировок.")
        return

    df["score"] = composite_score(df["rs21_rel"], df["rs63_rel"], df["conf"])
    df_sorted = df.sort_values("score", ascending=False)

    if top_n:
        df_show = df_sorted.head(top_n)
        lines = [f"Топ {top_n} — {universe_name}, сортировка по score:"]
        for i, (tk, row) in enumerate(df_show.iterrows(), 1):
            lines.append(fmt_row(i, tk, row))
        await update.message.reply_text("\n".join(lines))
    else:
        header = f"Полный рейтинг {universe_name}, сортировка по score:"
        await update.message.reply_text(header)
        chunk = []
        for i, (tk, row) in enumerate(df_sorted.iterrows(), 1):
            chunk.append(fmt_row(i, tk, row))
            if len(chunk) == 30:
                await update.message.reply_text("\n".join(chunk))
                chunk = []
        if chunk:
            await update.message.reply_text("\n".join(chunk))

# -------------------- ВСЕЛЕННЫЕ --------------------
# Вставь сюда свой «жёсткий» список (127 или 250) — одной строкой, как тебе нужно:
SP250 = """
AAPL MSFT NVDA AMZN GOOGL GOOG META AVGO LLY BRK.B JPM XOM UNH JNJ V MA TSLA ORCL PG COST MRK
ABBV HD PEP KO BAC WMT NFLX ADBE CRM TMO CSCO ACN DHR LIN WFC AMD MCD TXN ABT IBM AXP CAT GE
CVX AMGN NKE COP LMT QCOM PM NOW HON SBUX T RTX TGT SPGI GS ADP MDT LOW BKNG PGR C GILD ELV
DE SYK MU PLD ISRG LRCX INTU ZTS PANW MS BLK REGN USB MO PEP^ QQQ^ (заметка: заменишь на свой окончательный список)
""".split()

# ETFALL — базовые «обычные» ETF (добавлен MSTR по твоей просьбе)
ETFALL = [
    "SPY","VOO","IVV","RSP","QQQ","DIA","IWM","IWB","IWR","IJR",
    "XLK","XLY","XLF","XLE","XLI","XLP","XLV","XLU","XLB","XLC",
    "TLT","IEF","HYG","LQD","AGG","BND",
    "GLD","IAU","SLV","GDX",
    "SMH","SOXX","IBB","XBI","ITB","XHB","IYT","XOP","XME","XRT",
    "BITO","LIT","MAGS","ARKK","KRE","KBE",
    "MSTR"  # <— добавлен
]

# ETFX — плечевые/инверсные/одиночные (с твоими доп. тикерами)
ETFX = [
    "TQQQ","SQQQ","SPXL","SPXS","UPRO","SPXU","UDOW","SDOW",
    "SOXL","SOXS","TECL","TECS","FNGU","FNGD",
    "LABU","LABD","TNA","TZA",
    "UCO","SCO","BOIL","KOLD",
    "NUGT","DUST","UVXY",
    "TSLL","WEBL","UPRO","USD","BITX",
    "GGLL","AAPU","FBL","MSFU","AMZU","NVDL","CONL"
]

# -------------------- КОМАНДЫ --------------------
async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update):
        return
    tz = ZoneInfo(BOT_TZ)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    await update.message.reply_text(f"Я на связи ✅\n{now} {BOT_TZ}")

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update):
        return
    me = await context.bot.get_me()
    await update.message.reply_text(
        f"Bot: @{me.username} (id={me.id})\nMODE={BENCH_MODE}  DEFAULT_BENCH={BENCH_DEFAULT}\nTZ={BOT_TZ}"
    )

async def sp5005_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await universe_rank(update, context, "SP250", SP250, top_n=5)

async def sp50010_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await universe_rank(update, context, "SP250", SP250, top_n=10)

async def etfall_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await universe_rank(update, context, "ETFALL", ETFALL, top_n=10)

async def etfx_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await universe_rank(update, context, "ETFX", ETFX, top_n=10)

async def etf_full_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update):
        return
    args = [a.lower() for a in context.args] if context.args else []
    name = "etfall"
    tickers = ETFALL
    if args and args[0] in ("etfall","etfx"):
        name = args[0]
    if name == "etfx":
        tickers = ETFX
    await universe_rank(update, context, name.upper(), tickers, top_n=None)

async def benchmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update):
        return
    global BENCH_MODE
    if context.args:
        mode = context.args[0].lower().strip()
        if mode in ("global","smart"):
            BENCH_MODE = mode
            await update.message.reply_text(f"Режим бенчмарка: {BENCH_MODE}")
            return
    await update.message.reply_text("Формат: /benchmode <global|smart>")

async def setbenchmark_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update):
        return
    global BENCH_DEFAULT
    if not context.args:
        await update.message.reply_text("Формат: /setbenchmark <тикер>")
        return
    BENCH_DEFAULT = context.args[0].upper()
    await update.message.reply_text(f"Глобальный бенч: {BENCH_DEFAULT}")

async def benchcandidates_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update):
        return
    await update.message.reply_text("Кандидаты SMART:\n" + " ".join(BENCH_CANDIDATES))

# -------------------- MAIN --------------------
def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_TOKEN env var.")
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("diag", diag_cmd))

    app.add_handler(CommandHandler("sp5005", sp5005_cmd))
    app.add_handler(CommandHandler("sp50010", sp50010_cmd))
    app.add_handler(CommandHandler("etfall", etfall_cmd))
    app.add_handler(CommandHandler("etfx", etfx_cmd))
    app.add_handler(CommandHandler("etf", etf_full_cmd))

    app.add_handler(CommandHandler("benchmode", benchmode_cmd))
    app.add_handler(CommandHandler("setbenchmark", setbenchmark_cmd))
    app.add_handler(CommandHandler("benchcandidates", benchcandidates_cmd))

    # drop_pending_updates=True — чтобы при рестарте не «доглатывал» старые
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
