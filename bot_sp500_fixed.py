# bot_sp500_fixed.py
# Telegram-бот: Моментум + ConfirmScore для SP250/ETFALL/ETFX (БЕЗ AI)
# Требования: python-telegram-bot==20.7, yfinance==0.2.43, pandas==2.2.2, numpy==1.26.4, lxml, bs4, requests

import os
import io
import time
import math
import random
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest
from telegram.error import Conflict

# -------------------- ENV --------------------
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
BENCH_DEFAULT = os.getenv("BENCH_DEFAULT", "SPY").strip().upper()
BENCH_MODE = os.getenv("BENCH_MODE", "smart").strip().lower()  # "global" | "smart"
BOT_TZ = os.getenv("BOT_TZ", "Europe/Berlin")

# Кандидаты бенчмарков для SMART-режима
BENCH_CANDIDATES = [
    "SPY","QQQ","IWM","DIA",
    "XLK","XLY","XLF","XLP","XLV","XLU","XLB","XLC","XLE","XLI",
]

# -------------------- НАДЁЖНАЯ ЗАГРУЗКА ЦЕН --------------------
YF_OPTS = dict(period="180d", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=False)

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
]

def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": random.choice(UA_POOL)})
    return s

# алиасы капризных тикеров Yahoo
ALIAS: Dict[str, str] = {
    "BRK.B": "BRK-B",  # Berkshire Hathaway (класс B)
    "BF.B":  "BF-B",
    "BITX":  "BITB",
    "XYZ":   "SQ",     # Block, Inc.: у Yahoo тикер SQ
}

def _normalize_yf_df(df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            s = df['Close']
            if isinstance(s, pd.DataFrame):
                if ticker in s.columns:
                    s = s[ticker]
                else:
                    s = s.iloc[:, 0]
            return s
        return None
    else:
        if 'Close' not in df.columns:
            return None
        return df['Close']

def _fetch_from_yahoo_csv(ticker: str) -> Optional[pd.Series]:
    t = ALIAS.get(ticker, ticker)
    end = int(time.time())
    start = end - 400 * 24 * 3600
    url = (f"https://query1.finance.yahoo.com/v7/finance/download/{t}"
           f"?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true")
    try:
        r = _new_session().get(url, timeout=10)
        if r.status_code == 200 and "Date,Open,High,Low,Close" in r.text:
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty or "Date" not in df.columns:
                return None
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            s = df[col].astype(float)
            return s if not s.dropna().empty else None
    except Exception as e:
        print(f"[yahoo-csv] fail {ticker}: {e}")
    return None

def _fetch_from_stooq(ticker: str) -> Optional[pd.Series]:
    sym = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty or "Close" not in df.columns or "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).set_index("Date").sort_index()
        s = df["Close"].astype(float)
        return s if not s.dropna().empty else None
    except Exception as e:
        print(f"[stooq] fail {ticker}: {e}")
        return None

def _fetch_one_close(ticker: str) -> Optional[pd.Series]:
    t = ALIAS.get(ticker, ticker)
    sess = _new_session()
    try:
        df = yf.download(t, session=sess, **YF_OPTS)
        s = _normalize_yf_df(df, t)
        if s is not None and not s.dropna().empty:
            return s.rename(ticker)
    except Exception as e:
        print(f"[yfinance] fail {ticker} (1st): {e}")
    try:
        alt = dict(YF_OPTS); alt["period"] = "365d"
        df = yf.download(t, session=_new_session(), **alt)
        s = _normalize_yf_df(df, t)
        if s is not None and not s.dropna().empty:
            return s.rename(ticker)
    except Exception as e:
        print(f"[yfinance] fail {ticker} (2nd): {e}")
    s = _fetch_from_yahoo_csv(t)
    if s is not None and not s.dropna().empty:
        return s.rename(ticker)
    s = _fetch_from_stooq(t)
    if s is not None and not s.dropna().empty:
        return s.rename(ticker)
    return None

def _bulk_chunk_download(ticks: list[str], chunk: int = 8, tries: int = 3, delay: float = 3.0) -> Optional[pd.DataFrame]:
    for attempt in range(tries):
        colls, ok_any = [], False
        sess = _new_session()
        for i in range(0, len(ticks), chunk):
            part = ticks[i:i+chunk]
            syms = " ".join([ALIAS.get(t, t) for t in part])
            try:
                df = yf.download(syms, session=sess, **YF_OPTS)
                if isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.get_level_values(0):
                    blk = df["Close"].copy()
                    rename_map = {ALIAS.get(t, t): t for t in part}
                    blk = blk.rename(columns=rename_map)
                    blk = blk.loc[:, [t for t in part if t in blk.columns]]
                    if not blk.empty:
                        colls.append(blk); ok_any = True
                elif "Close" in df.columns and len(part) == 1:
                    s = df["Close"].rename(part[0])
                    colls.append(pd.concat([s], axis=1)); ok_any = True
            except Exception as e:
                print(f"[bulk-chunk] fail {part}: {e}")
            time.sleep(0.5)
        if ok_any:
            out = pd.concat(colls, axis=1)
            return out.dropna(how="all")
        time.sleep(delay)
    return None

async def load_close_matrix(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    data: List[pd.Series] = []
    bad: List[str] = []
    for tk in tickers:
        s = await asyncio.to_thread(_fetch_one_close, tk)
        if s is None or s.dropna().empty:
            bad.append(tk)
        else:
            data.append(s)
        pause = 0.07 if len(tickers) <= 50 else 0.12 if len(tickers) <= 100 else 0.18
        await asyncio.sleep(pause)

    if data:
        closes = pd.concat(data, axis=1).dropna(how="all").dropna(axis=1, how="all")
        return closes, bad

    bulk = _bulk_chunk_download(tickers, chunk=8, tries=4, delay=4.0)
    if bulk is not None and not bulk.empty:
        still_bad = [t for t in tickers if t not in bulk.columns]
        return bulk, still_bad

    raise RuntimeError(f"Не удалось загрузить ни один тикер из: {tickers}")

# -------------------- МЕТРИКИ --------------------
def pct_change(series: pd.Series, periods: int) -> float:
    if len(series) < periods + 1:
        return np.nan
    a = float(series.iloc[-periods-1]); b = float(series.iloc[-1])
    if a == 0 or math.isnan(a) or math.isnan(b):
        return np.nan
    return b / a - 1.0

def confirm_score(pr: pd.Series) -> float:
    if len(pr) < 200:
        return 0.0
    close = pr.iloc[-1]
    ma50 = pr.rolling(50).mean().iloc[-1]
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
    best = BENCH_DEFAULT; best_corr = -2
    for b in candidates:
        sb = closes[b].pct_change().dropna()
        corr = s.corr(sb)
        if pd.notna(corr) and corr > best_corr:
            best_corr = corr; best = b
    return best

def fmt_row(i, tk, row):
    return f"{i}) {tk} | score={row['score']:.4f} | RS21={row['rs21_rel']:+.2%} | RS63={row['rs63_rel']:+.2%} | conf={row['conf']:.2f}"

# -------------------- РАНЖИРОВАНИЕ --------------------
async def universe_rank(update: Update, context: ContextTypes.DEFAULT_TYPE, universe_name: str, tickers: List[str], top_n: Optional[int]):
    # (убрано сообщение «Считаю ... 20–30 сек»)

    bench_need = {BENCH_DEFAULT}
    if BENCH_MODE == "smart":
        bench_need.update(BENCH_CANDIDATES)
    use_tickers = sorted(set([t.upper() for t in tickers] + list(bench_need)))

    try:
        closes, bad = await load_close_matrix(use_tickers)
    except Exception as e:
        await update.message.reply_text(f"Не получилось загрузить котировки. Попробуй ещё раз.\nДетали: {e}")
        return

    missing = [t for t in tickers if t not in closes.columns]
    bad_all = sorted(set(bad + missing))
    if bad_all:
        await update.message.reply_text("⚠️ Пропущены тикеры без истории: " + ", ".join(bad_all))

    uni = [t for t in tickers if t in closes.columns]
    if not uni:
        await update.message.reply_text("Нечего ранжировать: нет данных по вселенной.")
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

    r1 = rank_pct(df["rs21_rel"].fillna(-1))
    r2 = rank_pct(df["rs63_rel"].fillna(-1))
    rc = df["conf"].fillna(0)
    df["score"] = 0.45*r1 + 0.35*r2 + 0.20*rc

    df_sorted = df.sort_values("score", ascending=False)
    if top_n:
        df_show = df_sorted.head(top_n)
        lines = [f"Топ {top_n} — {universe_name}, сортировка по score:"]
        for i, (tk, row) in enumerate(df_show.iterrows(), 1):
            lines.append(fmt_row(i, tk, row))
        await update.message.reply_text("\n".join(lines))
    else:
        await update.message.reply_text(f"Полный рейтинг {universe_name}, сортировка по score:")
        chunk = []
        for i, (tk, row) in enumerate(df_sorted.iterrows(), 1):
            chunk.append(fmt_row(i, tk, row))
            if len(chunk) == 30:
                await update.message.reply_text("\n".join(chunk)); chunk = []
        if chunk:
            await update.message.reply_text("\n".join(chunk))

# -------------------- ЖЁСТКО ЗАШИТЫЕ ВСЕЛЕННЫЕ --------------------
# SP250: TOP-250 S&P 500 по весу (порядок зашит)
SP250 = """
NVDA MSFT AAPL AMZN META AVGO GOOGL GOOG BRK.B TSLA
JPM WMT V LLY ORCL MA NFLX XOM JNJ COST
HD BAC ABBV PG PLTR CVX GE KO TMUS UNH
CSCO AMD WFC PM MS CRM ABT IBM AXP GS
MCD LIN DIS T RTX MRK PEP CAT UBER NOW
VZ INTU TMO BKNG C ANET SCHW BA QCOM BLK
TXN SPGI GEV ISRG BSX TJX ACN SYK AMGN LOW
NEE PGR ADBE COF DHR GILD PFE MU APH HON
ETN BX UNP PANW DE LRCX AMAT CMCSA
KKR ADP ADI COP MDT KLAC WELL MO CB
SNPS NKE INTC DASH LMT PLD CRWD MMC VRTX
SO ICE SBUX RCL CEG CME PH CDNS
BMY HCA DUK CVS TT AMT SHW WM
MCO ORLY GD MCK DELL NOC CTAS MMM NEM
AON CI PNC MSI COIN MDLZ AJG ECL
ITW APO ABNB USB EMR EQIX BK FI RSG
MAR TDG HWM UPS AZO WMB JCI ELV
ADSK ZTS CL EOG FCX PYPL HLT APD VST
URI NSC TRV MNST WDAY TEL CSX TFC
REGN GLW KMI SPG FTNT AFL AEP FAST
NXPI AXON ROP COR PWR DLR CMG CMI
GM ALL BDX MET NDAQ MPC SRE CARR SLB
O PSX FDX DHI PCAR PSA IDXX LHX ROST
D CBRE AMP CTVA PAYX GWW VLO EW CPRT
OKE F GRMN XYZ OXY DDOG BKR AIG KR
TTWO EXC AME MSCI KMB CCL XEL EBAY EA
TGT CCI FANG MPWR PEG YUM KDP DAL
SYY RMD KVUE STX ETR VMC ROK PRU
""".split()

# ETFALL (обычные; с MSTR)
ETFALL = [
    "SPY","VOO","IVV","RSP","QQQ","DIA","IWM","IWB","IWR","IJR",
    "XLK","XLY","XLF","XLE","XLI","XLP","XLV","XLU","XLB","XLC",
    "TLT","IEF","HYG","LQD","AGG","BND",
    "GLD","IAU","SLV","GDX",
    "SMH","SOXX","IBB","XBI","ITB","XHB","IYT","XOP","XME","XRT",
    "BITO","LIT","MAGS","ARKK","KRE","KBE","MSTR",
]

# ETFX (плечевые/инверсные + кастом)
ETFX = [
    "TQQQ","SQQQ","SPXL","SPXS","UPRO","SPXU","UDOW","SDOW",
    "SOXL","SOXS","TECL","TECS","FNGU","FNGD",
    "LABU","LABD","TNA","TZA",
    "UCO","SCO","BOIL","KOLD",
    "NUGT","DUST",
    "UVXY","TSLL","WEBL","UPRO","USD","BITX",
    "GGLL","AAPU","FBL","MSFU","AMZU","NVDL","CONL",
]

# -------------------- КОМАНДЫ --------------------
async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tz = ZoneInfo(BOT_TZ)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    await update.message.reply_text(f"Я на связи ✅\n{now} {BOT_TZ}")

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    me = await context.bot.get_me()
    await update.message.reply_text(
        f"Bot: @{me.username} (id={me.id})\n"
        f"MODE={BENCH_MODE}  DEFAULT_BENCH={BENCH_DEFAULT}\n"
        f"TZ={BOT_TZ}"
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
    args = [a.lower() for a in context.args] if context.args else []
    name = "etfall"; tickers = ETFALL
    if args and args[0] in ("etfall","etfx"):
        name = args[0]
    if name == "etfx":
        tickers = ETFX
    await universe_rank(update, context, name.upper(), tickers, top_n=None)

async def benchmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BENCH_MODE
    if context.args:
        mode = context.args[0].lower().strip()
        if mode in ("global","smart"):
            BENCH_MODE = mode
            await update.message.reply_text(f"Режим бенчмарка: {BENCH_MODE}")
            return
    await update.message.reply_text("Формат: /benchmode <global|smart>")

async def setbenchmark_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BENCH_DEFAULT
    if not context.args:
        await update.message.reply_text("Формат: /setbenchmark <тикер>")
        return
    BENCH_DEFAULT = context.args[0].upper()
    await update.message.reply_text(f"Глобальный бенч: {BENCH_DEFAULT}")

async def benchcandidates_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Кандидаты SMART:\n" + " ".join(BENCH_CANDIDATES))

async def getbench_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Текущие настройки:\nMODE={BENCH_MODE}\nDEFAULT={BENCH_DEFAULT}")

# -------------------- ИНИЦИАЛИЗАЦИЯ --------------------
async def _post_init(app: Application):
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        await asyncio.sleep(3)
        me = await app.bot.get_me()
        print(f"[init] webhook cleared. Bot: @{me.username} (id={me.id})")
    except Exception as e:
        print(f"[init] delete_webhook error: {e}")

def build_app() -> Application:
    request = HTTPXRequest(connect_timeout=30.0, read_timeout=60.0, write_timeout=30.0, pool_timeout=30.0)
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request)
        .post_init(_post_init)
        .build()
    )
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
    app.add_handler(CommandHandler("getbench", getbench_cmd))
    return app

# -------------------- MAIN --------------------
def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_TOKEN env var.")
    while True:
        app = build_app()
        try:
            app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
            break
        except Conflict as e:
            print(f"[conflict] {e}. retry in 5s")
            try:
                asyncio.run(app.bot.delete_webhook(drop_pending_updates=True))
            except Exception as ee:
                print(f"[conflict] delete_webhook err: {ee}")
            time.sleep(5); continue
        except Exception as e:
            print(f"[fatal] {e}. restart in 5s")
            time.sleep(5)

if __name__ == "__main__":
    main()
