# bot_sp500_fixed.py
# Telegram-бот: AI-моментум + ConfirmScore для SP250/ETFALL/ETFX
# Только рынок США. Устойчивый загрузчик котировок (мелкие батчи, ретраи, CSV+Stooq фолбэк).
# Исправлено: НЕТ group_by="ticker" (ломал мультизагрузку ETF), универсальный парсер MultiIndex.
# Зависимости: python-telegram-bot==20.7, yfinance==0.2.43, pandas==2.2.2, numpy==1.26.4,
#              requests>=2.31, lxml, beautifulsoup4, pandas_datareader==0.10.0

import os
import math
import time
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from pandas_datareader import data as pdr

# telegram-bot
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# -------------------- ENV --------------------
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
BENCH_DEFAULT = os.getenv("BENCH_DEFAULT", "SPY").strip().upper()
BENCH_MODE = os.getenv("BENCH_MODE", "smart").strip().lower()   # "global" | "smart"
BOT_TZ = os.getenv("BOT_TZ", "Europe/Berlin")

# -------------------- Параметры загрузки котировок --------------------
YF_SESSION = requests.Session()
YF_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Connection": "keep-alive",
})
YF_OPTS = dict(
    period="180d",
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=False,           # меньше 429
    session=YF_SESSION       # свой UA/keep-alive
)
BATCH_SIZE = 18
BATCH_PAUSE = 1.0
RETRIES = 3
FALLBACK_GAP = 0.8
HTTP_TIMEOUT = 12

# -------------------- США-санитайзер --------------------
ALIAS = {
    "BRK.B": "BRK-B", "BRK.A": "BRK-A",
    "BF.B": "BF-B", "BF.A": "BF-A",
    "HEI.A": "HEI-A", "HEI.B": "HEI-B",
    "GOOGLE": "GOOGL",
}
NON_US_SUFFIXES = (
    ".DE",".F",".SW",".MI",".PA",".BR",".AS",".L",".VX",".TO",".V",
    ".HK",".SS",".SZ",".KS",".KQ",".TW",".TWO",".SI",".BK",".IS",
    ".SA",".MX",".VI",".HE",".ST",".OL",".MC",".TA"
)
US_DOT_WHITELIST = {"BRK.A","BRK.B","BF.A","BF.B","HEI.A","HEI.B"}

def sanitize_us_tickers(tickers: list[str]) -> list[str]:
    out = []
    for t in tickers:
        t0 = t.upper().strip()
        if any(t0.endswith(suf) for suf in NON_US_SUFFIXES):
            continue
        if "." in t0 and t0 not in US_DOT_WHITELIST:
            continue
        t1 = ALIAS.get(t0, t0.replace(".", "-"))
        out.append(t1)
    # уникальность с сохранением порядка
    seen = set(); uniq = []
    for x in out:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

# -------------------- анти-дубли --------------------
RECENT_MSG_IDS = deque(maxlen=500)
def seen_message(update: Update) -> bool:
    mid = getattr(update.message, "message_id", None)
    if mid is None: return False
    if mid in RECENT_MSG_IDS: return True
    RECENT_MSG_IDS.append(mid); return False

# -------------------- котировки --------------------
def _parse_close_from_multi(df: pd.DataFrame, orig_tickers: list[str]) -> pd.DataFrame:
    """
    Универсальный разбор ответа yf.download для списка тикеров.
    Поддерживает ('Close','SPY') и ('SPY','Close').
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        if 'Close' in lvl0 or 'Adj Close' in lvl0:
            use = 'Adj Close' if 'Adj Close' in lvl0 else 'Close'
            close = df.xs(use, axis=1, level=0)
        elif 'Close' in lvl1 or 'Adj Close' in lvl1:
            use = 'Adj Close' if 'Adj Close' in lvl1 else 'Close'
            close = df.xs(use, axis=1, level=1)
        else:
            return pd.DataFrame()
        cols = [c for c in orig_tickers if c in close.columns]
        return close[cols] if cols else close

    # одиночный тикер
    col = 'Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else None)
    if not col:
        return pd.DataFrame()
    out = df[[col]].copy()
    if len(orig_tickers) == 1:
        out.columns = [orig_tickers[0]]
    return out

def _fallback_yahoo_csv(ticker: str) -> pd.Series | None:
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
    params = {
        "period1": int(time.time()) - 210*24*3600,
        "period2": int(time.time()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true"
    }
    try:
        r = YF_SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code != 200 or not r.text:
            return None
        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            return None
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            return None
        s = pd.Series(df[col].values, index=pd.to_datetime(df["Date"]), name=ticker)
        return s
    except Exception as e:
        print(f"[yahoo-csv] fail {ticker}: {e}")
        return None

def _fallback_stooq(ticker: str) -> pd.Series | None:
    try:
        s = pdr.get_data_stooq(f"{ticker.lower()}.us")
        if s is None or s.empty or "Close" not in s.columns:
            return None
        out = s["Close"].rename(ticker).sort_index()
        out.index = pd.to_datetime(out.index)
        return out
    except Exception as e:
        print(f"[stooq] fail {ticker}: {e}")
        return None

def _trim_incomplete_daily(df: pd.DataFrame) -> pd.DataFrame:
    try:
        ny = ZoneInfo("America/New_York")
        now = datetime.now(ny)
        if df.empty:
            return df
        last_date = df.index.max().date()
        if last_date == now.date() and now.hour < 23:
            return df.iloc[:-1]
        return df
    except Exception:
        return df

async def _fetch_one_close_strict(ticker: str) -> pd.Series | None:
    try:
        df = await asyncio.to_thread(yf.download, ticker, **YF_OPTS)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = df.columns.get_level_values(0)
                use = 'Adj Close' if 'Adj Close' in lvl0 else ('Close' if 'Close' in lvl0 else None)
                s = df[use] if use else None
                if s is not None:
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    return s.rename(ticker)
            else:
                col = 'Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else None)
                if col:
                    return df[col].rename(ticker)
    except Exception as e:
        print(f"[yfinance] single fail {ticker}: {e}")

    s = _fallback_yahoo_csv(ticker)
    if s is not None and not s.dropna().empty:
        return s
    return _fallback_stooq(ticker)

async def load_close_matrix(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    need = tickers[:]  # уже санитайзены выше
    got, frames = [], []

    for i in range(0, len(need), BATCH_SIZE):
        batch = need[i:i+BATCH_SIZE]
        last_err = None
        for attempt in range(RETRIES):
            try:
                df = await asyncio.to_thread(yf.download, batch, **YF_OPTS)
                close = _parse_close_from_multi(df, batch)
                if not close.empty:
                    frames.append(close)
                    got.extend(list(close.columns))
                break
            except Exception as e:
                last_err = e
                await asyncio.sleep(BATCH_PAUSE * (2 ** attempt))
        if last_err:
            print(f"[yfinance-multi] batch {batch[:3]}.. err: {last_err}")
        await asyncio.sleep(BATCH_PAUSE)

    closes = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if not closes.empty:
        closes = closes.loc[:, ~closes.columns.duplicated()]
        closes = _trim_incomplete_daily(closes)

    missing_orig = [t for t in tickers if t not in got]
    truly_bad = []
    for tk in missing_orig:
        s = await _fetch_one_close_strict(tk)
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
    if seen_message(update):
        return

    bench_need = {BENCH_DEFAULT}
    if BENCH_MODE == "smart":
        bench_need.update(BENCH_CANDIDATES)

    use_tickers_raw = [t.upper() for t in tickers] + list(bench_need)
    use_tickers = sanitize_us_tickers(sorted(set(use_tickers_raw)))

    closes, bad = await load_close_matrix(use_tickers)

    miss_for_universe = [t for t in tickers if t not in closes.columns]
    bad_all = sorted(set(bad + miss_for_universe))
    if bad_all:
        await update.message.reply_text("⚠️ Пропущены тикеры без данных: " + ", ".join(bad_all))

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

        # выравниваем последнюю дату
        last_day = pr.index.max()
        if bench in closes.columns:
            bpr = closes[bench].dropna()
            last_day = min(last_day, bpr.index.max())
            pr_use = pr.loc[:last_day]
            bpr_use = bpr.loc[:last_day]
        else:
            pr_use = pr
            bpr_use = None

        rs21 = pct_change(pr_use, 21)
        rs63 = pct_change(pr_use, 63)
        if bpr_use is not None:
            brs21 = pct_change(bpr_use, 21)
            brs63 = pct_change(bpr_use, 63)
            rs21_rel = (rs21 - brs21) if pd.notna(rs21) and pd.notna(brs21) else np.nan
            rs63_rel = (rs63 - brs63) if pd.notna(rs63) and pd.notna(brs63) else np.nan
        else:
            rs21_rel, rs63_rel = rs21, rs63

        conf = confirm_score(pr_use)

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

# SP250 — 250 тикеров из S&P 500 (только США), захардкожено
SP250 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","AVGO","LLY","JPM",
    "XOM","UNH","JNJ","V","MA","TSLA","ORCL","PG","COST","MRK",
    "ABBV","HD","PEP","KO","BAC","WMT","NFLX","ADBE","CRM","TMO",
    "CSCO","DHR","WFC","AMD","MCD","TXN","ABT","IBM","AXP","CAT",
    "GE","CVX","AMGN","NKE","COP","LMT","QCOM","PM","NOW","HON",
    "SBUX","RTX","TGT","SPGI","GS","ADP","MDT","LOW","BKNG","PGR",
    "C","GILD","ELV","DE","SYK","MU","PLD","ISRG","LRCX","INTU",
    "ZTS","PANW","MS","BLK","REGN","USB","MDLZ","BK","MRNA","VRTX",
    "CI","HUM","CVS","BDX","EW","BSX","ZBH","HCA","DGX","LH",
    "ILMN","IDXX","IQV","WST","MCK","CAH","DXCM","ALGN","TFC","PNC",
    "COF","DFS","FITB","KEY","RF","HBAN","CFG","MTB","NTRS","STT",
    "CME","ICE","CBOE","MKTX","MSCI","NDAQ","AIG","ALL","CB","TRV",
    "MET","PRU","PFG","AFL","CINF","HIG","MMC","WTW","BRO","BRK-B",
    "UPS","FDX","UNP","CSX","NSC","ODFL","JBHT","EXPD","CHRW","DAL",
    "AAL","UAL","LUV","CHTR","CMCSA","DIS","T","VZ","TMUS","PARA",
    "WBD","FOXA","EA","TTWO","PYPL","ADSK","ANSS","CDNS","SNPS","FTNT",
    "CRWD","WDAY","INTC","HPQ","DELL","HPE","AMAT","KLAC","TER","ON",
    "MCHP","MPWR","SWKS","QRVO","QCOM","FSLR","ENPH","APH","TEL","PH",
    "ETN","EMR","ITW","GD","NOC","LHX","HII","BA","MMM","ROP",
    "CARR","JCI","IR","FTV","DOV","ROK","AME","FAST","GWW","TDG",
    "WHR","NUE","STLD","CLF","MLM","VMC","FCX","ALB","APD","ECL",
    "SHW","PPG","DOW","DD","EMN","NEM","CF","MOS","FMC","BALL",
    "IP","PKG","WRK","AVY","IEX","NEE","SO","DUK","AEP","D",
    "EXC","SRE","ED","PEG","XEL","EIX","ES","AEE","NRG","WEC",
    "CMS","FE","CEG","MPC","VLO","PSX","KMI","WMB","OKE","LNG",
    "BKR","SLB","HAL","OXY","HES","EOG","PXD","FANG","CTRA","DVN",
    "MRO","APA","AMT","CCI","EQIX","DLR","SPG","O","PSA","WELL",
    "VTR","AVB","EQR","ESS","UDR","CPT","INVH","AMH","VICI","HST",
    "KIM","FRT","REG","BXP","WY","NVR","LEN","DHI","PHM","TOL",
    "CPRT","KMX","ORLY","AZO","ROST","TJX","DG","DLTR","HAS","HLT",
    "MAR","MGM","DRI","YUM","CMG","DPZ","NKE","VFC","PVH","RL",
    "TAP","STZ","BF-B","KHC","GIS","CPB","CLX","CL","KMB","CHD",
    "HSY","SJM","CAG","MKC","SYY","KR","ADM","BG","TSN","HRL",
    "WMT","COST","TGT"
]

# ETFALL — базовые ETF (США)
ETFALL = [
    "SPY","VOO","IVV","RSP","QQQ","DIA","IWM","IWB","IWR","IJR",
    "XLK","XLY","XLF","XLE","XLI","XLP","XLV","XLU","XLB","XLC",
    "TLT","IEF","HYG","LQD","AGG","BND",
    "GLD","IAU","SLV","GDX",
    "SMH","SOXX","IBB","XBI","ITB","XHB","IYT","XOP","XME","XRT",
    "BITO","LIT","MAGS","ARKK","KRE","KBE","MSTR"
]

# ETFX — плечевые/инверсные/одиночные (UPRO один раз)
ETFX = [
    "TQQQ","SQQQ","SPXL","SPXS","UPRO","SPXU","UDOW","SDOW",
    "SOXL","SOXS","TECL","TECS","FNGU","FNGD",
    "LABU","LABD","TNA","TZA",
    "UCO","SCO","BOIL","KOLD",
    "NUGT","DUST","UVXY",
    "TSLL","WEBL","USD","BITX",
    "GGLL","AAPU","FBL","MSFU","AMZU","NVDL","CONL"
]

# Кандидаты бенчмарков для SMART-режима
BENCH_CANDIDATES = [
    "SPY","QQQ","IWM","DIA",
    "XLK","XLY","XLF","XLE","XLI","XLP","XLV","XLU","XLB","XLC",
]

# -------------------- КОМАНДЫ --------------------
async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    tz = ZoneInfo(BOT_TZ)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    await update.message.reply_text(f"Я на связи ✅\n{now} {BOT_TZ}")

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
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
    if seen_message(update): return
    args = [a.lower() for a in context.args] if context.args else []
    name = "etfall"; tickers = ETFALL
    if args and args[0] in ("etfall","etfx"):
        name = args[0]
    if name == "etfx":
        tickers = ETFX
    await universe_rank(update, context, name.upper(), tickers, top_n=None)

async def benchmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    global BENCH_MODE
    if context.args:
        mode = context.args[0].lower().strip()
        if mode in ("global","smart"):
            BENCH_MODE = mode
            await update.message.reply_text(f"Режим бенчмарка: {BENCH_MODE}")
            return
    await update.message.reply_text("Формат: /benchmode <global|smart>")

async def setbenchmark_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    global BENCH_DEFAULT
    if not context.args:
        await update.message.reply_text("Формат: /setbenchmark <тикер>")
        return
    BENCH_DEFAULT = context.args[0].upper()
    await update.message.reply_text(f"Глобальный бенч: {BENCH_DEFAULT}")

async def benchcandidates_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
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

    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
