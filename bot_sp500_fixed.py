# bot_sp500_fixed.py
# Telegram-бот: AI-моментум + ConfirmScore для SP250/ETFALL/ETFX
# Зависимости: python-telegram-bot==20.7, yfinance==0.2.43, pandas==2.2.2, numpy==1.26.4, lxml, bs4, requests

import os
import math
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# telegram-bot
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest  # <- добавили

# -------------------- ENV --------------------
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
AI_CSV_URL = os.getenv("AI_CSV_URL", "none").strip()
W_AI = float(os.getenv("W_AI", "0.0"))
BENCH_DEFAULT = os.getenv("BENCH_DEFAULT", "SPY").strip().upper()
BENCH_MODE = os.getenv("BENCH_MODE", "global").strip().lower()  # "global" | "smart"
BOT_TZ = os.getenv("BOT_TZ", "Europe/Berlin")

BENCH_CANDIDATES = [
    "SPY","QQQ","IWM","DIA",
    "XLK","XLY","XLF","XLP","XLV","XLU","XLB","XLC","XLE","XLI",
]

# -------------------- Надёжная загрузка цен --------------------
YF_OPTS = dict(period="180d", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=False)

# Единая requests-сессия с User-Agent — уменьшает шанс пустых ответов Yahoo
YF_SESSION = requests.Session()
YF_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
})

# при желании можно подменять «капризные» тикеры на аналоги
ALIAS: Dict[str, str] = {
    # "BITX": "BITB",
    # "USD": "USD",
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
            return s.rename(ticker)
        return None
    else:
        if 'Close' not in df.columns:
            return None
        return df['Close'].rename(ticker)

def _fetch_one_close(ticker: str) -> Optional[pd.Series]:
    """Загрузка цены с fallback и общей сессией."""
    t = ALIAS.get(ticker, ticker)
    # 1-я попытка
    try:
        df = yf.download(t, session=YF_SESSION, **YF_OPTS)
        s = _normalize_yf_df(df, t)
        if s is not None and not s.dropna().empty:
            return s.rename(ticker)
    except Exception as e:
        print(f"[yfinance] fail {ticker} (1st): {e}")
    # 2-я попытка: длиннее период
    try:
        alt = dict(YF_OPTS)
        alt["period"] = "365d"
        df = yf.download(t, session=YF_SESSION, **alt)
        s = _normalize_yf_df(df, t)
        if s is not None and not s.dropna().empty:
            return s.rename(ticker)
    except Exception as e:
        print(f"[yfinance] fail {ticker} (2nd): {e}")
    return None

async def load_close_matrix(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    data: List[pd.Series] = []
    bad: List[str] = []
    for tk in tickers:
        s = _fetch_one_close(tk)
        if s is None or s.dropna().empty:
            bad.append(tk)
        else:
            data.append(s)
        await asyncio.sleep(0.15)  # мягкий rate-limit
    if not data:
        raise RuntimeError(f"Не удалось загрузить ни один тикер из: {bad}")
    closes = pd.concat(data, axis=1)
    closes = closes.dropna(how="all").dropna(axis=1, how="all")
    return closes, bad

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
    """Сколько МА (50/100/150/200) цена сейчас выше."""
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
    return score / 4.0  # 0..1

def rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average")

def pick_benchmark_for(ticker: str, closes: pd.DataFrame) -> str:
    """SMART: выбираем бенчмарк с макс. корреляцией за 180d из BENCH_CANDIDATES."""
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

def load_ai_scores() -> Dict[str, float]:
    if not AI_CSV_URL or AI_CSV_URL.lower() == "none":
        return {}
    try:
        text = requests.get(AI_CSV_URL, timeout=10).text
        out: Dict[str, float] = {}
        for line in text.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                tk = parts[0].upper()
                try:
                    out[tk] = float(parts[1])
                except:
                    pass
        return out
    except Exception as e:
        print(f"[AI CSV] fail: {e}")
        return {}

def composite_score(rs21_rel: pd.Series, rs63_rel: pd.Series, conf: pd.Series, ai: Dict[str,float]) -> pd.Series:
    r1 = rank_pct(rs21_rel.fillna(-1))
    r2 = rank_pct(rs63_rel.fillna(-1))
    rc = conf.fillna(0)
    base = 0.45*r1 + 0.35*r2 + 0.20*rc
    if not ai or W_AI <= 0:
        return base
    ai_series = pd.Series({k:v for k,v in ai.items() if k in base.index})
    if not ai_series.empty:
        ai_norm = rank_pct(ai_series)
        base.loc[ai_norm.index] = (1.0 - W_AI)*base.loc[ai_norm.index] + W_AI*ai_norm
    return base

def fmt_row(i, tk, row):
    return f"{i}) {tk} | score={row['score']:.4f} | RS21={row['rs21_rel']:+.2%} | RS63={row['rs63_rel']:+.2%} | conf={row['conf']:.2f}"

async def universe_rank(update: Update, context: ContextTypes.DEFAULT_TYPE, universe_name: str, tickers: List[str], top_n: Optional[int]):
    await update.message.reply_text(f"Считаю {universe_name}... это ~20–30 сек на первом запуске.")
    bench_need = {BENCH_DEFAULT}
    if BENCH_MODE == "smart":
        bench_need.update(BENCH_CANDIDATES)
    use_tickers = sorted(set([t.upper() for t in tickers] + list(bench_need)))

    # ключевой try/except — вместо падения даём понятный ответ
    try:
        closes, bad = await load_close_matrix(use_tickers)
    except Exception as e:
        await update.message.reply_text(f"Не получилось загрузить котировки (Yahoo). Попробуй ещё раз чуть позже.\nДетали: {e}")
        return

    missing = [t for t in tickers if t not in closes.columns]
    bad_all = sorted(set(bad + missing))
    if bad_all:
        await update.message.reply_text("⚠️ Пропущены тикеры без истории: " + ", ".join(bad_all))

    uni = [t for t in tickers if t in closes.columns]
    if not uni:
        await update.message.reply_text("Нечего ранжировать: нет данных по вселенной.")
        return

    ai = load_ai_scores() if W_AI > 0 else {}
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
        rows.append({"ticker": tk, "bench": bench, "rs21_rel": rs21_rel, "rs63_rel": rs63_rel, "conf": conf})

    df = pd.DataFrame(rows).set_index("ticker")
    if df.empty:
        await update.message.reply_text("Нет валидных данных после загрузки котировок.")
        return

    df["score"] = composite_score(df["rs21_rel"], df["rs63_rel"], df["conf"], ai)

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

# -------------------- ВСЕЛЕННЫЕ --------------------
SP250 = """
NVDA MSFT AAPL AMZN GOOGL GOOG BRK.B META TSLA AVGO LLY JPM UNH V MA XOM JNJ PG ORCL CVX MRK COST ABBV KO PEP BAC WMT NFLX ADBE CRM CSCO TMO ACN MCD AMD WFC DHR LIN ABT TXN IBM PM CAT COP NKE AMAT MDT HON UPS QCOM LMT MS BKNG LOW GE BMY SPGI T AMGN RTX C GS NOW SBUX GILD DE ADP INTU MU CHTR ELV MMC MDLZ PLD BLK ISRG ZTS ADI TGT REGN CB PGR SYK SO EOG HCA PANW KLAC BSX TJX CME MCO USB EQIX PXD AON FI AXP OXY SHW ITW NOC CDW LRCX PEP^ QQQ^
""".split()

ETFALL = [
    "SPY","VOO","IVV","RSP","QQQ","DIA","IWM","IWB","IWR","IJR",
    "XLK","XLY","XLF","XLE","XLI","XLP","XLV","XLU","XLB","XLC",
    "TLT","IEF","HYG","LQD","AGG","BND",
    "GLD","IAU","SLV","GDX",
    "SMH","SOXX","IBB","XBI","ITB","XHB","IYT","XOP","XME","XRT",
    "BITO","LIT","MAGS","ARKK","KRE","KBE",
]

ETFX = [
    "TQQQ","SQQQ","SPXL","SPXS","UPRO","SPXU","UDOW","SDOW",
    "SOXL","SOXS","TECL","TECS","FNGU","FNGD",
    "LABU","LABD","TNA","TZA",
    "UCO","SCO","BOIL","KOLD",
    "NUGT","DUST",
    "UVXY","TSLL","WEBL","UPRO","USD","BITX",
]

# -------------------- КОМАНДЫ --------------------
async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tz = ZoneInfo(BOT_TZ)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    await update.message.reply_text(f"Я на связи ✅\n{now} {BOT_TZ}")

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

async def setaisource_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_CSV_URL
    if not context.args:
        await update.message.reply_text("Формат: /setaisource <csv|none>\nПример: /setaisource https://.../ai_scores.csv")
        return
    AI_CSV_URL = context.args[0]
    await update.message.reply_text(f"Источник AI: {AI_CSV_URL}")

async def setw_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global W_AI
    if not context.args:
        await update.message.reply_text("Формат: /setw_ai <0..1>")
        return
    try:
        val = float(context.args[0])
        if 0.0 <= val <= 1.0:
            W_AI = val; await update.message.reply_text(f"W_AI = {W_AI}")
        else:
            await update.message.reply_text("Допустимо 0..1")
    except:
        await update.message.reply_text("Не понял число. Пример: /setw_ai 0.2")

# -------------------- ИНИЦИАЛИЗАЦИЯ: снять webhook --------------------
async def _post_init(app: Application):
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        me = await app.bot.get_me()
        print(f"[init] webhook cleared. Bot: @{me.username} (id={me.id})")
    except Exception as e:
        print(f"[init] delete_webhook error: {e}")

# -------------------- MAIN --------------------
def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_TOKEN env var.")

    # увеличиваем таймауты клиента Telegram (решает TimedOut)
    request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=60.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request)
        .post_init(_post_init)
        .build()
    )

    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("sp5005", sp5005_cmd))
    app.add_handler(CommandHandler("sp50010", sp50010_cmd))
    app.add_handler(CommandHandler("etfall", etfall_cmd))
    app.add_handler(CommandHandler("etfx", etfx_cmd))
    app.add_handler(CommandHandler("etf", etf_full_cmd))

    app.add_handler(CommandHandler("benchmode", benchmode_cmd))
    app.add_handler(CommandHandler("setbenchmark", setbenchmark_cmd))
    app.add_handler(CommandHandler("benchcandidates", benchcandidates_cmd))
    app.add_handler(CommandHandler("getbench", getbench_cmd))
    app.add_handler(CommandHandler("setaisource", setaisource_cmd))
    app.add_handler(CommandHandler("setw_ai", setw_ai_cmd))

    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

