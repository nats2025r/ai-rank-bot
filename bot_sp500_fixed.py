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
# ENV
# ──────────────────────────────────────────────────────────────────────────────
BOT_TOKEN     = os.getenv("TELEGRAM_TOKEN", "").strip()
AI_CSV_URL    = os.getenv("AI_CSV_URL", "none").strip()    # "none" или URL CSV: ticker,ai_score (0..1)
W_AI          = float(os.getenv("W_AI", "0.0"))            # 0..1 — вес AI в композите
BENCH_DEFAULT = os.getenv("BENCH_DEFAULT", "SPY").strip().upper()
BENCH_MODE    = os.getenv("BENCH_MODE", "global").strip().lower()  # "global" | "smart"
BOT_TZ        = os.getenv("BOT_TZ", "Europe/Berlin")

BENCH_CANDIDATES = [
    "SPY","QQQ","IWM","DIA",
    "XLK","XLV","XLY","XLP","XLE","XLF","XLI","XLB","XLRE","XLU"
]

# ──────────────────────────────────────────────────────────────────────────────
# === UNIVERSES ===
# ──────────────────────────────────────────────────────────────────────────────

# ====== SP250: фиксированный топ-250 (снимок по весам S&P 500, ~2025) ======
SP250 = [
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA",
    "JPM","WMT","V","LLY","ORCL","MA","NFLX","XOM","JNJ","COST",
    "HD","BAC","PLTR","ABBV","PG","CVX","KO","GE","TMUS","UNH",
    "CSCO","AMD","WFC","PM","CRM","MS","ABT","AXP","IBM","GS",
    "LIN","MCD","DIS","RTX","MRK","T","PEP","CAT","UBER","NOW",
    "VZ","INTU","TMO","TXN","BKNG","C","BA","BLK","SCHW","QCOM",
    "ANET","ISRG","SPGI","ACN","BSX","AMGN","TJX","ADBE","SYK","NEE",
    "DHR","LOW","PGR","PFE","GILD","HON","ETN","BX","MU","APH",
    "UNP","DE","AMAT","PANW","LRCX","CMCSA","ADI","COP","ADP","MDT",
    "KLAC","NKE","MO","WELL","SNPS","CB","INTC","LMT","CRWD","PLD",
    "SO","MMC","ICE","VRTX","SBUX","RCL","CEG","PH","CME","BMY",
    "CDNS","AMT","DUK","HCA","CVS","TT","MCO","SHW","WM","ORLY",
    "GD","MCK","CTAS","NOC","DELL","NEM","PNC","CI","ABNB","MDLZ",
    "AON","TDG","MSI","ECL","COIN","APO","AJG","ITW","EQIX","USB",
    "FI","BK","EMR","UPS","RSG","MAR","ELV","WMB","AZO","HWM",
    "JCI","ZTS","EOG","CL","ADSK","PYPL","APD","HLT","VST","FCX",
    "NSC","WDAY","REGN","URI","TRV","TEL","MNST","CSX","TFC","FTNT",
    "KMI","AEP","NXPI","SPG","AXON","GLW","DLR","AFL","FAST","ROP",
    "CMG","PWR","GM","CARR","BDX","SLB","CMI","MPC","FDX","NDAQ",
    "MET","PSX","SRE","O","ALL","PCAR","LHX","IDXX","PSA","D",
    "DHI","CTVA","PAYX","SQ","AMP","GWW","CBRE","ROST","WELL","PXD",
    "ODFL","KR","YUM","HPQ","TROW","SWKS","AEP","EIX","EA","EBAY",
    "F","FANG","FIS","FITB","FTV","GEHC","HES","HIG","HOLX","HPE",
    "HSY","IBM","IEX","IFF","ILMN","INCY","IR","IRM","IT","ITW",
    "JBHT","JBL","JKHY","JNPR","KDP","KEYS","KHC","KIM","KMB","KMX",
    "KHC","KMB","KMI","L","LEN","LH","LHX","LNT","LOW","LRCX",
    "LULU","LVS","LW","LYB","LYV","MAA","MAS","MCHP","META","MGM",
    "MHK","MKC","MKTX","MMM","MNST","MOS","MPWR","MRK","MRO","MSCI",
    "MTB","MTD","MU","NDAQ","NEE","NEM","NFLX","NI","NOC","NOW",
    "NRG","NSC","NTAP","NTRS","NUE","NVR","NWS","NWSA","NXPI","O",
    "OKE","OMC","ON","ORCL","ORLY","OXY","PAYC","PAYX","PCAR","PEG",
    "PEP","PFE","PFG","PG","PGR","PH","PKG","PLD","PLTR","PM",
    "PNR","PPG","PPL","PRU","PSA","PTC","PWR","PYPL","QCOM","QRVO",
    "RCL","RMD","ROK","ROL","ROP","ROST","RSG","RTX","SBAC","SBUX",
    "SCHW","SHW","SJM","SLB","SNA","SNPS","SO","SPG","SPGI","SRE",
    "STE","STT","STZ","SWK","SWKS","SYF","SYK","SYY","TAP","TDG",
    "TDY","TEL","TER","TFX","TGT","TJX","TMO","TMUS","TRV","TSLA",
    "TSN","TT","TTWO","TXN","TXT","UDR","UHS","ULTA","UNH","UNP",
    "UPS","URI","USB","V","VFC","VLO","VMC","VRSK","VRSN","VRTX",
    "VTR","VTRS","VZ","WAB","WAT","WBA","WDC","WM","WMB","WMT",
    "WST","WY","XEL","XOM","XYL","ZBH","ZBRA","ZION","ZTS"
]

# ====== ETFALL: 50 обычных, без плеча/инверса ======
ETFALL = [
    "SPY","VOO","IVV","VTI","QQQ","IWM","DIA","RSP",
    "EFA","IEFA","EEM","IEMG",
    "AGG","BND","LQD","HYG","TLT","IEF","SHY","TIP",
    "GLD","IAU","SLV","GDX",
    "XLK","XLV","XLY","XLP","XLF","XLE","XLI","XLB","XLRE","XLU",
    "SMH","SOXX","XBI","IBB","XOP","XME","KRE","XHB","IYR","VNQ",
    "ARKK","BITO","LIT","MAGS","VUG","VTV"
]

# ====== ETFX: 27 плечевых/инверсных ======
ETFX = [
    # Индексы
    "UPRO","SPXL","SSO","SDS","SPXS","UDOW","SDOW",
    "TQQQ","SQQQ","QLD","QID",
    "TNA","TZA",
    "WEBL","WEBS",
    # Сектора/темы
    "SOXL","SOXS","TECL","TECS","FNGU","FNGD",
    "LABU","LABD","NUGT","DUST","USD",
    # Нефть/крипто/прочее
    "UCO","SCO","BITX","TSLL"
]

# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────
def to_yahoo(t: str) -> str:
    t = t.strip().upper()
    t = t.replace("/", "-")
    if "." in t:
        t = t.replace(".", "-")
    return t

def from_yahoo(t: str) -> str:
    return t.replace("-", ".")

def load_ai_map(url: str) -> dict:
    if not url or url.lower() == "none":
        return {}
    try:
        df = pd.read_csv(url)
        df["ticker"] = df["ticker"].astype(str).str.upper()
        return {k: float(v) for k, v in zip(df["ticker"], df["ai_score"])}
    except Exception:
        return {}

AI_MAP = load_ai_map(AI_CSV_URL)

# Устойчивая загрузка котировок чанками с ретраями
async def fetch_history(tickers, period="400d", interval="1d") -> pd.DataFrame:
    tickers = [to_yahoo(t) for t in tickers]
    CHUNK = 35
    chunks = [tickers[i:i+CHUNK] for i in range(0, len(tickers), CHUNK)]

    close_parts, vol_parts = [], []

    for ch in chunks:
        got = False
        for attempt in range(3):
            try:
                df = yf.download(
                    tickers=" ".join(ch),
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
                if df is None or df.empty:
                    raise ValueError("empty batch")

                closes, vols = {}, {}
                for yy in ch:
                    try:
                        sub = df[yy] if isinstance(df.columns, pd.MultiIndex) else df
                        if "Close" in sub and "Volume" in sub and not sub.empty:
                            sym = from_yahoo(yy)
                            closes[sym] = sub["Close"].rename(sym)
                            vols[sym]   = sub["Volume"].rename(sym)
                    except Exception:
                        pass
                if closes: close_parts.append(pd.concat(closes.values(), axis=1))
                if vols:   vol_parts.append(pd.concat(vols.values(), axis=1))
                got = True
                break
            except Exception as e:
                await asyncio.sleep(1.5*(attempt+1))
                if attempt == 2:
                    print(f"[warn] batch failed, skipped: {ch[:3]}... ({len(ch)}), err={e}")
        if not got:
            continue

    close_df = (pd.concat(close_parts, axis=1).loc[:, lambda d: ~d.columns.duplicated()]
                if close_parts else pd.DataFrame())
    vol_df   = (pd.concat(vol_parts, axis=1).loc[:, lambda d: ~d.columns.duplicated()]
                if vol_parts else pd.DataFrame())
    return close_df, vol_df

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

# выбор бенча: global или smart (по корреляции)
def pick_benchmark(returns_tbl: pd.DataFrame, ticker: str, mode: str) -> str:
    if mode != "smart":
        return BENCH_DEFAULT
    if ticker not in returns_tbl.columns:
        return BENCH_DEFAULT
    y = returns_tbl[ticker].dropna()
    if len(y) < 40:
        return BENCH_DEFAULT
    best_t, best_r = BENCH_DEFAULT, -2.0
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

def score_universe(close: pd.DataFrame, vol: pd.DataFrame, universe: list,
                   bench_mode: str) -> pd.DataFrame:
    u = [t for t in universe if t in close.columns]
    if not u:
        return pd.DataFrame()

    # добираем бэнчи (если не подгрузились)
    bench_set = set([BENCH_DEFAULT] + BENCH_CANDIDATES)
    need_bench = [b for b in bench_set if b not in close.columns]
    if need_bench:
        add_close, add_vol = asyncio.run(fetch_history(need_bench))
        close = pd.concat([close, add_close], axis=1)

    ret_daily = close.pct_change().dropna()
    rows = []
    for t in u:
        px = close[t].dropna()
        if len(px) < 210:   # минимум данных
            continue

        b = pick_benchmark(ret_daily, t, bench_mode)
        bx = close[b].dropna() if b in close.columns else None
        if bx is None or len(bx) < 210:
            b = BENCH_DEFAULT
            bx = close[b].dropna()

        try:
            rs21 = (px.iloc[-1]/px.iloc[-22]-1) - (bx.iloc[-1]/bx.iloc[-22]-1)
            rs63 = (px.iloc[-1]/px.iloc[-64]-1) - (bx.iloc[-1]/bx.iloc[-64]-1)
        except Exception:
            continue

        ema20  = ema(px, 20).iloc[-1]
        ema50  = ema(px, 50).iloc[-1]
        ema200 = ema(px, 200).iloc[-1]
        trend_ok = 1.0 if (px.iloc[-1] > ema20 > ema50 > ema200) else 0.0

        window = px.tail(252)
        if len(window) < 30: 
            continue
        hi = window.max()
        prox_hi = float(px.iloc[-1]/hi)

        vol_t = float(window.pct_change().std())
        vol_b = float(bx.tail(len(window)).pct_change().std())
        vol_rel = 1.0 - min(1.0, max(0.0, (vol_t / (vol_b+1e-9))))

        base = (0.30*trend_ok +
                0.30*max(0.0, min(1.0, (prox_hi-0.85)/0.15)) +
                0.20*max(0.0, min(1.0, rs21/0.10)) +
                0.20*vol_rel)

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
    df = df[(df["RS21"] > 0) & (df["RS63"] > 0)]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df

def fmt_row(i, r):
    return (f"{i}) {r['ticker']} | RS21={r['RS21']:+.2%} | RS63={r['RS63']:+.2%} "
            f"| score={r['score']:.3f} | bench={r['bench']} | "
            f"Confirm={r['ConfirmScore']:.3f}")

async def run_rank(universe: list, topn: int = 10, bench_mode: str = None) -> str:
    bench_mode = bench_mode or BENCH_MODE
    name = "SP250" if universe is SP250 else ("ETFALL" if universe is ETFALL else ("ETFX" if universe is ETFX else "UNIVERSE"))
    close, vol = await fetch_history(universe + [BENCH_DEFAULT] + BENCH_CANDIDATES)
    if close.empty:
        return "Не удалось загрузить котировки. Попробуй ещё раз."
    df = score_universe(close, vol, universe, bench_mode)
    if df.empty:
        return f"{name}: подходящих бумаг не найдено (фильтр RS21/RS63>0)."
    df_top = df.head(topn)
    lines = [f"Топ {topn} — {name} (режим бенча: {bench_mode})"]
    for i, r in enumerate(df_top.itertuples(index=False), start=1):
        lines.append(fmt_row(i, r._asdict()))
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# Команды
# ──────────────────────────────────────────────────────────────────────────────
async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Я на связи ✅")

async def sp5005_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Считаю SP250… это ~20–30 сек на первом запуске.")
    await update.message.reply_text(await run_rank(SP250, topn=5))

async def sp50010_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Считаю SP250… это ~20–30 сек на первом запуске.")
    await update.message.reply_text(await run_rank(SP250, topn=10))

async def etfall_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Считаю ETFALL…")
    await update.message.reply_text(await run_rank(ETFALL, topn=10))

async def etfx_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Считаю ETFX…")
    await update.message.reply_text(await run_rank(ETFX, topn=10))

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
            await update.message.reply_text("\n".join(lines)); lines=[]
    if lines:
        await update.message.reply_text("\n".join(lines))

# Настройки/сервис
async def benchmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BENCH_MODE
    if not context.args:
        await update.message.reply_text(f"Режим бенчмарка: {BENCH_MODE}"); return
    m = context.args[0].strip().lower()
    if m not in ("global","smart"):
        await update.message.reply_text("Формат: /benchmode <global|smart>"); return
    BENCH_MODE = m
    await update.message.reply_text(f"Режим бенча установлен: {BENCH_MODE}")

async def setbenchmark_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BENCH_DEFAULT
    if not context.args:
        await update.message.reply_text("Формат: /setbenchmark <TICKER>"); return
    BENCH_DEFAULT = context.args[0].upper()
    await update.message.reply_text(f"Глобальный бенч: {BENCH_DEFAULT}")

async def benchcandidates_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Кандидаты SMART:\n" + " ".join(BENCH_CANDIDATES))

async def getbench_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Текущие настройки:\nMODE={BENCH_MODE}\nDEFAULT={BENCH_DEFAULT}")

async def setaisource_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_CSV_URL, AI_MAP
    if not context.args:
        await update.message.reply_text("Формат: /setaisource <csv|none> (ticker,ai_score)"); return
    AI_CSV_URL = context.args[0].strip()
    AI_MAP = load_ai_map(AI_CSV_URL)
    await update.message.reply_text(f"Источник AI: {AI_CSV_URL}")

async def setw_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global W_AI
    if not context.args:
        await update.message.reply_text(f"Текущий W_AI={W_AI}. Формат: /setw_ai <0..1>"); return
    try:
        W_AI = max(0.0, min(1.0, float(context.args[0])))
        await update.message.reply_text(f"W_AI установлен: {W_AI}")
    except Exception:
        await update.message.reply_text("Ошибка: нужен числовой 0..1")

# Диагностика сети
async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        df, _ = await fetch_history(["SPY"], period="120d", interval="1d")
        if df.empty or "SPY" not in df.columns:
            await update.message.reply_text("❌ SPY не загрузился. Похоже, сеть/доступ к Yahoo.")
        else:
            await update.message.reply_text("✅ Данные тянутся: SPY ок.")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка сети: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Авто-сброс вебхука при старте (на всякий)
# ──────────────────────────────────────────────────────────────────────────────
async def _post_init(app):
    try:
        info = await app.bot.get_webhook_info()
        if info.url:
            await app.bot.delete_webhook(drop_pending_updates=True)
            print(f"[boot] Webhook cleared (was: {info.url})")
        me = await app.bot.get_me()
        print(f"[boot] Bot: @{me.username} (id={me.id})")
    except Exception as e:
        print(f"[boot] webhook check error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_TOKEN env var.")

    app = Application.builder().token(BOT_TOKEN).post_init(_post_init).build()

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
    app.add_handler(CommandHandler("diag",          diag_cmd))

    print("Bot started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
