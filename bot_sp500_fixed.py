# bot_sp500_fixed.py
# AI-моментум + ConfirmScore для SP250/ETFALL/ETFX (только США)
# Загрузка цен: Stooq CSV (параллельно) -> Yahoo CSV (параллельно) -> yfinance single (опционально)
# Требования: python-telegram-bot==20.7, yfinance==0.2.43, pandas==2.2.2, numpy==1.26.4,
#             requests>=2.31, beautifulsoup4, lxml, pandas_datareader==0.10.0

import os, math, time, asyncio, logging
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ---------- лог ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("ai-rank-bot")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ---------- ENV ----------
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
BENCH_DEFAULT = os.getenv("BENCH_DEFAULT", "SPY").strip().upper()
BENCH_MODE = os.getenv("BENCH_MODE", "smart").strip().lower()  # "global" | "smart"
BOT_TZ = os.getenv("BOT_TZ", "Europe/Berlin")

# ---------- загрузка котировок ----------
YF_SESSION = requests.Session()
YF_SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "*/*", "Connection": "keep-alive"})
YF_OPTS = dict(period="180d", interval="1d", auto_adjust=True, progress=False,
               threads=False, session=YF_SESSION)  # ВАЖНО: без group_by!
CONCURRENCY = 12         # параллель CSV
HTTP_TIMEOUT = 6         # короткие сокет-таймауты
YF_SINGLE_TIMEOUT = 8    # таймаут одиночного yfinance
YF_PRIMARY = False       # использовать ли yfinance как резерв (True) или полностью выключить (False)

# ---------- США-санитайзер ----------
ALIAS = {"BRK.B":"BRK-B","BRK.A":"BRK-A","BF.B":"BF-B","BF.A":"BF-A","HEI.A":"HEI-A","HEI.B":"HEI-B","GOOGLE":"GOOGL"}
NON_US_SUFFIXES = (".DE",".F",".SW",".MI",".PA",".BR",".AS",".L",".VX",".TO",".V",".HK",".SS",".SZ",".KS",".KQ",".TW",".TWO",".SI",".BK",".IS",".SA",".MX",".VI",".HE",".ST",".OL",".MC",".TA")
US_DOT_WHITELIST = {"BRK.A","BRK.B","BF.A","BF.B","HEI.A","HEI.B"}

def sanitize_us_tickers(tickers: list[str]) -> list[str]:
    out=[]
    for t in tickers:
        t0=t.upper().strip()
        if any(t0.endswith(suf) for suf in NON_US_SUFFIXES): continue
        if "." in t0 and t0 not in US_DOT_WHITELIST: continue
        out.append(ALIAS.get(t0, t0.replace(".","-")))
    seen=set(); uniq=[]
    for x in out:
        if x not in seen: seen.add(x); uniq.append(x)
    return uniq

# ---------- анти-дубли ----------
RECENT_MSG_IDS = deque(maxlen=500)
def seen_message(update: Update) -> bool:
    mid = getattr(update.message, "message_id", None)
    if mid is None: return False
    if mid in RECENT_MSG_IDS: return True
    RECENT_MSG_IDS.append(mid); return False

# ---------- утилиты CSV ----------
def _stooq_csv_url(ticker: str) -> str:
    return f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"

def _yahoo_csv_url(ticker: str) -> tuple[str, dict]:
    url=f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
    params={"period1": int(time.time())-210*24*3600, "period2": int(time.time()),
            "interval":"1d","events":"history","includeAdjustedClose":"true"}
    return url, params

async def _fetch_csv_one(session: requests.Session, url: str, params: dict|None, name: str) -> pd.Series|None:
    try:
        r = await asyncio.to_thread(session.get, url, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code!=200 or not r.text: return None
        df = pd.read_csv(StringIO(r.text))
        if df.empty or "Date" not in df.columns: return None
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if not col: return None
        s = pd.Series(df[col].values, index=pd.to_datetime(df["Date"]), name=name)
        return s
    except Exception:
        return None

async def _fetch_many_csv_stooq(tickers: list[str]) -> dict[str, pd.Series]:
    sem = asyncio.Semaphore(CONCURRENCY)
    async def go(tk):
        async with sem:
            return tk, await _fetch_csv_one(YF_SESSION, _stooq_csv_url(tk), None, tk)
    res = await asyncio.gather(*[go(t) for t in tickers])
    return {k:v for k,v in res if v is not None}

async def _fetch_many_csv_yahoo(tickers: list[str]) -> dict[str, pd.Series]:
    sem = asyncio.Semaphore(CONCURRENCY)
    async def go(tk):
        async with sem:
            u,p = _yahoo_csv_url(tk)
            return tk, await _fetch_csv_one(YF_SESSION, u, p, tk)
    res = await asyncio.gather(*[go(t) for t in tickers])
    return {k:v for k,v in res if v is not None}

# ---------- одиночный резерв (yfinance) ----------
async def _fetch_one_close_strict(ticker: str) -> pd.Series|None:
    try:
        df = await asyncio.wait_for(asyncio.to_thread(yf.download, ticker, **YF_OPTS), timeout=YF_SINGLE_TIMEOUT)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                lvl0=df.columns.get_level_values(0)
                use='Adj Close' if 'Adj Close' in lvl0 else ('Close' if 'Close' in lvl0 else None)
                s=df[use] if use else None
                if s is not None:
                    if isinstance(s, pd.DataFrame): s=s.iloc[:,0]
                    return s.rename(ticker)
            else:
                col='Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else None)
                if col: return df[col].rename(ticker)
    except Exception:
        return None
    return None

def _trim_incomplete_daily(df: pd.DataFrame) -> pd.DataFrame:
    try:
        ny=ZoneInfo("America/New_York"); now=datetime.now(ny)
        if df.empty: return df
        if df.index.max().date()==now.date() and now.hour<23: return df.iloc[:-1]
        return df
    except Exception: return df

# ---------- главный загрузчик ----------
async def load_close_matrix(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    need = tickers[:]
    got: list[str] = []
    closes = pd.DataFrame()

    # 1) Stooq CSV (параллельно)
    stq = await _fetch_many_csv_stooq(need)
    if stq:
        stq_df = pd.concat(list(stq.values()), axis=1)
        closes = stq_df
        got.extend(stq_df.columns.tolist())

    # 2) Yahoo CSV (параллельно)
    still = [t for t in need if t not in got]
    if still:
        yahoo = await _fetch_many_csv_yahoo(still)
        if yahoo:
            yh_df = pd.concat(list(yahoo.values()), axis=1)
            closes = yh_df if closes.empty else closes.join(yh_df, how="outer")
            got.extend(yh_df.columns.tolist())

    # 3) yfinance single (только если включили флаг)
    still = [t for t in need if t not in got]
    truly_bad = []
    if YF_PRIMARY and still:
        for tk in still:
            s = await _fetch_one_close_strict(tk)
            if s is None or s.dropna().empty:
                truly_bad.append(tk)
            else:
                closes = s.to_frame() if closes.empty else closes.join(s, how="outer")
    else:
        truly_bad = still

    if closes.empty:
        return pd.DataFrame(), truly_bad

    closes = closes.loc[:, ~closes.columns.duplicated()]
    closes.index = pd.to_datetime(closes.index)
    closes = _trim_incomplete_daily(closes)
    closes = closes.sort_index().dropna(how="all").dropna(axis=1, how="all")
    missing = [t for t in need if t not in closes.columns]
    return closes, missing

# ---------- расчётные функции ----------
def pct_change(series: pd.Series, periods: int) -> float:
    if len(series) < periods+1: return np.nan
    a=float(series.iloc[-periods-1]); b=float(series.iloc[-1])
    if a==0 or math.isnan(a) or math.isnan(b): return np.nan
    return b/a-1.0

def confirm_score(pr: pd.Series) -> float:
    if len(pr) < 200: return 0.0
    close=pr.iloc[-1]
    ma50,ma100,ma150,ma200=(pr.rolling(n).mean().iloc[-1] for n in (50,100,150,200))
    return sum(close>ma for ma in (ma50,ma100,ma150,ma200))/4.0

def rank_pct(s: pd.Series) -> pd.Series: return s.rank(pct=True, method="average")
def composite_score(rs21_rel, rs63_rel, conf): return 0.45*rank_pct(rs21_rel.fillna(-1)) + 0.35*rank_pct(rs63_rel.fillna(-1)) + 0.20*conf.fillna(0)

def pick_benchmark_for(ticker: str, closes: pd.DataFrame) -> str:
    BENCH_CANDIDATES = ["SPY","QQQ","IWM","DIA","XLK","XLY","XLF","XLE","XLI","XLP","XLV","XLU","XLB","XLC"]
    cands=[b for b in BENCH_CANDIDATES if b in closes.columns]
    if not cands: return BENCH_DEFAULT
    s=closes[ticker].pct_change().dropna(); best=BENCH_DEFAULT; bc=-2
    for b in cands:
        corr=s.corr(closes[b].pct_change().dropna())
        if pd.notna(corr) and corr>bc: bc=corr; best=b
    return best

def fmt_row(i, tk, row): return f"{i}) {tk} | score={row['score']:.4f} | RS21={row['rs21_rel']:+.2%} | RS63={row['rs63_rel']:+.2%} | conf={row['conf']:.2f}"

# ---------- ранжирование ----------
async def universe_rank(update: Update, context: ContextTypes.DEFAULT_TYPE, universe_name: str, tickers: list[str], top_n: int|None):
    if seen_message(update): return
    bench_need={BENCH_DEFAULT}
    if BENCH_MODE=="smart":
        bench_need.update(["SPY","QQQ","IWM","DIA","XLK","XLY","XLF","XLE","XLI","XLP","XLV","XLU","XLB","XLC"])
    use_tickers = sanitize_us_tickers(sorted(set([t.upper() for t in tickers]+list(bench_need))))
    try:
        closes, bad = await asyncio.wait_for(load_close_matrix(use_tickers), timeout=150)
    except asyncio.TimeoutError:
        await update.message.reply_text("Источник котировок отвечает слишком долго. Попробуйте ещё раз позже."); return

    miss_for_universe=[t for t in tickers if t not in closes.columns]
    bad_all=sorted(set(bad+miss_for_universe))
    if bad_all: await update.message.reply_text("⚠️ Пропущены тикеры без данных: " + ", ".join(bad_all))

    uni=[t for t in tickers if t in closes.columns]
    if not uni:
        await update.message.reply_text("Нет данных для ранжирования (источник котировок недоступен)."); return

    rows=[]
    for tk in uni:
        pr=closes[tk].dropna()
        if pr.empty: continue
        bench=pick_benchmark_for(tk, closes) if BENCH_MODE=="smart" else BENCH_DEFAULT
        if bench not in closes.columns: bench=BENCH_DEFAULT
        last_day=pr.index.max()
        if bench in closes.columns:
            bpr=closes[bench].dropna(); last_day=min(last_day, bpr.index.max())
            pr_use=pr.loc[:last_day]; bpr_use=bpr.loc[:last_day]
        else: pr_use=pr; bpr_use=None
        rs21=pct_change(pr_use,21); rs63=pct_change(pr_use,63)
        if bpr_use is not None:
            brs21=pct_change(bpr_use,21); brs63=pct_change(bpr_use,63)
            rs21_rel=(rs21-brs21) if pd.notna(rs21) and pd.notna(brs21) else np.nan
            rs63_rel=(rs63-brs63) if pd.notna(rs63) and pd.notna(brs63) else np.nan
        else: rs21_rel,rs63_rel=rs21,rs63
        conf=confirm_score(pr_use)
        rows.append({"ticker":tk,"bench":bench,"rs21_rel":rs21_rel,"rs63_rel":rs63_rel,"conf":conf})

    df=pd.DataFrame(rows).set_index("ticker")
    if df.empty: await update.message.reply_text("Нет валидных данных после загрузки котировок."); return
    df["score"]=composite_score(df["rs21_rel"], df["rs63_rel"], df["conf"])
    df_sorted=df.sort_values("score", ascending=False)

    if top_n:
        lines=[f"Топ {top_n} — {universe_name}, сортировка по score:"]
        for i,(tk,row) in enumerate(df_sorted.head(top_n).iterrows(),1): lines.append(fmt_row(i,tk,row))
        await update.message.reply_text("\n".join(lines))
    else:
        await update.message.reply_text(f"Полный рейтинг {universe_name}, сортировка по score:")
        chunk=[]
        for i,(tk,row) in enumerate(df_sorted.iterrows(),1):
            chunk.append(fmt_row(i,tk,row))
            if len(chunk)==30: await update.message.reply_text("\n".join(chunk)); chunk=[]
        if chunk: await update.message.reply_text("\n".join(chunk))

# ---------- ВСЕЛЕННЫЕ ----------
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

ETFALL = [
    "SPY","VOO","IVV","RSP","QQQ","DIA","IWM","IWB","IWR","IJR",
    "XLK","XLY","XLF","XLE","XLI","XLP","XLV","XLU","XLB","XLC",
    "TLT","IEF","HYG","LQD","AGG","BND",
    "GLD","IAU","SLV","GDX",
    "SMH","SOXX","IBB","XBI","ITB","XHB","IYT","XOP","XME","XRT",
    "BITO","LIT","MAGS","ARKK","KRE","KBE","MSTR"
]

ETFX = [
    "TQQQ","SQQQ","SPXL","SPXS","UPRO","SPXU","UDOW","SDOW",
    "SOXL","SOXS","TECL","TECS","FNGU","FNGD",
    "LABU","LABD","TNA","TZA",
    "UCO","SCO","BOIL","KOLD",
    "NUGT","DUST","UVXY",
    "TSLL","WEBL","USD","BITX",
    "GGLL","AAPU","FBL","MSFU","AMZU","NVDL","CONL"
]

# ---------- команды ----------
async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    tz=ZoneInfo(BOT_TZ); now=datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    await update.message.reply_text(f"Я на связи ✅\n{now} {BOT_TZ}")

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    me = await context.bot.get_me()
    await update.message.reply_text(f"Bot: @{me.username} (id={me.id})\nMODE={BENCH_MODE}  DEFAULT_BENCH={BENCH_DEFAULT}\nTZ={BOT_TZ}")

async def sp5005_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):  await universe_rank(update, context, "SP250", SP250, top_n=5)
async def sp50010_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE): await universe_rank(update, context, "SP250", SP250, top_n=10)
async def etfall_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):   await universe_rank(update, context, "ETFALL", ETFALL, top_n=10)
async def etfx_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):     await universe_rank(update, context, "ETFX",   ETFX,   top_n=10)

async def etf_full_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    args=[a.lower() for a in context.args] if context.args else []
    name="etfall"; tickers=ETFALL
    if args and args[0] in ("etfall","etfx"): name=args[0]
    if name=="etfx": tickers=ETFX
    await universe_rank(update, context, name.upper(), tickers, top_n=None)

async def benchmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    global BENCH_MODE
    if context.args:
        mode=context.args[0].lower().strip()
        if mode in ("global","smart"):
            BENCH_MODE=mode; await update.message.reply_text(f"Режим бенчмарка: {BENCH_MODE}"); return
    await update.message.reply_text("Формат: /benchmode <global|smart>")

async def setbenchmark_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    global BENCH_DEFAULT
    if not context.args: await update.message.reply_text("Формат: /setbenchmark <тикер>"); return
    BENCH_DEFAULT=context.args[0].upper(); await update.message.reply_text(f"Глобальный бенч: {BENCH_DEFAULT}")

async def benchcandidates_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    await update.message.reply_text("Кандидаты SMART:\nSPY QQQ IWM DIA XLK XLY XLF XLE XLI XLP XLV XLU XLB XLC")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error("Unhandled error: %s", context.error)
    try:
        if update and getattr(update, "effective_message", None):
            await update.effective_message.reply_text("⚠️ Внутренняя ошибка. Проверьте позже.")
    except Exception: pass

# ---------- MAIN ----------
def main() -> None:
    if not BOT_TOKEN: raise SystemExit("Set TELEGRAM_TOKEN env var.")
    app = Application.builder().token(BOT_TOKEN).concurrent_updates(True).build()

    # очищаем возможный webhook, чтобы не было конфликтов
    import asyncio as _a
    _a.get_event_loop().run_until_complete(app.bot.delete_webhook(drop_pending_updates=True))

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
    app.add_error_handler(error_handler)

    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

