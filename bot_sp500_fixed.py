# -*- coding: utf-8 -*-
"""
bot_sp500_quant.py — Переписанный "по-квантовому" ранкер для Telegram

Ключевое отличие от исходника:
- Композитный скор на базе набора факторов:
  • Моментум 12–1 и 6–1 (с пропуском последнего месяца)
  • Остаточный моментум (market-neutral) через rolling beta к бенчмарку
  • Близость к 52-нед. максимуму
  • Качество тренда (R² линейного тренда лог-цены за 252 дня)
  • Краткосрочный реверс (против последнего месяца)
  • ConfirmScore (MA50/100/150/200)
- Риск-штрафы: downside semivol (63д), rolling Max Drawdown (252д), неликвидность через ADV ($)
- Все сигналы считаются кросс-секционно и нормируются (z-score) в дату ранжирования
- Асинхронная загрузка котировок (Stooq/Yahoo CSV; yfinance — опционально)
- Режимы: SMART-бенчмарк на тикер (корреляционная привязка) или глобальный BENCH_DEFAULT

Зависимости:
python-telegram-bot==20.7, yfinance==0.2.43, pandas==2.2.2, numpy==1.26.4, requests>=2.31, beautifulsoup4, lxml, pandas_datareader==0.10.0

Важно: В учебных целях. Не является инвестсоветом.
"""

import os, math, time, asyncio, logging, warnings
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

# -------------------------- Лог --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("quant-rank-bot")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# -------------------------- ENV --------------------------
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
BENCH_DEFAULT = os.getenv("BENCH_DEFAULT", "SPY").strip().upper()
BENCH_MODE = os.getenv("BENCH_MODE", "smart").strip().lower()  # "global" | "smart"
BOT_TZ = os.getenv("BOT_TZ", "Europe/Berlin")
SECTOR_NEUTRAL = os.getenv("SECTOR_NEUTRAL", "false").strip().lower() in ("1","true","yes","y")
MIN_PRICE = float(os.getenv("MIN_PRICE", "5.0"))
MIN_ADV_USD = float(os.getenv("MIN_ADV_USD", "2000000"))
USE_CONFIRM = os.getenv("USE_CONFIRM", "true").strip().lower() in ("1","true","yes","y")

# ------------------ Загрузка котировок (цена + объём) ------------------
YF_SESSION = requests.Session()
YF_SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "*/*", "Connection": "keep-alive"})
YF_OPTS = dict(period="365d", interval="1d", auto_adjust=True, progress=False, threads=False, session=YF_SESSION)
CONCURRENCY = 12
HTTP_TIMEOUT = 6
YF_SINGLE_TIMEOUT = 8
YF_PRIMARY = False  # резерв

# --- США-санитайзер
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

# --- анти-дубли сообщений Telegram
RECENT_MSG_IDS = deque(maxlen=500)
def seen_message(update: Update) -> bool:
    mid = getattr(update.message, "message_id", None)
    if mid is None: return False
    if mid in RECENT_MSG_IDS: return True
    RECENT_MSG_IDS.append(mid); return False

# --- CSV источники
def _stooq_csv_url(ticker: str) -> str:
    return f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"

def _yahoo_csv_url(ticker: str) -> tuple[str, dict]:
    url=f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
    params={"period1": int(time.time())-400*24*3600, "period2": int(time.time()),
            "interval":"1d","events":"history","includeAdjustedClose":"true"}
    return url, params

async def _fetch_csv_one(session: requests.Session, url: str, params: dict|None, name: str) -> tuple[pd.Series|None, pd.Series|None]:
    try:
        r = await asyncio.to_thread(session.get, url, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code!=200 or not r.text: return None, None
        df = pd.read_csv(StringIO(r.text))
        if df.empty or "Date" not in df.columns: return None, None
        # Столбцы цены и объёма
        col_price = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        col_vol = "Volume" if "Volume" in df.columns else None
        if not col_price: return None, None
        s_px = pd.Series(df[col_price].values, index=pd.to_datetime(df["Date"]), name=name)
        s_vol = pd.Series(df[col_vol].values, index=pd.to_datetime(df["Date"]), name=name) if col_vol else None
        return s_px, s_vol
    except Exception:
        return None, None

async def _fetch_many_csv_stooq(tickers: list[str]) -> tuple[dict[str,pd.Series], dict[str,pd.Series]]:
    sem = asyncio.Semaphore(CONCURRENCY)
    async def go(tk):
        async with sem:
            px, vol = await _fetch_csv_one(YF_SESSION, _stooq_csv_url(tk), None, tk)
            return tk, px, vol
    res = await asyncio.gather(*[go(t) for t in tickers])
    px = {k:v for k, v, _ in res if v is not None}
    vo = {k:v for k, _, v in res if v is not None}
    return px, vo

async def _fetch_many_csv_yahoo(tickers: list[str]) -> tuple[dict[str,pd.Series], dict[str,pd.Series]]:
    sem = asyncio.Semaphore(CONCURRENCY)
    async def go(tk):
        async with sem:
            u,p = _yahoo_csv_url(tk)
            px, vol = await _fetch_csv_one(YF_SESSION, u, p, tk)
            return tk, px, vol
    res = await asyncio.gather(*[go(t) for t in tickers])
    px = {k:v for k, v, _ in res if v is not None}
    vo = {k:v for k, _, v in res if v is not None}
    return px, vo

async def _fetch_one_close_strict(ticker: str) -> tuple[pd.Series|None, pd.Series|None]:
    try:
        df = await asyncio.wait_for(asyncio.to_thread(yf.download, ticker, **YF_OPTS), timeout=YF_SINGLE_TIMEOUT)
        if df is not None and not df.empty:
            # Цена
            if isinstance(df.columns, pd.MultiIndex):
                lvl0=df.columns.get_level_values(0)
                use='Adj Close' if 'Adj Close' in lvl0 else ('Close' if 'Close' in lvl0 else None)
                s_px=df[use] if use else None
                if isinstance(s_px, pd.DataFrame): s_px=s_px.iloc[:,0]
            else:
                col='Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else None)
                s_px=df[col] if col else None
            # Объём
            s_vol = None
            try:
                s_vol = df['Volume'] if 'Volume' in df.columns else None
                if isinstance(s_vol, pd.DataFrame): s_vol=s_vol.iloc[:,0]
            except Exception:
                s_vol = None
            if s_px is not None:
                return s_px.rename(ticker), (s_vol.rename(ticker) if s_vol is not None else None)
    except Exception:
        return None, None
    return None, None

def _trim_incomplete_daily(df: pd.DataFrame) -> pd.DataFrame:
    try:
        ny=ZoneInfo("America/New_York"); now=datetime.now(ny)
        if df.empty: return df
        if df.index.max().date()==now.date() and now.hour<23: return df.iloc[:-1]
        return df
    except Exception: return df

async def load_price_volume_matrix(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    need = tickers[:]
    got: list[str] = []
    px_df = pd.DataFrame()
    vol_df = pd.DataFrame()

    # 1) Stooq
    stq_px, stq_vol = await _fetch_many_csv_stooq(need)
    if stq_px:
        px_df = pd.concat(list(stq_px.values()), axis=1)
        got.extend(list(stq_px.keys()))
    if stq_vol:
        vol_df = pd.concat(list(stq_vol.values()), axis=1)

    # 2) Yahoo
    still = [t for t in need if t not in got]
    if still:
        yh_px, yh_vol = await _fetch_many_csv_yahoo(still)
        if yh_px:
            yh_px_df = pd.concat(list(yh_px.values()), axis=1)
            px_df = yh_px_df if px_df.empty else px_df.join(yh_px_df, how="outer")
            got.extend(list(yh_px.keys()))
        if yh_vol:
            yh_vol_df = pd.concat(list(yh_vol.values()), axis=1)
            vol_df = yh_vol_df if vol_df.empty else vol_df.join(yh_vol_df, how="outer")

    # 3) yfinance одиночный
    still = [t for t in need if t not in got]
    truly_bad = []
    if YF_PRIMARY and still:
        for tk in still:
            s_px, s_vol = await _fetch_one_close_strict(tk)
            if s_px is None or s_px.dropna().empty:
                truly_bad.append(tk)
            else:
                px_df = s_px.to_frame() if px_df.empty else px_df.join(s_px, how="outer")
                if s_vol is not None and not s_vol.dropna().empty:
                    vol_df = s_vol.to_frame() if vol_df.empty else vol_df.join(s_vol, how="outer")
    else:
        truly_bad = still

    if px_df.empty:
        return pd.DataFrame(), pd.DataFrame(), truly_bad

    px_df = px_df.loc[:, ~px_df.columns.duplicated()].sort_index()
    vol_df = vol_df.loc[:, [c for c in px_df.columns if c in vol_df.columns]].sort_index() if not vol_df.empty else vol_df

    px_df.index = pd.to_datetime(px_df.index)
    vol_df.index = pd.to_datetime(vol_df.index) if not vol_df.empty else vol_df.index

    px_df = _trim_incomplete_daily(px_df).dropna(how="all").dropna(axis=1, how="all")
    if not vol_df.empty:
        vol_df = vol_df.reindex(px_df.index).dropna(how="all").dropna(axis=1, how="all")

    missing = [t for t in need if t not in px_df.columns]
    return px_df, vol_df, missing

# ------------------ Квант-факторы ------------------

def pct_change_safe(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change()
    return rets.replace([np.inf, -np.inf], np.nan)

def momentum_return(prices: pd.DataFrame, lookback: int, gap: int = 21) -> pd.Series:
    return (prices.shift(gap).iloc[-1] / prices.shift(lookback).iloc[-1]) - 1.0

def short_term_reversal(prices: pd.DataFrame, window: int = 21) -> pd.Series:
    r1m = (prices.iloc[-1] / prices.shift(window).iloc[-1]) - 1.0
    return -r1m

def proximity_52w_high(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    roll_max = prices.rolling(window).max().iloc[-1]
    return prices.iloc[-1] / roll_max

def rolling_trend_r2_last(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    logp = np.log(prices.replace(0, np.nan)).tail(window)
    out = {}
    for tk in logp.columns:
        y = logp[tk].dropna().values
        if len(y) < 20:
            out[tk] = np.nan; continue
        x = np.arange(len(y), dtype=float)
        x_mean = x.mean(); y_mean = y.mean()
        cov_xy = ((x - x_mean) * (y - y_mean)).sum()
        var_x = ((x - x_mean) ** 2).sum()
        var_y = ((y - y_mean) ** 2).sum()
        if var_x <= 0 or var_y <= 0:
            out[tk] = np.nan; continue
        beta = cov_xy / var_x
        y_hat = y_mean + beta * (x - x_mean)
        sse = ((y - y_hat) ** 2).sum()
        r2 = 1.0 - sse / var_y if var_y>0 else np.nan
        out[tk] = float(r2)
    return pd.Series(out)

def downside_semi_vol_last(prices: pd.DataFrame, window: int = 63) -> pd.Series:
    r = pct_change_safe(prices).tail(window)
    r_pos_to_zero = r.copy(); r_pos_to_zero[r_pos_to_zero > 0] = 0.0
    semi = np.sqrt((r_pos_to_zero.pow(2)).mean())
    return semi

def rolling_max_drawdown_last(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    px = prices.tail(window)
    roll_max = px.cummax()
    dd = px/roll_max - 1.0
    return dd.min()

def residual_momentum_last(prices: pd.DataFrame, market: pd.Series, lookback: int = 252, gap: int = 21) -> pd.Series:
    market = market.reindex(prices.index).ffill().dropna()
    r_m = pct_change_safe(market.to_frame()).iloc[:,0]
    r = pct_change_safe(prices).dropna(how="all")
    window = lookback
    # beta_ticker = Cov(r_i, r_m) / Var(r_m) за окно lookback (берём последнее значение)
    cov = r.tail(window).apply(lambda s: np.cov(s.dropna().align(r_m.tail(window), join='inner')[0], r_m.tail(window).dropna())[0,1] if s.dropna().size>5 else np.nan)
    var_m = np.var(r_m.tail(window).dropna())
    beta = cov / (var_m if var_m>0 else np.nan)
    resid = r.sub(r_m * beta, axis=0)
    one_plus = 1.0 + resid
    cum = one_plus.cumprod()
    val = (cum.shift(gap).iloc[-1] / cum.shift(lookback).iloc[-1]) - 1.0
    return val

def confirm_score_last(prices: pd.DataFrame) -> pd.Series:
    if prices.shape[0] < 200: return pd.Series(0.0, index=prices.columns)
    close = prices.iloc[-1]
    ma = {n: prices.rolling(n).mean().iloc[-1] for n in (50,100,150,200)}
    score = sum((close > ma[n]).astype(float) for n in (50,100,150,200)) / 4.0
    return score

def adv_usd_last(prices: pd.DataFrame, volumes: pd.DataFrame|None, window: int = 63) -> pd.Series:
    if volumes is None or volumes.empty: 
        return pd.Series(np.nan, index=prices.columns)
    dollar_vol = (volumes * prices).rolling(window).mean()
    return dollar_vol.iloc[-1]

def cs_zscore_last(series_df: pd.DataFrame|dict[str,pd.Series], clip: float = 3.0) -> dict[str, pd.Series]:
    """На вход даём словарь факторов {name: Series}; на выходе — те же Series с z-score."""
    if isinstance(series_df, dict):
        factors = series_df
    else:
        raise ValueError("cs_zscore_last: ожидается dict[name->Series].")
    # общий набор тикеров
    common = None
    for s in factors.values():
        if common is None: common = s.index
        else: common = common.intersection(s.index)
    out = {}
    df = pd.DataFrame({k: v.reindex(common) for k, v in factors.items()})
    mu = df.mean(axis=1, skipna=True)
    sd = df.std(axis=1, ddof=0, skipna=True).replace(0, np.nan)
    z = (df.sub(mu, axis=0)).div(sd, axis=0).clip(-clip, clip)
    for k in df.columns:
        out[k] = z[k]
    return out

# ------------------ Бенчмарк-подбор ------------------

def pick_benchmark_for(ticker: str, closes: pd.DataFrame) -> str:
    BENCH_CANDIDATES = ["SPY","QQQ","IWM","DIA","XLK","XLY","XLF","XLE","XLI","XLP","XLV","XLU","XLB","XLC"]
    cands=[b for b in BENCH_CANDIDATES if b in closes.columns]
    if not cands: return BENCH_DEFAULT
    s=closes[ticker].pct_change().dropna(); best=BENCH_DEFAULT; bc=-2
    for b in cands:
        corr=s.corr(closes[b].pct_change().dropna())
        if pd.notna(corr) and corr>bc: bc=corr; best=b
    return best

# ------------------ Кванто-ранжирование вселенной ------------------

def composite_score_quant(signals: dict[str,pd.Series]) -> pd.Series:
    """
    Принимает нормированные (z) факторы. Положительные: +, штрафы: -.
    Весовая схема подобрана консервативно.
    """
    w_pos = {
        "mom12_1": 0.30,
        "mom6_1": 0.20,
        "resid_mom12_1": 0.20,
        "r2_trend": 0.10,
        "prox_52w": 0.10,
        "reversal1m": 0.05,
        "confirm": 0.05,
    }
    w_pen = {
        "downside_vol": -0.08,
        "max_dd": -0.07,
        "illiquidity": -0.05,
    }
    score = None
    for k,w in w_pos.items():
        if k in signals:
            score = (signals[k]*w) if score is None else score + signals[k]*w
    for k,w in w_pen.items():
        if k in signals:
            score = score + signals[k]*w if score is not None else signals[k]*w
    return score

async def quant_rank(update: Update, context: ContextTypes.DEFAULT_TYPE, universe_name: str, tickers: list[str], top_n: int|None):
    if seen_message(update): return
    bench_need={BENCH_DEFAULT}
    if BENCH_MODE=="smart":
        bench_need.update(["SPY","QQQ","IWM","DIA","XLK","XLY","XLF","XLE","XLI","XLP","XLV","XLU","XLB","XLC"])
    use_tickers = sanitize_us_tickers(sorted(set([t.upper() for t in tickers]+list(bench_need))))

    try:
        closes, volumes, bad = await asyncio.wait_for(load_price_volume_matrix(use_tickers), timeout=180)
    except asyncio.TimeoutError:
        await update.message.reply_text("Источник котировок отвечает слишком долго. Попробуйте ещё раз позже."); return

    miss_for_universe=[t for t in tickers if t not in closes.columns]
    bad_all=sorted(set(bad+miss_for_universe))
    if bad_all: await update.message.reply_text("⚠️ Пропущены тикеры без данных: " + ", ".join(bад_all))

    uni=[t for t in tickers if t in closes.columns]
    if not uni:
        await update.message.reply_text("Нет данных для ранжирования."); return

    # Выравниваем дату: берём последнюю общую дату с бенчмарками
    last_date = closes[uni].dropna(how="all").index.max()
    if BENCH_DEFAULT in closes.columns:
        last_date = min(last_date, closes[BENCH_DEFAULT].dropna().index.max())
    closes = closes.loc[:last_date]
    volumes = volumes.loc[:last_date] if volumes is not None and not volumes.empty else volumes

    # Бенчмарк для остаточного момента
    market = closes[BENCH_DEFAULT].dropna() if BENCH_DEFAULT in closes.columns else None

    # Фильтры ликвидности/цены
    price_now = closes[uni].iloc[-1]
    adv_now = adv_usd_last(closes[uni], volumes[uni] if volumes is not None and not volumes.empty else None, window=63)
    liquid_mask = (price_now >= MIN_PRICE) & (adv_now >= MIN_ADV_USD if adv_now.notna().any() else True)
    if not liquid_mask.any():
        await update.message.reply_text("После фильтров цена/ликвидность не осталось бумаг."); return
    universe = [t for t in uni if liquid_mask.get(t, False)]

    P = closes[universe].dropna(how="all")
    # Сигналы (последние значения)
    mom12_1 = momentum_return(P, lookback=252, gap=21)
    mom6_1  = momentum_return(P, lookback=126, gap=21)
    reversal1m = short_term_reversal(P, window=21)
    prox_52w = proximity_52w_high(P, window=252)
    r2_trend = rolling_trend_r2_last(P, window=252)
    resid_mom12_1 = residual_momentum_last(P, market, lookback=252, gap=21) if market is not None else pd.Series(np.nan, index=P.columns)
    dsv = downside_semi_vol_last(P, window=63)
    mdd = rolling_max_drawdown_last(P, window=252)
    adv = adv_usd_last(P, volumes[universe] if volumes is not None and not volumes.empty else None, window=63)

    confirm = confirm_score_last(P) if USE_CONFIRM else pd.Series(0.0, index=P.columns)

    # Относительные к SMART-бенчмарку ренты на 1м/3м (как у вас) — для информации/отладки
    # и чтобы пользователю было привычно видеть RS21/RS63:
    rows_rs={}
    for tk in universe:
        bench = pick_benchmark_for(tk, closes) if BENCH_MODE=="smart" else BENCH_DEFAULT
        pr=closes[tk].dropna(); bpr=closes[bench].dropna() if bench in closes.columns else None
        last = min(pr.index.max(), bpr.index.max()) if bpr is not None else pr.index.max()
        pr_use = pr.loc[:last]; bpr_use = bpr.loc[:last] if bpr is not None else None
        def pc(series, periods): 
            if len(series)<periods+1: return np.nan
            a=float(series.iloc[-periods-1]); b=float(series.iloc[-1])
            return (b/a-1.0) if a!=0 else np.nan
        rs21=pc(pr_use,21); rs63=pc(pr_use,63)
        brs21=pc(bpr_use,21) if bpr_use is not None else np.nan
        brs63=pc(bpr_use,63) if bpr_use is not None else np.nan
        rows_rs[tk]=(bench, (rs21 - brs21) if pd.notna(rs21) and pd.notna(brs21) else np.nan,
                           (rs63 - brs63) if pd.notna(rs63) and pd.notna(brs63) else np.nan)

    # Нормируем факторы (z-score по кросс-секции)
    z_factors = cs_zscore_last({
        "mom12_1": mom12_1,
        "mom6_1": mom6_1,
        "resid_mom12_1": resid_mom12_1,
        "reversal1m": reversal1m,
        "prox_52w": prox_52w,
        "r2_trend": r2_trend,
        "downside_vol": dsv,
        "max_dd": mdd,
        "illiquidity": -adv,  # меньше ADV => хуже
        "confirm": confirm,
    })

    score = composite_score_quant(z_factors)
    df = pd.DataFrame({
        "score": score,
        "mom12_1": z_factors["mom12_1"],
        "mom6_1": z_factors["mom6_1"],
        "resid_mom12_1": z_factors["resid_mom12_1"],
        "r2_trend": z_factors["r2_trend"],
        "prox_52w": z_factors["prox_52w"],
        "rev1m": z_factors["reversal1m"],
        "dsv": z_factors["downside_vol"],
        "mdd": z_factors["max_dd"],
        "illiquidity": z_factors["illiquidity"],
        "confirm": z_factors["confirm"],
    }).dropna(subset=["score"]).sort_values("score", ascending=False)

    # Вспом. столбцы RS21/RS63 к SMART-бенчу (не в композите):
    rs21_rel = []; rs63_rel = []; benchs = []
    for tk in df.index:
        bench, rs21, rs63 = rows_rs.get(tk, (BENCH_DEFAULT, np.nan, np.nan))
        rs21_rel.append(rs21); rs63_rel.append(rs63); benchs.append(bench)
    df["bench"]=benchs; df["RS21_rel"]=rs21_rel; df["RS63_rel"]=rs63_rel

    # Ответ пользователю
    def fmt_row(i, tk, row): 
        return (f"{i}) {tk} | score={row['score']:+.3f} | RS21={row['RS21_rel']:+.2%} | RS63={row['RS63_rel']:+.2%} | "
                f"12-1Z={row['mom12_1']:+.2f} 6-1Z={row['mom6_1']:+.2f} residZ={row['resid_mom12_1']:+.2f} "
                f"R²Z={row['r2_trend']:+.2f} 52wZ={row['prox_52w']:+.2f} revZ={row['rev1m']:+.2f} "
                f"dsvZ={row['dsv']:+.2f} mddZ={row['mdd']:+.2f} illiqZ={row['illiquidity']:+.2f} confZ={row['confirm']:+.2f} "
                f"| bench={row['bench']}")

    if top_n:
        lines=[f"Топ {top_n} — {universe_name} (quant-rank):"]
        for i,(tk,row) in enumerate(df.head(top_n).iterrows(),1): lines.append(fmt_row(i,tk,row))
        await update.message.reply_text("\n".join(lines))
    else:
        await update.message.reply_text(f"Полный quant-рейтинг {universe_name}:")
        chunk=[]
        for i,(tk,row) in enumerate(df.iterrows(),1):
            chunk.append(fmt_row(i,tk,row))
            if len(chunk)==25: await update.message.reply_text("\n".join(chunk)); chunk=[]
        if chunk: await update.message.reply_text("\n".join(chunk))

# ------------------ Вселенные ------------------
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

# ------------------ Команды ------------------

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    tz=ZoneInfo(BOT_TZ); now=datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    await update.message.reply_text(f"Я на связи ✅\n{now} {BOT_TZ}")

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if seen_message(update): return
    me = await context.bot.get_me()
    await update.message.reply_text(f"Bot: @{me.username} (id={me.id})\nMODE={BENCH_MODE}  DEFAULT_BENCH={BENCH_DEFAULT}\nTZ={BOT_TZ}\nSECTOR_NEUTRAL={SECTOR_NEUTRAL}")

# Квант-команды
async def sp500q5_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):  await quant_rank(update, context, "SP250", SP250, top_n=5)
async def sp500q10_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE): await quant_rank(update, context, "SP250", SP250, top_n=10)
async def etfallq_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):   await quant_rank(update, context, "ETFALL", ETFALL, top_n=10)
async def etfxq_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):     await quant_rank(update, context, "ETFX",   ETFX,   top_n=10)

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

# ------------------ MAIN ------------------
def main() -> None:
    if not BOT_TOKEN: raise SystemExit("Set TELEGRAM_TOKEN env var.")
    app = Application.builder().token(BOT_TOKEN).concurrent_updates(True).build()

    # очищаем возможный webhook, чтобы не было конфликтов
    import asyncio as _a
    _a.get_event_loop().run_until_complete(app.bot.delete_webhook(drop_pending_updates=True))

    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("diag", diag_cmd))

    # Квантовые команды
    app.add_handler(CommandHandler("sp500q5", sp500q5_cmd))
    app.add_handler(CommandHandler("sp500q10", sp500q10_cmd))
    app.add_handler(CommandHandler("etfallq", etfallq_cmd))
    app.add_handler(CommandHandler("etfxq", etfxq_cmd))

    # Настройки бенча
    app.add_handler(CommandHandler("benchmode", benchmode_cmd))
    app.add_handler(CommandHandler("setbenchmark", setbenchmark_cmd))
    app.add_handler(CommandHandler("benchcandidates", benchcandidates_cmd))

    app.add_error_handler(error_handler)
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
