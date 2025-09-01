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

# ────────────────────────────────────────────────────────────────────────────────
# ENV (можно задать в Render → Environment)
# ────────────────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
AI_CSV_URL = os.getenv("AI_CSV_URL", "none").strip()        # "none" или URL CSV: ticker,ai_score
W_AI = float(os.getenv("W_AI", "0.0"))                      # 0..1 — вес AI в композите
BENCH_DEFAULT = os.getenv("BENCH_DEFAULT", "SPY").strip().upper()
BENCH_MODE = os.getenv("BENCH_MODE", "global").strip().lower()  # "global" | "smart"
BOT_TZ = os.getenv("BOT_TZ", "Europe/Berlin")

# Кандидаты бенчмарков для SMART-режима (корреляционный выбор по тикеру)
BENCH_CANDIDATES = [
    "SPY","QQQ","IWM","DIA",
    "XLK","XLV","XLF","XLY","XLI","XLC","XLP","XLE","XLU","XLB","XLRE"
]

# ────────────────────────────────────────────────────────────────────────────────
# === UNIVERSES ===
# SP250  — фиксированный список акций (S&P 500 Top-250 по весу)
# ETFALL — 50 обычных ETF (без плеча/инверса)
# ETFX   — расширенный список (плечевые/инверсные/спец)
# ────────────────────────────────────────────────────────────────────────────────

SP250 = [
"NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK-B","TSLA",
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
"AMP","GWW","CBRE","ROST","OKE","EW","DDOG","VLO","CPRT","OXY",
"F","GRMN","AIG","KR","BKR","EXC","MSCI","TGT","CCL","CCI",
"FANG","TTWO","EA","KMB","XEL","AME","EBAY","PEG","YUM","DAL",
"RMD","MPWR","KVUE","LVS","KDP","ETR","LYV","ROK","PRU","SYY"
]

ETFALL = [
    # Broad US market / large-cap
    "SPY","VOO","IVV","VTI","ITOT",
    # Tech/Nasdaq & Semis
    "QQQ","XLK","SMH","SOXX",
    # Mid/Small
    "IWM","IJH","IJR","MDY",
    # Sectors
    "XLE","XLF","XLY","XLV","XLI","XLC","XLP","XLU","XLB","XLRE",
    # Style
    "VTV","VUG","IWD","IWF",
    # Dividends
    "VYM","SCHD",
    # Bonds
    "AGG","BND","TLT","LQD","HYG","IEF","SHY","JNK","BNDX",
    # International
    "EFA","VEA","EEM","VWO",
    # REITs
    "VNQ",
    # Commodities / tips
    "GLD","IAU","TIP",
    # Added populars
    "BITO","LIT","RSP","MAGS"
]

ETFX = [
    # QQQ 3x / -3x
    "TQQQ","SQQQ",
    # Semis 3x / -3x
    "SOXL","SOXS",
    # S&P500 3x / -3x
    "SPXL","SPXS","UPRO","SPXU",
    # Tech 3x / -3x
    "TECL","TECS","WEBL","WEBS",
    # Russell 2000 3x / -3x
    "TNA","TZA",
    # Biotech 3x / -3x
    "LABU","LABD",
    # Financials 3x / -3x
    "FAS","FAZ",
    # Energy 2x / -2x
    "ERX","ERY",
    # Oil 2x / -2x
    "UCO","SCO",
    # Natural Gas 2x / -2x
    "BOIL","KOLD",
    # QQQ 2x / -2x
    "QLD","QID",
    # 20+Y Treasuries 3x / -3x
    "TMF","TMV",
    # Gold miners bull/bear
    "NUGT","DUST",
    # Tech single-stock (Tesla) 1.5x/-1.5x (proxy via TSLL/TSLS)
    "TSLL","TSLS",
    # Dow 3x bull/bear
    "UDOW","SDOW",
    # Semiconductors bear alt, etc.
    # Special/crypto
    "BITX",
    # Utilities 2x bear (proxy) — пара к USD
    "USD","SSG"
]

ETF_CORE2x3 = ["SPY","QQQ","IWM","XLK","XLV","XLF"]

# ────────────────────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────────────────────

def _safe_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 20) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def load_ai_scores_csv(url: str):
    if not url or url.lower() == "none":
        return None
    import pandas as pd
    df = pd.read_csv(url)
    # нормализуем названия
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if "ticker" not in df.columns or "ai_score" not in df.columns:
        raise ValueError("AI CSV must have columns: ticker, ai_score")
    df["ticker"] = (
        df["ticker"].astype(str).str.upper()
        .str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    )
    return {row.ticker: float(row.ai_score) for row in df.itertuples(index=False)}

# ────────────────────────────────────────────────────────────────────────────────
# Market data & features
# ────────────────────────────────────────────────────────────────────────────────

def fetch_features(tickers: list[str]) -> pd.DataFrame:
    """
    Возвращает MultiIndex (date, ticker) с колонками:
    close, high, low, volume, adv20, atr20, volRel, mom, ret21
    """
    # Берём ~400 дней, чтобы хватило на 252д high и 200д MA
    data = yf.download(
        tickers,
        period="430d",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        # по каждому тикеру
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t].rename(columns=str.lower)
            if not {"close","high","low","volume"}.issubset(df.columns):
                continue
            df = df.dropna(subset=["close"])
            if df.empty:
                continue
            atr20 = _atr(df["high"], df["low"], df["close"], 20)
            adv20 = (df["close"] * df["volume"]).rolling(20).mean()
            volRel = atr20 / df["close"]
            # Моментум 6м-1м (примерно 126д и 21д)
            c = df["close"]
            ret126 = c.pct_change(126)
            ret21 = c.pct_change(21)
            mom = ret126 - ret21
            out = pd.DataFrame({
                "ticker": t,
                "close": df["close"],
                "high": df["high"],
                "low": df["low"],
                "volume": df["volume"],
                "adv20": adv20,
                "atr20": atr20,
                "volRel": volRel,
                "mom": mom,
                "ret21": (1 + c.pct_change()).rolling(21).apply(np.prod, raw=True) - 1.0
            })
            frames.append(out)
    else:
        # единичный тикер
        df = data.rename(columns=str.lower)
        t = tickers[0]
        atr20 = _atr(df["high"], df["low"], df["close"], 20)
        adv20 = (df["close"] * df["volume"]).rolling(20).mean()
        volRel = atr20 / df["close"]
        c = df["close"]
        ret126 = c.pct_change(126)
        ret21 = c.pct_change(21)
        mom = ret126 - ret21
        out = pd.DataFrame({
            "ticker": t,
            "close": df["close"],
            "high": df["high"],
            "low": df["low"],
            "volume": df["volume"],
            "adv20": adv20,
            "atr20": atr20,
            "volRel": volRel,
            "mom": mom,
            "ret21": (1 + c.pct_change()).rolling(21).apply(np.prod, raw=True) - 1.0
        })
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["ticker","close","high","low","volume","adv20","atr20","volRel","mom","ret21"]).set_index(["ticker"])

    df_all = pd.concat(frames)
    df_all = df_all.reset_index().rename(columns={"index":"date"})
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.set_index(["date","ticker"]).sort_index()
    return df_all

def compute_score_snap(feats: pd.DataFrame, ai_map: dict | None = None, w_ai: float = 0.0) -> pd.DataFrame:
    last_date = feats.index.get_level_values(0).max()
    snap = feats.xs(last_date, level=0).reset_index()

    zM = _safe_z(snap["mom"].values)
    zL = _safe_z(np.log(np.clip(snap["adv20"].values, 1.0, None)))
    zV = _safe_z(snap["volRel"].values)

    base = 1.0*zM + 0.3*zL - 0.3*zV

    out = pd.DataFrame({
        "ticker": snap["ticker"].values,
        "score_base": base,
        "ret21": snap["ret21"].values,
        "volRel": snap["volRel"].values,
    })

    w = max(0.0, min(1.0, float(w_ai)))
    if ai_map:
        out["ai_score"] = out["ticker"].map(ai_map)
        if out["ai_score"].notna().sum() >= 2:
            z_ai = _safe_z(out["ai_score"].fillna(out["ai_score"].mean()).values)
            z_base = _safe_z(out["score_base"].values)
            out["score"] = (1-w)*z_base + w*z_ai
        else:
            out["score"] = _safe_z(out["score_base"].values)
    else:
        out["ai_score"] = np.nan
        out["score"] = _safe_z(out["score_base"].values)

    return out

# ────────────────────────────────────────────────────────────────────────────────
# ConfirmScore analytics
# ────────────────────────────────────────────────────────────────────────────────
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_n = _ema(tr, n)
    pdm_n = _ema(plus_dm, n)
    mdm_n = _ema(minus_dm, n)
    pdi = 100 * (pdm_n / tr_n)
    mdi = 100 * (mdm_n / tr_n)
    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    adx = _ema(dx, n)
    return adx

def _linreg_slope_r2(y: pd.Series) -> tuple[float, float]:
    y = y.dropna()
    n = len(y)
    if n < 5:
        return float("nan"), float("nan")
    x = np.arange(n)
    slope, intercept = np.polyfit(x, y.values, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y.values - y_hat)**2)
    ss_tot = np.sum((y.values - np.mean(y.values))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else float("nan")
    return float(slope), float(r2)

def _hv(series: pd.Series, n: int) -> float:
    r = np.log(series/series.shift(1)).dropna()
    if len(r) < max(5, n//2):
        return float("nan")
    return float(r.tail(n).std(ddof=0))

def _max_drawdown(series: pd.Series, n: int) -> float:
    s = series.dropna().tail(n)
    if s.empty:
        return float("nan")
    roll_max = s.cummax()
    dd = (s/roll_max - 1.0)
    return float(dd.min())

def _up_down_vol_ratio(close: pd.Series, volume: pd.Series, n: int = 20) -> float:
    ret = close.pct_change()
    up_vol = volume.where(ret > 0, 0.0).tail(n).sum()
    down_vol = volume.where(ret < 0, 0.0).tail(n).sum()
    if down_vol == 0:
        return float("inf") if up_vol > 0 else float("nan")
    return float(up_vol / down_vol)

def fetch_benchmark_series(tickers: list[str], period: str = "220d") -> dict[str, pd.Series]:
    out = {}
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                s = data[t]["Close"].dropna()
                s.name = t
                out[t] = s
    else:
        s = data["Close"].dropna()
        s.name = tickers[0]
        out[tickers[0]] = s
    return out

def pick_best_benchmark(close: pd.Series, bench_map: dict[str, pd.Series]) -> str:
    ret_s = close.pct_change().dropna().tail(180)
    best_t = None
    best_corr = -1.0
    for t, s in bench_map.items():
        ret_b = s.reindex_like(ret_s, method="ffill").pct_change().dropna()
        paired = pd.concat([ret_s, ret_b], axis=1).dropna()
        if len(paired) < 30:
            continue
        corr = paired.corr().iloc[0,1]
        if pd.notna(corr) and corr > best_corr:
            best_corr = float(corr)
            best_t = t
    return best_t or BENCH_DEFAULT

def compute_confirm_metrics(feats: pd.DataFrame, bench_close_or_map, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    feats: MultiIndex (date,ticker) с колонками close, high, low, volume, adv20
    bench_close_or_map: Series (глобальный бенч) или dict[ticker->Series] (SMART)
    """
    last_date = feats.index.get_level_values(0).max() if today is None else today
    out_rows = []

    def _get_bench_for_t(ticker: str):
        if isinstance(bench_close_or_map, dict):
            return bench_close_or_map.get(ticker)
        return bench_close_or_map

    for t in feats.index.get_level_values(1).unique():
        df = feats.xs(t, level=1, drop_level=False).reset_index(level=1, drop=True)
        df = df.loc[df.index<=last_date]
        if len(df) < 260:
            continue
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

        # Cum returns
        ret = c.pct_change()
        ret21 = (ret+1).rolling(21).apply(np.prod, raw=True).iloc[-1] - 1.0
        ret63 = (ret+1).rolling(63).apply(np.prod, raw=True).iloc[-1] - 1.0

        bench_s = _get_bench_for_t(t)
        if bench_s is None or bench_s.empty:
            alpha21, alpha63 = ret21, ret63
            bench_t = None
        else:
            b21 = (bench_s.pct_change()+1).rolling(21).apply(np.prod, raw=True).iloc[-1] - 1.0
            b63 = (bench_s.pct_change()+1).rolling(63).apply(np.prod, raw=True).iloc[-1] - 1.0
            alpha21 = ret21 - (b21 if np.isfinite(b21) else 0.0)
            alpha63 = ret63 - (b63 if np.isfinite(b63) else 0.0)
            bench_t = getattr(bench_s, "name", None)

        # MAs
        sma20 = c.rolling(20).mean().iloc[-1]
        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        lastc = c.iloc[-1]
        ma_ok = (lastc > sma50 > sma200) and (sma20 > sma50)

        # ADX14
        adx14 = _adx(h, l, c, 14).iloc[-1]

        # RS (price/bench) линрег-наклон и R^2
        _b = bench_s
        rs = (c / _b.reindex_like(c, method="ffill")).dropna() if _b is not None else (c / c).dropna()
        slope_rs, r2_rs = _linreg_slope_r2(rs.tail(60))
        rs_ok = (slope_rs > 0) and (r2_rs >= 0.2)

        # 52w proximity
        hhv252 = c.rolling(252).max().iloc[-1]
        dist52w = lastc/hhv252 - 1.0 if hhv252 and np.isfinite(hhv252) else np.nan
        prox_ok = (-0.02 <= dist52w <= 0.05)

        # Volume confirmation (Up/Down + breakout w/expansion)
        ud_ratio = _up_down_vol_ratio(c, v, 20)
        recent = df.tail(5)
        recent20h = recent["close"] >= df["close"].rolling(20).max().reindex(recent.index)
        vol_exp = recent["volume"] >= 1.5 * (df["close"]*df["volume"]).rolling(20).mean().reindex(recent.index)
        breakout_ok = bool((recent20h & vol_exp).any())
        vol_ok = (ud_ratio >= 1.2) and breakout_ok

        # Volatility compression & DD
        hv20 = _hv(c, 20); hv60 = _hv(c, 60)
        hv_ok = (np.isfinite(hv20) and np.isfinite(hv60) and hv60 > 0 and (hv20/hv60) <= 0.8)
        dd63 = _max_drawdown(c, 63)
        dd_ok = (dd63 >= -0.15) if np.isfinite(dd63) else False

        # Distance to SMA20
        dist_sma20 = (lastc/sma20 - 1.0) if sma20 else np.nan
        dist_ok = (0.0 <= dist_sma20 <= 0.08) if np.isfinite(dist_sma20) else False

        score = 0
        score += 2 if ma_ok else 0
        score += 2 if (adx14 is not None and adx14 >= 20) or rs_ok else 0
        score += 2 if rs_ok else 0
        score += 1 if prox_ok else 0
        score += 2 if vol_ok else 0
        score += 1 if hv_ok else 0
        score += 1 if dd_ok else 0
        score += 1 if dist_ok else 0

        out_rows.append({
            "ticker": t,
            "alpha21": float(alpha21) if np.isfinite(alpha21) else np.nan,
            "alpha63": float(alpha63) if np.isfinite(alpha63) else np.nan,
            "ma_ok": bool(ma_ok),
            "adx14": float(adx14) if np.isfinite(adx14) else np.nan,
            "rs_slope": float(slope_rs) if np.isfinite(slope_rs) else np.nan,
            "rs_r2": float(r2_rs) if np.isfinite(r2_rs) else np.nan,
            "ud_vol": float(ud_ratio) if np.isfinite(ud_ratio) else np.nan,
            "dist52w": float(dist52w) if np.isfinite(dist52w) else np.nan,
            "hv20_over_60": float(hv20/hv60) if (np.isfinite(hv20) and np.isfinite(hv60) and hv60!=0) else np.nan,
            "dd63": float(dd63) if np.isfinite(dd63) else np.nan,
            "dist_sma20": float(dist_sma20) if np.isfinite(dist_sma20) else np.nan,
            "breakout_ok": bool(breakout_ok),
            "confirm": int(score),
            "bench": bench_t,
        })
    return pd.DataFrame(out_rows)

# ────────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ────────────────────────────────────────────────────────────────────────────────
def format_simple_list(df: pd.DataFrame, kind: str, title: str, topn: int = 10) -> str:
    df = df.head(topn).reset_index(drop=True)
    lines = [f"{title}, сортировка по score:"]
    for i, row in enumerate(df.itertuples(index=False), 1):
        lines.append(f"{i}) {row.ticker} [{kind}] | score={row.score:0.4f} | 21д={row.ret21*100:0.2f}% | vol={row.volRel:0.4f}")
    return "```\n" + "\n".join(lines) + "\n```"

# ────────────────────────────────────────────────────────────────────────────────
# Commands
# ────────────────────────────────────────────────────────────────────────────────
HELP_TEXT = (
    "Я на связи ✅\n\n"
    "Команды:\n"
    "/ping — проверить, что живой\n"
    "/sp5005 — топ 5 из SP250 (RS21/63>0, ConfirmScore)\n"
    "/sp50010 — топ 10 из SP250 (RS21/63>0, ConfirmScore)\n"
    "/etfall — топ-10 из ETFALL\n"
    "/etfx — топ-10 из ETFX\n"
    "/etf [etfall|core2x3|etfx] — полный рейтинг ETF по score\n"
    "\nНастройки:\n"
    "/benchmode <global|smart> — режим бенчмарка\n"
    "/setbenchmark <TICKER> — глобальный бенч (для global)\n"
    "/benchcandidates — кандидаты SMART\n"
    "/getbench — показать текущие настройки бенча\n"
    "/setaisource <csv|none> — источник AI скор (ticker,ai_score)\n"
    "/setw_ai <0..1> — вес AI в композите\n"
)

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

# /sp500 с умным отбором (RS-фильтры + ConfirmScore)
async def sp500_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        topn = int(context.args[0]) if context.args else 10
        topn = max(1, min(50, topn))
    except Exception:
        topn = 10

    feats = fetch_features(SP250)
    ai_map = load_ai_scores_csv(AI_CSV_URL) if (AI_CSV_URL and AI_CSV_URL.lower() != "none") else None
    snap = compute_score_snap(feats, ai_map=ai_map, w_ai=W_AI)

    # Bench logic
    if BENCH_MODE == "smart":
        bench_series = fetch_benchmark_series(BENCH_CANDIDATES, period="220d")
        per_t_bench = {}
        for t in feats.index.get_level_values(1).unique():
            s_close = feats.xs(t, level=1)["close"]
            chosen = pick_best_benchmark(s_close, bench_series)
            per_t_bench[t] = bench_series.get(chosen)
            if per_t_bench[t] is not None:
                per_t_bench[t].name = chosen
        bench_obj = per_t_bench
        bench_label = "SMART"
    else:
        b = fetch_benchmark_series([BENCH_DEFAULT], period="220d")[BENCH_DEFAULT]
        bench_obj = b
        bench_label = BENCH_DEFAULT

    # Confirm & RS
    confirm_df = compute_confirm_metrics(feats, bench_obj)
    merged = snap.merge(confirm_df, on="ticker", how="left")
    merged = merged[(merged["alpha21"] > 0) & (merged["alpha63"] > 0)]
    if merged.empty:
        await update.message.reply_text("Нет бумаг, обогнавших выбранный бенчмарк за 21/63д.")
        return

    candidates = merged.sort_values("score", ascending=False)
    strict = candidates[candidates["confirm"] >= 7]
    selected = strict.head(topn)
    if len(selected) < topn:
        rest = candidates[~candidates["ticker"].isin(selected["ticker"])].sort_values(["confirm","score"], ascending=[False, False])
        selected = pd.concat([selected, rest.head(topn-len(selected))], axis=0)

    selected = selected.sort_values("score", ascending=False).reset_index(drop=True)

    lines = [f"Топ {topn} (SP250). Бенчмарк: {bench_label}. AI W={W_AI:0.2f}. Условия: RS21/63>0, Confirm≥7 (с добором)."]
    for i, row in enumerate(selected.itertuples(index=False), 1):
        rs_flag = "✓" if (row.alpha21>0 and row.alpha63>0) else "·"
        ma_flag = "✓" if row.ma_ok else "·"
        vol_flag = "✓" if (row.ud_vol and row.ud_vol>=1.2 and row.breakout_ok) else "·"
        hv_flag = "✓" if (isinstance(row.hv20_over_60, float) and row.hv20_over_60<=0.8) else "·"
        dd_flag = "✓" if (row.dd63 is not None and row.dd63>=-0.15) else "·"
        bench_t = row.bench if isinstance(row.bench, str) and len(row.bench or "")>0 else bench_label
        lines.append(
            f"{i}) {row.ticker} [Акции] | b={bench_t} | "
            f"score={row.score:0.4f} | RS21={row.alpha21*100:0.1f}% RS63={row.alpha63*100:0.1f}% | "
            f"ADX={row.adx14:0.1f} | U/D={row.ud_vol:0.2f} | 52w={row.dist52w*100:0.1f}% | "
            f"CONF={int(row.confirm)}/12  [{rs_flag} RS {ma_flag} MA {vol_flag} VOL {hv_flag} HV {dd_flag} DD]"
        )
    text = "```\n" + "\n".join(lines) + "\n```"
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def sp5005_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.args = ["5"]
    await sp500_cmd(update, context)

async def sp50010_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.args = ["10"]
    await sp500_cmd(update, context)

# ETF рейтинги
async def etfall_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    feats = fetch_features(ETFALL)
    ai_map = load_ai_scores_csv(AI_CSV_URL) if (AI_CSV_URL and AI_CSV_URL.lower() != "none") else None
    snap = compute_score_snap(feats, ai_map=ai_map, w_ai=W_AI).sort_values("score", ascending=False)
    text = format_simple_list(snap, "ETF", "ETFALL топ 10", 10)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def etfx_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    feats = fetch_features(ETFX)
    ai_map = load_ai_scores_csv(AI_CSV_URL) if (AI_CSV_URL and AI_CSV_URL.lower() != "none") else None
    snap = compute_score_snap(feats, ai_map=ai_map, w_ai=W_AI).sort_values("score", ascending=False)
    text = format_simple_list(snap, "ETF", "ETFX топ 10", 10)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

# Универсальный /etf [etfall|core2x3|etfx]
async def etf_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = (context.args[0].lower() if context.args else "etfall")
    if mode in ("etfx","leveraged","lev25","lev"):
        tickers = ETFX
        label = "etfx"
    elif mode == "core2x3":
        tickers = ETF_CORE2x3
        label = "core2x3"
    elif mode in ("etfall","all"):
        tickers = ETFALL
        label = "etfall"
    else:
        tickers = ETFALL
        label = "etfall"
    feats = fetch_features(tickers)
    ai_map = load_ai_scores_csv(AI_CSV_URL) if (AI_CSV_URL and AI_CSV_URL.lower() != "none") else None
    snap = compute_score_snap(feats, ai_map=ai_map, w_ai=W_AI).sort_values("score", ascending=False)
    text = format_simple_list(snap, "ETF", f"ETF рейтинг ({label})", min(20, len(snap)))
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

# Настройки AI
async def setaisource_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_CSV_URL
    if not context.args:
        await update.message.reply_text("Формат: /setaisource <csv_url|none>")
        return
    AI_CSV_URL = context.args[0].strip()
    await update.message.reply_text(f"AI source: {AI_CSV_URL}")

async def setw_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global W_AI
    if not context.args:
        await update.message.reply_text("Формат: /setw_ai <0..1>")
        return
    try:
        W_AI = max(0.0, min(1.0, float(context.args[0])))
    except Exception:
        await update.message.reply_text("Не удалось распарсить число.")
        return
    await update.message.reply_text(f"W_AI = {W_AI}")

# Настройки бенчмарков
async def setbenchmark_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BENCH_DEFAULT
    if not context.args:
        await update.message.reply_text("Формат: /setbenchmark <тикер>")
        return
    BENCH_DEFAULT = context.args[0].strip().upper()
    await update.message.reply_text(f"Бенчмарк по умолчанию: {BENCH_DEFAULT}")

async def benchmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BENCH_MODE
    if not context.args:
        await update.message.reply_text("Формат: /benchmode <global|smart>")
        return
    mode = context.args[0].strip().lower()
    if mode not in ("global","smart"):
        await update.message.reply_text("Поддерживаются: global, smart")
        return
    BENCH_MODE = mode
    await update.message.reply_text(f"Режим бенчмарка: {BENCH_MODE}")

async def benchcandidates_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Кандидаты SMART:\n" + " ".join(BENCH_CANDIDATES))

async def getbench_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Текущие настройки:\nMODE={BENCH_MODE}\nDEFAULT={BENCH_DEFAULT}\nSMART candidates: {' '.join(BENCH_CANDIDATES)}")

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_TOKEN env var.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("sp500", sp500_cmd))
    app.add_handler(CommandHandler("sp5005", sp5005_cmd))
    app.add_handler(CommandHandler("sp50010", sp50010_cmd))
    app.add_handler(CommandHandler("etfall", etfall_cmd))
    app.add_handler(CommandHandler("etfx", etfx_cmd))
    app.add_handler(CommandHandler("etf", etf_cmd))

    app.add_handler(CommandHandler("setaisource", setaisource_cmd))
    app.add_handler(CommandHandler("setw_ai", setw_ai_cmd))

    app.add_handler(CommandHandler("setbenchmark", setbenchmark_cmd))
    app.add_handler(CommandHandler("benchmode", benchmode_cmd))
    app.add_handler(CommandHandler("benchcandidates", benchcandidates_cmd))
    app.add_handler(CommandHandler("getbench", getbench_cmd))

    app.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None)

if __name__ == "__main__":
    main()

