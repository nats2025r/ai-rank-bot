import os
import time
import asyncio
import logging
import yfinance as yf
import pandas as pd
import numpy as np

from flask import Flask, request
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ================== ЛОГИ ==================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ai-rank-bot")

# ================== ENV ===================
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
URL   = os.getenv("WEBHOOK_URL")  # напр.: https://ai-rank-bot-1.onrender.com
if not TOKEN:
    raise RuntimeError("Не найден BOT_TOKEN")
if not URL:
    raise RuntimeError("Не найден WEBHOOK_URL")

# ================== APPs ==================
app = Flask(__name__)
tg_app = Application.builder().token(TOKEN).build()

# ================== КОНСТ =================
DAYS   = 21
PERIOD = "6mo"
BENCH  = "SPY"

# ---------- первые ~250 тикеров S&P 500 (жёстко зашито) ----------
SPX250 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO","JPM","V","UNH","TSLA",
    "XOM","WMT","MA","JNJ","PG","HD","COST","MRK","ADBE","PEP","ABBV","NFLX","CRM","KO","CSCO",
    "ACN","ORCL","TMO","AMD","MCD","WELL","DHR","LIN","INTU","CMCSA","WFC","TXN","AMAT","MS",
    "COP","NEE","INTC","VRTX","PFE","PM","GE","LOW","CVX","BX","HON","NOW","IBM","CAT","AMGN",
    "SCHW","LMT","BKNG","SPGI","ELV","UPS","QCOM","SBUX","PLD","UNP","DE","GS","RTX","MDT",
    "BLK","AXP","ISRG","ADI","TJX","GILD","MO","MMC","CI","SYK","CB","UBER","ZTS","REGN",
    "PYPL","C","ETN","T","KLAC","FI","SO","MRVL","BDX","DUK","PGR","PNC","EQIX","EOG",
    "ABNB","ICE","PH","ITW","SHW","CSX","ADP","EL","FDX","APD","AON","HCA","NKE","SLB","BK",
    "WM","MPC","EMR","GM","CHTR","TT","MAR","AEP","KMI","MMM","ORLY","ROP","CL","MCO",
    "HUM","SPG","DAL","PXD","HAL","LRCX","MNST","D","KR","CCI","KHC","LULU","TRV","AIG","CDNS",
    "CEG","DD","GIS","PSX","MSI","SRE","CTAS","EW","STZ","PCAR","OXY","CMG","NOC","ADM","ED",
    "A","PRU","KMB","TEL","USB","DHI","ROST","IDXX","AZO","PAYX","TGT","BIIB","ECL","ALL",
    "HLT","F","VLO","WBA","AFL","KEYS","HPQ","YUM","ODFL","EXC","AMP","MTB","DLR",
    "ALB","KDP","DTE","PHM","VRSK","FTNT","ES","LEN","HSY","HES","CTVA","NEM","GWW","RSG",
    "CDW","XEL","WTW","OKE","IQV","AEE","SYY","EA","ZBH","PAYC","WEC","CE","GLW","MSCI","EIX",
    "ROK","SWK","DOV","WMB","DVN","NUE","RMD","FANG","VICI","CMS","CNC","PPL","PEAK",
    "IRM","ACGL","WAT","ILMN","FTV","BAX","CARR","HIG","VTR","OTIS","HBAN","ON","RF","HPE",
    "COF","MTD","WDC","URI","MLM","FERG","TSCO","CFG","MCHP","CTSH","EXR","BALL","BBY","CF",
    "EBAY","PTC","DOW","DG","DRI","ETSY","KMX","ALGN","LYB","MKC","CAG","PKI","ZBRA","AAL",
    "NRG","GNRC","LVS","IFF","BRO","ENPH","PWR","STT","TRGP","APA","NDAQ","NTRS","MCK","LKQ",
    "WRB","EPAM","FICO","STE","TDG","HUBB","CPRT","ANET","PCG","FAST","BR","CBOE","HWM","TTWO",
    "EXPE","MTCH","MOS","AES","SWKS","CHD","BBWI","TROW","NVR","ESS","REG","ULTA","TSN","HST",
    "VMC","CME","FSLR","NTAP","KEY","ETR","BIO","RJF","JBHT","AVB","INVH","SJM",
    "NWL","PARA","HAS","SEE","DGX","HSIC","JKHY","LHX","RCL","CCL","AAP","AOS","AKAM","GL",
    "INCY","WRK","GEN","CPB","BKR","BXP"
]

# ---------- ETF наборы ----------
ETF_CORE = ["SPY","QQQ","DIA","IWM","ARKK","XLF","XLK","XLE"]
ETF_X2X3 = ["TQQQ","SPXL","UPRO","SOXL","SQQQ","SPXS"]
ETF_ALL  = ETF_CORE + ETF_X2X3

# ================== YF HELPERS (надёжные) ==================
def _yf_download_safe(tickers, period=PERIOD, group_by="column", auto_adjust=False):
    """
    Устойчивая обёртка над yfinance.download:
    - 3 попытки
    - threads=False (меньше шанс на ошибки в free-инстансах)
    """
    for i in range(3):
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                group_by=group_by,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                return df
            log.warning("yfinance: empty df for %s (try %d)", tickers, i + 1)
        except Exception as e:
            log.warning("yfinance error for %s (try %d): %s", tickers, i + 1, e)
        time.sleep(1 + i * 2)
    return pd.DataFrame()

def _download_batch(tickers: list[str]) -> pd.DataFrame:
    data = _yf_download_safe(tickers=tickers, period=PERIOD, group_by="column", auto_adjust=False)
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        return data
    # если один тикер — нормализуем к мультииндексу
    data.columns = pd.MultiIndex.from_product([data.columns, [tickers[0]]])
    return data

def _enough(close: pd.Series, days: int) -> bool:
    return close.dropna().size >= days + 1

def _calc_scores_for_list(tickers: list[str], benchmark: str = BENCH, days: int = DAYS) -> dict[str, float]:
    data  = _download_batch(tickers)
    bench = _yf_download_safe(benchmark, period=PERIOD)
    if data.empty or bench.empty:
        log.warning("Пустые данные: data.empty=%s bench.empty=%s", data.empty, bench.empty)
        return {}

    bench_close = bench["Close"].dropna()
    if not _enough(bench_close, days):
        log.warning("Недостаточно истории для %s", benchmark)
        return {}
    ret_bench = bench_close.iloc[-1] / bench_close.iloc[-days] - 1

    out: dict[str, float] = {}
    for t in tickers:
        try:
            if ("Close", t) not in data.columns or ("Volume", t) not in data.columns:
                continue
            close = data[("Close", t)].dropna()
            volm  = data[("Volume", t)].dropna()
            if not _enough(close, days) or volm.tail(days * 3).dropna().empty:
                continue

            ret = close.iloc[-1] / close.iloc[-days] - 1
            ret_rel = ret - ret_bench
            vol = close.pct_change().dropna().std()
            if not np.isfinite(vol) or vol == 0:
                continue

            avg21 = float(volm.tail(days).mean())
            avg60 = float(volm.tail(days * 3).mean())
            vol_confirm = (avg21 / avg60) if (avg60 and avg60 > 0) else 1.0

            score = (ret_rel / vol) * vol_confirm
            if np.isfinite(score):
                out[t] = float(score)
        except Exception as e:
            log.exception("Ошибка %s: %s", t, e)
    return out

def _parse_n(args, default=10, minv=3, maxv=50) -> int:
    try:
        n = int(args[0]) if args else default
    except:
        n = default
    return max(minv, min(maxv, n))

# ================== КОМАНДЫ =================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я AI-рейтинг акций.\n\n"
        "/wake — разбудить/прогреть\n"
        "/ping — проверить связь\n"
        "/diag — диагностика\n"
        "/sp500 [N] — топ N из первых 250 тикеров\n"
        "/sp5005 | /sp50010 — быстрые пресеты\n"
        "/etf core|all|x2x3 [N] — рейтинг ETF\n"
        "/mix [N] — SPX250 + ETF"
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅")

async def cmd_wake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        _ = _yf_download_safe(BENCH, period="5d")
        await update.message.reply_text("Готов к работе ⚡️")
    except Exception as e:
        await update.message.reply_text(f"Wake error: {e}")

async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        info = await tg_app.bot.get_webhook_info()
        wh = info.url or "<none>"
        wh_state = f"ok (pending: {info.pending_update_count})" if info.url else "not set"
    except Exception as e:
        wh = f"error: {e}"
        wh_state = "error"

    try:
        df = _yf_download_safe("SPY", period="5d")
        yf_msg = "SPY: пусто" if df.empty else f"SPY ok, last close={float(df['Close'].dropna().iloc[-1]):.2f}"
    except Exception as e:
        yf_msg = f"yfinance exception: {type(e).__name__}: {e}"

    text = (
        "<b>DIAG</b>\n"
        f"- webhook_url (env): {os.getenv('WEBHOOK_URL')}\n"
        f"- getWebhookInfo: {wh} [{wh_state}]\n"
        f"- yfinance: {yf_msg}\n"
        "versions: ptb 20.3, flask 2.3.2, yf 0.2.26\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

async def _rank_and_reply(update: Update, tickers: list[str], n: int, label: str):
    await update.message.reply_text(f"Считаю рейтинг {label}… (N={n}, D={DAYS})")
    scores = await asyncio.to_thread(_calc_scores_for_list, tickers)
    if not scores:
        await update.message.reply_text(f"Не удалось получить данные {BENCH}.")
        return
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    lines = [f"📊 Топ-{n} {label} по score:"]
    for t, s in top:
        lines.append(f"{t}: {s:.2f}")
    await update.message.reply_text("\n".join(lines))

# — S&P 500
async def cmd_sp500(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = _parse_n(context.args, default=10)
    await _rank_and_reply(update, SPX250, n, "S&P 500 (первые 250)")

async def cmd_sp5005(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _rank_and_reply(update, SPX250, 5, "S&P 500 (первые 250)")

async def cmd_sp50010(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _rank_and_reply(update, SPX250, 10, "S&P 500 (первые 250)")

# — ETF
async def cmd_etf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Используй: /etf core|all|x2x3 [N]")
        return
    mode = context.args[0].lower()
    n = _parse_n(context.args[1:], default=10)

    if mode == "core":
        await _rank_and_reply(update, ETF_CORE, n, "ETF Core")
    elif mode == "all":
        await _rank_and_reply(update, ETF_ALL, n, "ETF All")
    elif mode in ("x2x3", "leveraged"):
        await _rank_and_reply(update, ETF_X2X3, n, "ETF x2/x3")
    else:
        await update.message.reply_text("Неизвестный режим. Используй: core | all | x2x3")

# — MIX
async def cmd_mix(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = _parse_n(context.args, default=10)
    combo = SPX250 + ETF_ALL
    await _rank_and_reply(update, combo, n, "Mix (SPX250 + ETF)")

# ================== РЕГИСТРАЦИЯ КОМАНД =================
tg_app.add_handler(CommandHandler("start",   cmd_start))
tg_app.add_handler(CommandHandler("ping",    cmd_ping))
tg_app.add_handler(CommandHandler("wake",    cmd_wake))
tg_app.add_handler(CommandHandler("diag",    cmd_diag))
tg_app.add_handler(CommandHandler("sp500",   cmd_sp500))
tg_app.add_handler(CommandHandler("sp5005",  cmd_sp5005))
tg_app.add_handler(CommandHandler("sp50010", cmd_sp50010))
tg_app.add_handler(CommandHandler("etf",     cmd_etf))
tg_app.add_handler(CommandHandler("mix",     cmd_mix))

# ================== FLASK ROUTES =================
@app.route("/", methods=["GET"])
def root():
    return "ok", 200

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/wake-http", methods=["GET"])
def wake_http():
    try:
        _ = _yf_download_safe(BENCH, period="5d")
        return "warmed", 200
    except Exception as e:
        return f"wake error: {e}", 500

@app.route(f"/webhook/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), tg_app.bot)
    tg_app.update_queue.put_nowait(update)
    return "ok", 200

# ================== ENTRYPOINT (waitress) =================
if __name__ == "__main__":
    from waitress import serve

    async def main():
        webhook_url = f"{URL}/webhook/{TOKEN}"

        # 1) Telegram bot
        await tg_app.initialize()
        await tg_app.start()

        # 2) Webhook
        await tg_app.bot.delete_webhook(drop_pending_updates=True)
        await tg_app.bot.set_webhook(webhook_url, allowed_updates=["message"])
        log.info("Webhook установлен: %s", webhook_url)

        # 3) HTTP server (sync) in executor
        port = int(os.environ.get("PORT", 8080))
        loop = asyncio.get_running_loop()
        log.info("Starting HTTP on 0.0.0.0:%d (waitress)...", port)
        await loop.run_in_executor(None, lambda: serve(app, host="0.0.0.0", port=port))

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
