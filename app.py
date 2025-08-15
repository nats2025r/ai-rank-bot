import logging
import pandas as pd
import yfinance as yf

async def strong(update: Update, context: ContextTypes.DEFAULT_TYPE):
    results = []
    for t in TICKERS:
        try:
            df = yf.download(
                t, period="6mo", interval="1d",
                auto_adjust=False, progress=False, threads=False
            )
            # нормализуем в Series
            if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
                s = df["Adj Close"].dropna()
            elif isinstance(df, pd.DataFrame) and "Close" in df.columns:
                s = df["Close"].dropna()
            else:
                logging.warning("No usable data for %s", t)
                continue
            if s.empty or len(s) < 40:
                continue
        except Exception as e:
            logging.warning("Download failed for %s: %s", t, e)
            continue

        # метрики
        ret30 = float(s.pct_change(30).iloc[-1])
        vol = float(s.pct_change().dropna().std())
        score = ret30 / vol if vol else 0.0
        results.append((t, ret30, vol, score))

    if not results:
        await update.message.reply_text("Сейчас нет данных (Yahoo тупит). Попробуйте ещё раз чуть позже.")
        return

    results.sort(key=lambda x: x[3], reverse=True)
    top = results[:10]
    msg = "📊 Сильные идеи (по AI-скор):\n" + "\n".join(
        f"{i+1:>2}) {t:<6} | 30д={r:.2%} | vol={v:.4f} | score={s:.4f}"
        for i, (t, r, v, s) in enumerate(top)
    )
    await update.message.reply_text(msg)

