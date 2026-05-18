import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
# -----------------------------
# GLOBAL COLOR ENGINE
# -----------------------------
def get_bias_color(bias):
    if "Bullish" in bias:
        return "#0f2e1f"  # dark green
    elif "Bearish" in bias:
        return "#3a0f14"  # dark red
    else:
        return "#3a320f"  # muted yellow
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from streamlit_autorefresh import st_autorefresh
st.set_page_config(page_title="Market State Intelligence System", layout="wide")
st.markdown("""
<style>
/* APP BACKGROUND */
.stApp {
    background:#f4f6fb;
}
.js-plotly-plot {
    touch-action: none;
    overscroll-behavior: contain;
}


/* MAIN CONTAINER */
.block-container {
    padding-top:1rem !important;
    padding-bottom:1rem !important;
    padding-left:1rem !important;
    padding-right:1rem !important;
    max-width:100% !important;
}
.app-title {
    font-size:40px !important;
    font-weight:900 !important;
    color:#0b1f3a !important;
    line-height:1.05 !important;
    margin-top:10px !important;
}

.app-caption {
    font-size:20px !important;
    color:#334155 !important;
    margin-top:8px !important;
    margin-bottom:16px !important;
}
/* SIDEBAR */
section[data-testid="stSidebar"] {
    background:#eef2f8;
    border-right:1px solid #d6dce8;
}

/* HEADERS */
h1 {
    font-size:18px !important;
    color:#1e3a5f; !important;
}
h2 {
    font-size:18px !important;
    color:#1e3a5f; !important;
}
h3 {
    background:#0b1f3a;
    color:white !important;
    padding:5px 9px;
    border-radius:7px;
    font-size:16px !important;
    margin-top:8px;
    margin-bottom:6px;
}

/* TEXT */
p, span, label {
 font-size:16px !important;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background:#ffffff;
    border:1px solid #d9dee8;
    border-radius:8px;
    padding:6px 8px;
    box-shadow:0 2px 8px rgba(0,0,0,0.05);
}

[data-testid="stMetricLabel"] {
    font-size:12px !important;
    color:#475569 !important;
}

[data-testid="stMetricValue"] {
    font-size:18px !important;
    color:#0f172a !important;
}

/* ALERT BLOCKS */
.stAlert {
    border-radius:8px;
    border:1px solid #d9dee8;
}

/* INPUTS */
input, textarea, button {
    font-size:13px !important;
}

/* SELECTBOX */
div[data-baseweb="select"] > div {
    border-color:#94a3b8 !important;
    box-shadow:none !important;
    border-radius:10px !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius:14px;
    overflow:hidden;
}

/* CHART AREA */
.js-plotly-plot {
    border-radius:14px;
    overflow:hidden;
    box-shadow:0 4px 14px rgba(0,0,0,0.12);
}

/* DIVIDER */
hr {
    border:none;
    border-top:1px solid #d6dce8;
    margin:8px 0;
}
</style>
""", unsafe_allow_html=True)

components.html("""
<script>
const doc = window.parent.document;

function lockScroll(e) {
    const plot = e.target.closest('.js-plotly-plot');
    if (plot) {
        e.preventDefault();
    }
}

doc.addEventListener('wheel', lockScroll, { passive: false });
</script>
""", height=0)
@st.cache_data(ttl=10, show_spinner=False)
def load_data(ticker, period, interval):
    ticker = str(ticker).upper().strip()

    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                prepost=True,   # ✅ THIS IS THE KEY CHANGE
                progress=False,
                threads=False
            )

            if not df.empty:
                df = df.reset_index()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                return df

        except Exception:
            pass

    return pd.DataFrame()

st.markdown("""
<div class="app-title">
Infinity Market's State Intelligence System
</div>

<div class="app-caption">
Market Intelligence + Options + AI + Historical Pattern + Trade Engine
</div>
""", unsafe_allow_html=True)


def add_indicators(df):
    df = df.copy()

    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()

    macd = MACD(df["Close"])
    df["MACD_HIST"] = macd.macd_diff()

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )
    df["ATR"] = atr.average_true_range()

    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["EMA_200"] = df["Close"].ewm(span=200).mean()

    x_col = "Datetime" if "Datetime" in df.columns else "Date"
    df[x_col] = pd.to_datetime(df[x_col])
    df["Session"] = df[x_col].dt.date

    df["VWAP"] = (
        (df["Close"] * df["Volume"]).groupby(df["Session"]).cumsum()
        / df["Volume"].groupby(df["Session"]).cumsum()
    )
    df["VWAP_Dist"] = df["Close"] - df["VWAP"]
    df["VWAP_STD"] = df["VWAP_Dist"].rolling(20).std()

    df["VWAP_Upper_1"] = df["VWAP"] + df["VWAP_STD"]
    df["VWAP_Lower_1"] = df["VWAP"] - df["VWAP_STD"]

    df["VWAP_Upper_2"] = df["VWAP"] + df["VWAP_STD"] * 2
    df["VWAP_Lower_2"] = df["VWAP"] - df["VWAP_STD"] * 2
    df["Volume_MA"] = df["Volume"].rolling(20).mean()
    df["Volume_Spike"] = df["Volume"] > df["Volume_MA"] * 1.5
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Close", "High", "Low", "Open", "Volume"])

    return df


def classify_timeframe(df):
    latest = df.iloc[-1]
    prev = df.iloc[-5]
    close = latest["Close"]

    score = 0

    if close > latest["EMA_20"] > latest["EMA_50"]:
        score += 2
    elif close < latest["EMA_20"] < latest["EMA_50"]:
        score -= 2

    score += 1 if close > latest["EMA_200"] else -1

    if latest["RSI"] > 60:
        score += 1
    elif latest["RSI"] < 40:
        score -= 1

    score += 1 if latest["MACD_HIST"] > prev["MACD_HIST"] else -1

    atr_pct = latest["ATR"] / close

    if atr_pct < 0.015:
        volatility = "Compression"
    elif atr_pct < 0.035:
        volatility = "Expansion"
    else:
        volatility = "Instability"

    if score >= 4:
        state, bias = "Bull Expansion", "Bullish"
    elif score >= 2:
        state, bias = "Bull Pullback / Continuation", "Bullish"
    elif score <= -4:
        state, bias = "Bear Expansion", "Bearish"
    elif score <= -2:
        state, bias = "Bear Bounce / Continuation", "Bearish"
    elif volatility == "Compression":
        state, bias = "Compression", "Neutral"
    else:
        state, bias = "Mixed / Chop", "Neutral"

    return {
        "state": state,
        "bias": bias,
        "score": score,
        "rsi": round(latest["RSI"], 2),
        "volatility": volatility,
        "atr_pct": round(atr_pct * 100, 2)
    }


def multi_timeframe_analysis(ticker):
    configs = {
        "Daily": ("1y", "1d"),
        "4H": ("6mo", "1h"),
        "1H": ("3mo", "1h"),
        "15m": ("1mo", "15m"),
    }

    results = {}

    for tf, (period, interval) in configs.items():
        df = load_data(ticker, period, interval)

        if df.empty or len(df) < 50:
            results[tf] = None
            continue

        df = add_indicators(df)
        results[tf] = classify_timeframe(df)

    return results


def overall_bias(mtf):
    weights = {"Daily": 4, "4H": 3, "1H": 2, "15m": 1}

    bull = 0
    bear = 0
    neutral = 0

    for tf, result in mtf.items():
        if result is None:
            continue

        weight = weights[tf]

        if result["bias"] == "Bullish":
            bull += weight
        elif result["bias"] == "Bearish":
            bear += weight
        else:
            neutral += weight

    direction_total = bull + bear

    if direction_total == 0:
        bull_prob = 50
        bear_prob = 50
    else:
        bull_prob = round((bull / direction_total) * 100)
    bear_prob = 100 - bull_prob

    neutral_prob = 0

    if bull_prob >= 50 and bull_prob > bear_prob:
        final = "Bullish Alignment"
    elif bear_prob >= 50 and bear_prob > bull_prob:
        final = "Bearish Alignment"
    else:
        final = "Mixed / Wait"

    return final, bull_prob, bear_prob, neutral_prob


def calculate_fibs(df, lookback=120):
    recent = df.tail(lookback)
    high = recent["High"].max()
    low = recent["Low"].min()
    diff = high - low

    return {
        "38.2 Retracement": high - diff * 0.382,
        "50.0 Retracement": high - diff * 0.5,
        "61.8 Retracement": high - diff * 0.618,
        "127.2 Extension": high + diff * 0.272,
        "161.8 Extension": high + diff * 0.618,
        "222.0 Extension": high + diff * 1.22,
    }


def volume_profile(df, bins=24):
    hist, edges = np.histogram(df["Close"], bins=bins, weights=df["Volume"])

    profile = pd.DataFrame({
        "price": (edges[:-1] + edges[1:]) / 2,
        "volume": hist
    }).sort_values("volume", ascending=False)

    return (
        profile.iloc[0]["price"],
        profile.head(5)["price"].tolist(),
        profile.tail(5)["price"].tolist()
    )


def liquidity_levels(df):
    recent = df.tail(80)

    return {
        "Prior High Liquidity": recent["High"].max(),
        "Prior Low Liquidity": recent["Low"].min(),
        "VWAP Magnet": df.iloc[-1]["VWAP"],
        "Current Price": df.iloc[-1]["Close"],
    }
def intraday_levels(df):
    x_col = "Datetime" if "Datetime" in df.columns else "Date"
    temp = df.copy()
    temp[x_col] = pd.to_datetime(temp[x_col])
    temp["date"] = temp[x_col].dt.date
    today = temp["date"].iloc[-1]

    today_df = temp[temp["date"] == today]
    prev_df = temp[temp["date"] < today]

    premarket = today_df[
        (today_df[x_col].dt.time >= pd.to_datetime("04:00").time()) &
        (today_df[x_col].dt.time < pd.to_datetime("09:30").time())
    ]

    opening = today_df[
        (today_df[x_col].dt.time >= pd.to_datetime("09:30").time()) &
        (today_df[x_col].dt.time <= pd.to_datetime("10:00").time())
    ]

    return {
        "premarket_high": premarket["High"].max() if not premarket.empty else None,
        "premarket_low": premarket["Low"].min() if not premarket.empty else None,
        "overnight_high": prev_df["High"].tail(80).max() if not prev_df.empty else None,
        "overnight_low": prev_df["Low"].tail(80).min() if not prev_df.empty else None,
        "opening_range_high": opening["High"].max() if not opening.empty else None,
        "opening_range_low": opening["Low"].min() if not opening.empty else None,
    }


def expected_move_engine(df, options_data=None, ticker=None):
    price = df.iloc[-1]["Close"]
    atr = df.iloc[-1]["ATR"]

    move = atr
    source = "ATR fallback"

    if ticker:
        try:
            stock = yf.Ticker(ticker)
            expiration = stock.options[0]
            chain = stock.option_chain(expiration)

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            calls["distance"] = abs(calls["strike"] - price)
            puts["distance"] = abs(puts["strike"] - price)

            atm_call = calls.loc[calls["distance"].idxmin()]
            atm_put = puts.loc[puts["distance"].idxmin()]

            call_price = (atm_call["bid"] + atm_call["ask"]) / 2
            put_price = (atm_put["bid"] + atm_put["ask"]) / 2

            straddle = call_price + put_price

            if straddle > 0:
                move = straddle
                source = "ATM Straddle"

        except Exception:
            pass

    return {
        "expected_low": round(price - move, 2),
        "expected_high": round(price + move, 2),
        "range_size": round(move * 2, 2),
        "source": source
    }


def market_maker_engine(df, levels, options_data, ticker):
    intraday = intraday_levels(df)
    expected = expected_move_engine(df, options_data, ticker)

    upside_candidates = [
        levels["Prior High Liquidity"],
        intraday.get("premarket_high"),
        intraday.get("overnight_high"),
        intraday.get("opening_range_high"),
        options_data["call_wall"] if options_data else None,
    ]

    downside_candidates = [
        levels["Prior Low Liquidity"],
        intraday.get("premarket_low"),
        intraday.get("overnight_low"),
        intraday.get("opening_range_low"),
        options_data["put_wall"] if options_data else None,
    ]

    upside = [x for x in upside_candidates if x is not None]
    downside = [x for x in downside_candidates if x is not None]

    return {
        "expected": expected,
        "intraday": intraday,
        "upper_magnet": round(max(upside), 2) if upside else None,
        "lower_magnet": round(min(downside), 2) if downside else None,
    }

def opening_range_break_engine(df, intraday):
    price = df.iloc[-1]["Close"]
    volume = df.iloc[-1]["Volume"]
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1]

    orb_high = intraday.get("opening_range_high")
    orb_low = intraday.get("opening_range_low")

    if orb_high is None or orb_low is None:
        return {"trigger": "NO DATA", "strength": "N/A"}

    # breakout logic
    if price > orb_high:
        trigger = "ORB BREAKOUT"
    elif price < orb_low:
        trigger = "ORB BREAKDOWN"
    else:
        trigger = "INSIDE RANGE"

    # volume confirmation
    if volume > avg_vol * 1.5:
        strength = "HIGH CONFIRMATION"
    elif volume > avg_vol:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    return {
    "trigger": trigger,
    "strength": strength,
    "orb_high": orb_high,
    "orb_low": orb_low
}


def options_positioning(ticker, current_price):
    stock = yf.Ticker(ticker)

    try:
        expirations = stock.options
    except Exception:
        return None

    if not expirations:
        return None

    expiration = expirations[0]

    try:
        chain = stock.option_chain(expiration)
    except Exception:
        return None

    if chain is None or chain.calls is None or chain.puts is None:
        return None

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    if calls.empty or puts.empty:
        return None

    if calls.empty or puts.empty:
        return None

    calls["type"] = "call"
    puts["type"] = "put"

    options = pd.concat([calls, puts])

    calls_oi = calls["openInterest"].fillna(0).sum()
    puts_oi = puts["openInterest"].fillna(0).sum()

    put_call_ratio = puts_oi / calls_oi if calls_oi > 0 else 0

    call_wall = calls.loc[calls["openInterest"].fillna(0).idxmax()]
    put_wall = puts.loc[puts["openInterest"].fillna(0).idxmax()]

    options["distance"] = abs(options["strike"] - current_price)
    near_money = options[options["distance"] <= current_price * 0.05]

    near_call_oi = near_money[near_money["type"] == "call"]["openInterest"].fillna(0).sum()
    near_put_oi = near_money[near_money["type"] == "put"]["openInterest"].fillna(0).sum()

    if put_call_ratio > 1.25:
        positioning = "Put-Heavy / Potential Squeeze Fuel"
    elif put_call_ratio < 0.75:
        positioning = "Call-Heavy / Long Crowding Risk"
    else:
        positioning = "Balanced Options Positioning"

    if current_price < call_wall["strike"] and near_call_oi > near_put_oi:
        dealer_pressure = "Possible upside magnet toward call wall"
    elif current_price > put_wall["strike"] and near_put_oi > near_call_oi:
        dealer_pressure = "Possible downside magnet toward put wall"
    else:
        dealer_pressure = "No dominant near-money dealer pressure"

    return {
        "expiration": expiration,
        "put_call_ratio": round(put_call_ratio, 2),
        "call_wall": float(call_wall["strike"]),
        "put_wall": float(put_wall["strike"]),
        "call_wall_oi": int(call_wall["openInterest"]),
        "put_wall_oi": int(put_wall["openInterest"]),
        "positioning": positioning,
        "dealer_pressure": dealer_pressure,
    }
def gamma_exposure_engine(ticker, current_price):
    try:
        stock = yf.Ticker(ticker)
        expiration = stock.options[0]
        chain = stock.option_chain(expiration)

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["type"] = "call"
        puts["type"] = "put"

        options = pd.concat([calls, puts])
        options = options.dropna(subset=["strike", "openInterest", "impliedVolatility"])

        options["distance"] = abs(options["strike"] - current_price)
        options = options[options["distance"] <= current_price * 0.10]

        options["gamma_proxy"] = (
            options["openInterest"].fillna(0)
            * options["impliedVolatility"].fillna(0)
            / options["distance"].replace(0, 0.01)
        )

        call_gex = options[options["type"] == "call"]["gamma_proxy"].sum()
        put_gex = options[options["type"] == "put"]["gamma_proxy"].sum()

        net_gex = call_gex - put_gex

        max_gamma_strike = options.loc[options["gamma_proxy"].idxmax()]["strike"]

        if net_gex > 0:
            regime = "Positive Gamma / Pinning"
            pressure = "Price may stay controlled near major strikes"
        elif net_gex < 0:
            regime = "Negative Gamma / Expansion Risk"
            pressure = "Breakouts/breakdowns can accelerate"
        else:
            regime = "Neutral Gamma"
            pressure = "No clear dealer gamma pressure"

        return {
            "net_gex": round(net_gex, 2),
            "call_gex": round(call_gex, 2),
            "put_gex": round(put_gex, 2),
            "max_gamma_strike": round(float(max_gamma_strike), 2),
            "regime": regime,
            "pressure": pressure
        }

    except Exception:
        return None

def delta_flow_engine(ticker, current_price):
    try:
        stock = yf.Ticker(ticker)
        expiration = stock.options[0]
        chain = stock.option_chain(expiration)

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["distance"] = abs(calls["strike"] - current_price)
        puts["distance"] = abs(puts["strike"] - current_price)

        calls = calls[calls["distance"] <= current_price * 0.10]
        puts = puts[puts["distance"] <= current_price * 0.10]

        call_delta_proxy = (
            calls["openInterest"].fillna(0)
            * calls["impliedVolatility"].fillna(0)
            / calls["distance"].replace(0, 0.01)
        ).sum()

        put_delta_proxy = (
            puts["openInterest"].fillna(0)
            * puts["impliedVolatility"].fillna(0)
            / puts["distance"].replace(0, 0.01)
        ).sum()

        net_delta = call_delta_proxy - put_delta_proxy

        if net_delta > 0:
            bias = "Bullish Delta Pressure"
            read = "Dealer hedging may support upside"
        elif net_delta < 0:
            bias = "Bearish Delta Pressure"
            read = "Dealer hedging may support downside"
        else:
            bias = "Neutral Delta"
            read = "No clear delta pressure"

        return {
            "net_delta": round(net_delta, 2),
            "call_delta": round(call_delta_proxy, 2),
            "put_delta": round(put_delta_proxy, 2),
            "bias": bias,
            "read": read
        }

    except Exception:
        return None

def contract_selection_engine(price, mm, probability, panel, final_bias):
    expected_high = mm["expected"]["expected_high"]
    expected_low = mm["expected"]["expected_low"]

    fast_move = (
        panel["orb_trigger"] in ["ORB BREAKOUT", "ORB BREAKDOWN"]
        and panel["orb_strength"] == "HIGH CONFIRMATION"
    )

    bullish = "Bullish" in final_bias or panel["orb_trigger"] == "ORB BREAKOUT"
    bearish = "Bearish" in final_bias or panel["orb_trigger"] == "ORB BREAKDOWN"

    if probability["confidence"] < 60:
        return {
            "contract": "WAIT",
            "strike": "N/A",
            "expiry": "N/A",
            "target": "N/A",
            "note": "Confidence too low. No contract suggested."
        }

    if bullish:
        contract = "CALL"
        target = expected_high
        strike = round(price) if fast_move else round(price - 1)
    elif bearish:
        contract = "PUT"
        target = expected_low
        strike = round(price) if fast_move else round(price + 1)
    else:
        return {
            "contract": "WAIT",
            "strike": "N/A",
            "expiry": "N/A",
            "target": "N/A",
            "note": "No clear direction."
        }

    expiry = "0-1 DTE" if fast_move else "3-7 DTE"
    note = "FAST MOVE → ATM for delta" if fast_move else "SLOW MOVE → slightly ITM with more time"

    return {
        "contract": contract,
        "strike": strike,
        "expiry": expiry,
        "target": target,
        "note": note
    }

def trade_context_engine(levels, delta_flow, orb, final_bias):
    price = levels["Current Price"]
    vwap = levels["VWAP Magnet"]

    delta_bias = delta_flow["bias"] if delta_flow else "Neutral Delta"
    orb_trigger = orb["trigger"]
    orb_strength = orb["strength"]

    if price > vwap:
        control = "Buyers gaining control above VWAP"
    else:
        control = "Sellers in control below VWAP"

    if orb_trigger == "ORB BREAKOUT" and orb_strength == "HIGH CONFIRMATION":
        if "Bearish" in delta_bias:
            context = "⚡ SQUEEZE / COUNTER-TREND LONG"
            note = "Bearish delta, but price is breaking out with volume. Scalp long only."
        else:
            context = "🚀 TREND LONG"
            note = "Bullish breakout confirmed by volume."

    elif orb_trigger == "ORB BREAKDOWN" and orb_strength == "HIGH CONFIRMATION":
        if "Bullish" in delta_bias:
            context = "⚡ LONG TRAP / COUNTER-TREND SHORT"
            note = "Bullish positioning failed. Breakdown has volume."
        else:
            context = "🔴 TREND SHORT"
            note = "Bearish breakdown confirmed by volume."

    elif "Bullish" in final_bias and price > vwap:
        context = "🟢 BULLISH LEAN"
        note = "Bias and VWAP agree, but wait for trigger."

    elif "Bearish" in final_bias and price < vwap:
        context = "🔴 BEARISH LEAN"
        note = "Bias and VWAP agree, but wait for trigger."

    else:
        context = "⏳ MIXED / WAIT"
        note = "Signals are conflicting. Wait for cleaner confirmation."

    return {
        "control": control,
        "context": context,
        "note": note
    }

def ai_market_analysis(state, levels, fibs, options_data, mtf):
    insights = []

    insights.append(f"Current market environment shows **{state}** conditions.")

    price = levels["Current Price"]
    high = levels["Prior High Liquidity"]
    low = levels["Prior Low Liquidity"]
    vwap = levels["VWAP Magnet"]

    if price < high:
        insights.append("Price is below prior highs — liquidity likely sits above.")

    if price > low:
        insights.append("Downside liquidity remains below prior lows.")

    if price > vwap:
        insights.append("Price is above VWAP — short-term buyers have control.")
    else:
        insights.append("Price is below VWAP — short-term sellers have control.")

    fib_618 = fibs.get("61.8 Retracement")
    fib_161 = fibs.get("161.8 Extension")

    if fib_161 and price > fib_161:
        insights.append("Price is extended beyond 161.8 — potential exhaustion or squeeze zone.")

    if fib_618 and abs(price - fib_618) / price < 0.01:
        insights.append("Price is near the 61.8 retracement — key reaction zone.")

    if options_data:
        if options_data["put_call_ratio"] > 1.2:
            insights.append("Put-heavy positioning may create squeeze fuel if price rises.")

        if options_data["put_call_ratio"] < 0.8:
            insights.append("Call-heavy positioning may create downside risk if longs get trapped.")

        insights.append(options_data["dealer_pressure"])

    bullish = sum(1 for tf in mtf.values() if tf and tf["bias"] == "Bullish")
    bearish = sum(1 for tf in mtf.values() if tf and tf["bias"] == "Bearish")

    if bullish > bearish:
        insights.append("Higher timeframes favor upside continuation.")
    elif bearish > bullish:
        insights.append("Higher timeframes favor downside pressure.")
    else:
        insights.append("Timeframes are mixed — lower confidence environment.")

    return insights


def find_similar_setups(df, forward=10):
    results = []

    current = df.iloc[-1]

    for i in range(50, len(df) - forward):
        row = df.iloc[i]
        score = 0

        if abs(row["RSI"] - current["RSI"]) < 5:
            score += 1

        if current["ATR"] != 0 and abs(row["ATR"] - current["ATR"]) / current["ATR"] < 0.2:
            score += 1

        if (row["Close"] > row["EMA_50"]) == (current["Close"] > current["EMA_50"]):
            score += 1

        if (row["MACD_HIST"] > 0) == (current["MACD_HIST"] > 0):
            score += 1

        if score >= 3:
            future_price = df.iloc[i + forward]["Close"]
            setup_price = row["Close"]
            return_pct = (future_price - setup_price) / setup_price * 100
            results.append(return_pct)

    return results


def analyze_forward_returns(returns):
    if not returns:
        return None

    returns = np.array(returns)

    avg = np.mean(returns)
    win_rate = np.sum(returns > 0) / len(returns) * 100

    up_moves = returns[returns > 0]
    down_moves = returns[returns < 0]

    avg_up = np.mean(up_moves) if len(up_moves) > 0 else 0
    avg_down = np.mean(down_moves) if len(down_moves) > 0 else 0

    return {
        "samples": len(returns),
        "avg_return": round(avg, 2),
        "win_rate": round(win_rate, 1),
        "avg_up": round(avg_up, 2),
        "avg_down": round(avg_down, 2),
    }


def trade_setup_engine(final_bias, levels, options_data, pattern_stats):
    price = levels["Current Price"]
    vwap = levels["VWAP Magnet"]

    score = 0
    reasons = []

    if final_bias == "Bullish Alignment":
        score += 2
        reasons.append("Higher timeframes bullish")

    elif final_bias == "Bearish Alignment":
        score -= 2
        reasons.append("Higher timeframes bearish")

    if price > vwap:
        score += 1
        reasons.append("Price above VWAP")
    else:
        score -= 1
        reasons.append("Price below VWAP")

    if options_data:
        if "Put-Heavy" in options_data["positioning"]:
            score += 1
            reasons.append("Put-heavy positioning may fuel squeeze")
        elif "Call-Heavy" in options_data["positioning"]:
            score -= 1
            reasons.append("Call-heavy positioning creates long-crowding risk")

    if pattern_stats:
        if pattern_stats["win_rate"] > 60:
            score += 2
            reasons.append("Historical pattern favors upside")
        elif pattern_stats["win_rate"] < 40:
            score -= 2
            reasons.append("Historical pattern favors downside")

    if score >= 4:
        setup = "High-Quality Long Setup"
    elif score >= 2:
        setup = "Moderate Long Setup"
    elif score <= -4:
        setup = "High-Quality Short Setup"
    elif score <= -2:
        setup = "Moderate Short Setup"
    else:
        setup = "No Trade / Wait"

    return {
        "setup": setup,
        "score": score,
        "reasons": reasons,
    }

def risk_management_engine(trade_setup, levels):
    price = levels["Current Price"]
    prior_high = levels["Prior High Liquidity"]
    prior_low = levels["Prior Low Liquidity"]
    vwap = levels["VWAP Magnet"]

    setup = trade_setup["setup"]

    if "Long" in setup:
        invalidation = min(vwap, prior_low)
        target = prior_high
        risk = price - invalidation
        reward = target - price
    elif "Short" in setup:
        invalidation = max(vwap, prior_high)
        target = prior_low
        risk = invalidation - price
        reward = price - target
    else:
        return {
            "mode": "No Trade",
            "invalidation": None,
            "target": None,
            "rr": None,
            "message": "No clean trade edge. Preserve capital."
        }

    rr = reward / risk if risk > 0 else 0

    if rr >= 2:
        mode = "Deploy"
    elif rr >= 1.2:
        mode = "Reduced Size"
    else:
        mode = "No Trade"

    return {
        "mode": mode,
        "invalidation": round(invalidation, 2),
        "target": round(target, 2),
        "rr": round(rr, 2),
        "message": "Risk/reward acceptable." if rr >= 1.2 else "Risk/reward not favorable."
    }
def watchlist_engine(tickers):
    results = []

    for ticker in tickers:
        df = load_data(ticker, "1y", "1d")

        if df.empty:
            results.append({
                "Symbol": ticker,
                "Last": "N/A",
                "Net Chg": 0,
                "% Chg": 0,
                "Bid": "N/A",
                "Ask": "N/A",
                "ATR": 0,
                "Bias": "No Data",
                "Bullish %": 50,
                "Bearish %": 50,
                "Setup": "Data Offline",
                "Score": 0,
                "Action": "CHECK"
            })
            continue

        df = add_indicators(df)

        levels = liquidity_levels(df)
        mtf = multi_timeframe_analysis(ticker)
        final_bias, bull_prob, bear_prob, neutral_prob = overall_bias(mtf)

        historical_returns = find_similar_setups(df)
        pattern_stats = analyze_forward_returns(historical_returns)

        current_price = levels["Current Price"]
        options_data = options_positioning(ticker, current_price)

        trade_setup = trade_setup_engine(
            final_bias,
            levels,
            options_data,
            pattern_stats
        )

        quote = quote_snapshot(ticker, df)

        action = "WAIT"

        if trade_setup["score"] >= 4:
            action = "READY"
        elif trade_setup["score"] <= 0:
            action = "AVOID"

        results.append({
            "Symbol": ticker,
            "Last": quote["last"],
            "Net Chg": quote["net_change"],
            "% Chg": quote["pct_change"],
            "Bid": quote["bid"],
            "Ask": quote["ask"],
            "ATR": quote["atr"],
            "Bias": final_bias,
            "Bullish %": bull_prob,
            "Bearish %": bear_prob,
            "Setup": trade_setup["setup"],
            "Score": trade_setup["score"],
            "Action": action
        })

    return pd.DataFrame(results)

def entry_engine(trade_setup, levels):
    price = levels["Current Price"]
    vwap = levels["VWAP Magnet"]
    prior_high = levels["Prior High Liquidity"]
    prior_low = levels["Prior Low Liquidity"]

    setup = trade_setup["setup"]

    if "Long" in setup:
        entry_type = "Pullback Long"
        entry_zone_low = vwap
        entry_zone_high = vwap + (price - vwap) * 0.5
        breakout_entry = prior_high
        confirmation = "Hold above VWAP / reclaim prior high"
        avoid = "Avoid chasing above prior high"

    elif "Short" in setup:
        entry_type = "Pullback Short"
        entry_zone_low = vwap - (vwap - price) * 0.5
        entry_zone_high = vwap
        breakout_entry = prior_low
        confirmation = "Hold below VWAP / lose prior low"
        avoid = "Avoid chasing below prior low"

    else:
        return {
            "entry_type": "No Trade",
            "entry_zone": None,
            "breakout_entry": None,
            "confirmation": "Wait for cleaner setup",
            "avoid": "No edge right now"
        }

    return {
        "entry_type": entry_type,
        "entry_zone": f"{round(entry_zone_low, 2)} - {round(entry_zone_high, 2)}",
        "breakout_entry": round(breakout_entry, 2),
        "confirmation": confirmation,
        "avoid": avoid
    }
def trigger_engine(entry_plan, levels):
    price = levels["Current Price"]
    vwap = levels["VWAP Magnet"]

    trigger = "WAIT"
    message = "No trigger yet"

    if entry_plan["entry_type"] == "Pullback Long":
        low, high = map(float, entry_plan["entry_zone"].split(" - "))

        if low <= price <= high:
            trigger = "ACTIVE"
            message = "Price in pullback zone — watch for bounce"

        if price > entry_plan["breakout_entry"]:
            trigger = "BREAKOUT"
            message = "Breakout triggered above key level"

    elif entry_plan["entry_type"] == "Pullback Short":
        low, high = map(float, entry_plan["entry_zone"].split(" - "))

        if low <= price <= high:
            trigger = "ACTIVE"
            message = "Price in short zone — watch for rejection"

        if price < entry_plan["breakout_entry"]:
            trigger = "BREAKDOWN"
            message = "Breakdown triggered below key level"

    return {
        "trigger": trigger,
        "message": message
    }
def alert_engine(price, entry_plan, trigger_plan):
    alerts = []

    entry_zone = entry_plan.get("entry_zone")

    if entry_zone and " - " in str(entry_zone):
        try:
            low, high = map(float, entry_zone.split(" - "))
            if low <= price <= high:
                alerts.append("🟡 PRICE IN ENTRY ZONE")
        except:
            pass

    if trigger_plan["trigger"] == "BREAKOUT":
        alerts.append("🚀 BREAKOUT TRIGGERED")
    elif trigger_plan["trigger"] == "BREAKDOWN":
        alerts.append("🔴 BREAKDOWN TRIGGERED")
    elif trigger_plan["trigger"] == "ACTIVE":
        alerts.append("⚡ SETUP ACTIVE")

    return alerts

def probability_engine(final_bias, trade_setup, levels, options_data, delta_flow, gex, pattern_stats):
    score = 50

    if "Bullish" in final_bias:
        score += 10
    elif "Bearish" in final_bias:
        score += 10

    if trade_setup["score"] >= 4:
        score += 15
    elif trade_setup["score"] >= 2:
        score += 8
    elif trade_setup["score"] <= -2:
        score += 8

    if options_data:
        if "Put-Heavy" in options_data["positioning"]:
            score += 8
        if "Call-Heavy" in options_data["positioning"]:
            score += 8

    if delta_flow:
        if "Bearish" in delta_flow["bias"] and "Bearish" in final_bias:
            score += 12
        elif "Bullish" in delta_flow["bias"] and "Bullish" in final_bias:
            score += 12
        else:
            score -= 8

    if gex:
        if "Negative Gamma" in gex["regime"]:
            score += 8
        elif "Positive Gamma" in gex["regime"]:
            score -= 5

    if pattern_stats:
        if pattern_stats["win_rate"] >= 60:
            score += 10
        elif pattern_stats["win_rate"] <= 45:
            score -= 8

    score = max(0, min(95, score))

    if score >= 75:
        strength = "STRONG"
    elif score >= 60:
        strength = "MEDIUM"
    else:
        strength = "WEAK"

    return {
        "confidence": round(score),
        "strength": strength,
        "target_1_prob": min(90, round(score + 10)),
        "target_2_prob": round(score),
        "runner_prob": max(10, round(score - 20))
    }
def position_size_engine(probability, account_size=10000, max_risk_pct=1):
    confidence = probability["confidence"]

    if confidence >= 80:
        size = "FULL SIZE"
        risk_pct = max_risk_pct
    elif confidence >= 65:
        size = "HALF SIZE"
        risk_pct = max_risk_pct * 0.5
    elif confidence >= 50:
        size = "SMALL SIZE"
        risk_pct = max_risk_pct * 0.25
    else:
        size = "NO TRADE"
        risk_pct = 0

    risk_dollars = account_size * (risk_pct / 100)

    return {
        "size": size,
        "risk_pct": round(risk_pct, 2),
        "risk_dollars": round(risk_dollars, 2)
    }
def auto_trade_signal(probability, final_bias, trigger_plan):
    confidence = probability["confidence"]
    trigger = trigger_plan["trigger"]

    if confidence < 50 or trigger == "WAIT":
        return {
            "signal": "🟡 WAIT",
            "color": "orange",
            "message": "No confirmed trade yet"
        }

    if "Bullish" in final_bias:
        if confidence >= 75:
            signal = "🟢 STRONG CALL"
        else:
            signal = "🟢 CALL WATCH"

        return {
            "signal": signal,
            "color": "lime",
            "message": "Bullish setup confirmed"
        }

    if "Bearish" in final_bias:
        if confidence >= 75:
            signal = "🔴 STRONG PUT"
        else:
            signal = "🔴 PUT WATCH"

        return {
            "signal": signal,
            "color": "red",
            "message": "Bearish setup confirmed"
        }

    return {
        "signal": "🟡 WAIT",
        "color": "orange",
        "message": "Mixed environment"
    }

def pro_trade_label_engine(final_bias, delta_flow, orb, probability):
    delta_bias = delta_flow["bias"] if delta_flow else ""

    if orb["trigger"] == "ORB BREAKOUT" and orb["strength"] == "HIGH CONFIRMATION":
        if "Bearish" in delta_bias:
            return "⚡ SCALP LONG / SQUEEZE", "Bearish delta trapped by bullish breakout."
        return "🚀 TREND LONG", "Breakout confirmed with volume."

    if orb["trigger"] == "ORB BREAKDOWN" and orb["strength"] == "HIGH CONFIRMATION":
        if "Bullish" in delta_bias:
            return "⚡ SCALP SHORT / LONG TRAP", "Bullish positioning failed into breakdown."
        return "🔴 TREND SHORT", "Breakdown confirmed with volume."

    if probability["confidence"] < 60:
        return "⏳ WAIT", "Low confidence. No clean trade."

    return "🟡 FADE / CHOP", "Signals mixed. Avoid chasing."

def exit_alert_engine(price, risk_plan, trade_setup):
    alerts = []

    target = risk_plan.get("target")
    stop = risk_plan.get("invalidation")

    if target is not None:
        if "Long" in trade_setup["setup"] and price >= target:
            alerts.append("🎯 TARGET HIT (LONG)")
        elif "Short" in trade_setup["setup"] and price <= target:
            alerts.append("🎯 TARGET HIT (SHORT)")

    if stop is not None:
        if "Long" in trade_setup["setup"] and price <= stop:
            alerts.append("❌ STOP HIT (LONG)")
        elif "Short" in trade_setup["setup"] and price >= stop:
            alerts.append("❌ STOP HIT (SHORT)")

    return alerts

def send_telegram_alert(message):
    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
    chat_id = st.secrets["TELEGRAM_CHAT_ID"]

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message
    }

    try:
        requests.post(url, data=payload, timeout=5)
    except Exception:
        pass
def get_regime_color(bias):
    if "Bullish" in bias:
        return "green"
    elif "Bearish" in bias:
        return "red"
    else:
        return "orange"
def regime_warning_engine(final_bias, trade_setup, levels, fibs, options_data):
    price = levels["Current Price"]
    high = levels["Prior High Liquidity"]
    low = levels["Prior Low Liquidity"]
    vwap = levels["VWAP Magnet"]

    fib_161 = fibs.get("161.8 Extension")
    fib_222 = fibs.get("222.0 Extension")

    regime = "🟡 Chop / Wait"
    warning = "No clean dominant regime."
    action = "Wait"

    if "Bullish" in final_bias:
        regime = "🟢 Bullish Environment"
        warning = "Bullish structure active."
        action = "Look for long setups"

        if price > high:
            regime = "🟠 Bullish but Extended"
            warning = "Price is above prior high liquidity. Do not chase."
            action = "Wait for pullback"

        if fib_161 and price > fib_161:
            regime = "🟠 Exhaustion Risk"
            warning = "Price is above 161.8 extension. Trend may continue, but entry risk is elevated."
            action = "Avoid chasing"

        if fib_222 and price > fib_222:
            regime = "🟣 Blowoff / Regime Transition Risk"
            warning = "Price is beyond 222 extension. Possible squeeze climax or transition."
            action = "Protect profits / wait"

    elif "Bearish" in final_bias:
        regime = "🔴 Bearish Environment"
        warning = "Bearish structure active."
        action = "Look for short setups"

        if price < low:
            regime = "🟠 Bearish but Extended"
            warning = "Price is below prior low liquidity. Do not chase downside."
            action = "Wait for bounce"

    else:
        if abs(price - vwap) / price < 0.005:
            regime = "🔵 Compression / Balance"
            warning = "Price is near VWAP. Market may be coiling before expansion."
            action = "Wait for break"

    if options_data:
        if "Call-Heavy" in options_data["positioning"] and "Bullish" in final_bias:
            warning += " Call-heavy positioning adds long-crowding risk."

        if "Put-Heavy" in options_data["positioning"] and "Bearish" in final_bias:
            warning += " Put-heavy positioning adds downside-crowding risk."

    return {
        "regime": regime,
        "warning": warning,
        "action": action
    }

def stage15_final_panel(final_bias, levels, options_data, entry_plan, risk_plan, mm, orb):
    price = levels["Current Price"]
    vwap = levels["VWAP Magnet"]
    prior_high = levels["Prior High Liquidity"]
    prior_low = levels["Prior Low Liquidity"]

    control = "Buyers in control above VWAP" if price > vwap else "Sellers in control below VWAP"

    if "Bullish" in final_bias and price > vwap:
        decision = "CALL ONLY"
        color = "lime"
    elif "Bearish" in final_bias and price < vwap:
        decision = "PUT ONLY"
        color = "red"
    else:
        decision = "WAIT"
        color = "orange"

    call_wall = options_data["call_wall"] if options_data else "N/A"
    put_wall = options_data["put_wall"] if options_data else "N/A"
    positioning = options_data["positioning"] if options_data else "No options data"

    expected_low = mm["expected"]["expected_low"]
    expected_high = mm["expected"]["expected_high"]

    return {
    "decision": decision,
    "color": color,
    "control": control,
    "environment": final_bias,
    "positioning": positioning,
    "expected_range": f"{expected_low} - {expected_high}",
    "upside": f"{round(prior_high, 2)} / Call Wall: {call_wall}",
    "downside": f"{round(prior_low, 2)} / Put Wall: {put_wall}",
    "magnet": f"VWAP: {round(vwap, 2)}",
    "best_entry": entry_plan.get("entry_zone"),
    "avoid": entry_plan.get("avoid"),
    "target": risk_plan.get("target"),
    "bear_flip": round(prior_low, 2),

    # ✅ MM ENGINE
    "upper_magnet": mm["upper_magnet"],
    "lower_magnet": mm["lower_magnet"],
    "premarket_high": mm["intraday"]["premarket_high"],
    "premarket_low": mm["intraday"]["premarket_low"],
    "opening_high": mm["intraday"]["opening_range_high"],
    "opening_low": mm["intraday"]["opening_range_low"],

    # 🔥 THIS WAS MISSING
    "orb_trigger": orb["trigger"],
    "orb_strength": orb["strength"],
}

@st.cache_data(ttl=10, show_spinner=False)
def quote_snapshot(ticker, df):
    if df is None or df.empty or len(df) < 2:
        return {
            "ticker": ticker,
            "last": 0,
            "net_change": 0,
            "pct_change": 0,
            "bid": 0,
            "ask": 0,
            "atr": 0,
            "color": "gray"
        }

    stock = yf.Ticker(ticker)
    last = float(df.iloc[-1]["Close"])
    atr = float(df.iloc[-1]["ATR"]) if "ATR" in df.columns else 0

    # ✅ CORRECT DAILY REFERENCE
    try:
        hist = stock.history(period="5d", interval="1d", auto_adjust=False)

        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
        else:
            prev_close = float(df.iloc[-2]["Close"])

    except Exception:
        prev_close = float(df.iloc[-2]["Close"])

    # ✅ BID / ASK
    try:
        info = stock.fast_info
        bid = float(info.get("bid", last) or last)
        ask = float(info.get("ask", last) or last)
    except Exception:
        bid = last
        ask = last

    net_change = last - prev_close
    pct_change = (net_change / prev_close * 100) if prev_close else 0

    return {
        "ticker": ticker,
        "last": round(last, 2),
        "net_change": round(net_change, 2),
        "pct_change": round(pct_change, 2),
        "bid": round(bid, 2),
        "ask": round(ask, 2),
        "atr": round(atr, 2),
        "color": "green" if net_change >= 0 else "red"
    }

def quotron_bar(watchlist):
    quotes = []

    for ticker in watchlist:
        df = load_data(ticker, "5d", "5m")

        if df.empty or len(df) < 2:
            continue

        df = add_indicators(df)
        q = quote_snapshot(ticker, df)
        quotes.append(q)

    return quotes


@st.cache_data(ttl=10, show_spinner=False)
def stage14_rank_engine(watchlist):
    rows = []

    for ticker in watchlist:
        df = load_data(ticker, "5d", "5m")

        if df.empty:
            df = load_data(ticker, "1mo", "30m")
        if df.empty:
            df = load_data(ticker, "1y", "1d")

        if df.empty:
            rows.append({
                "Symbol": ticker, "Last": None, "Net Chg": None, "% Chg": None,
                "Bid": None, "Ask": None, "ATR": None,
                "Bias": "No Data", "Bull %": 50, "Bear %": 50,
                "Contract": "WAIT", "Timing": "WAIT",
                "Setup": "Data Offline", "Trigger": "WAIT",
                "Mode": "No Data", "Score": 0,
                "Action": "CHECK", "Rank Score": -999
            })
            continue

        df = add_indicators(df)

        if df.empty:
            continue

        levels = liquidity_levels(df)
        fibs = calculate_fibs(df)
        mtf = multi_timeframe_analysis(ticker)
        final_bias, bull_prob, bear_prob, neutral_prob = overall_bias(mtf)

        hist = find_similar_setups(df)
        pattern_stats = analyze_forward_returns(hist)
        options_data = options_positioning(ticker, levels["Current Price"])

        trade_setup = trade_setup_engine(final_bias, levels, options_data, pattern_stats)
        risk_plan = risk_management_engine(trade_setup, levels)
        entry_plan = entry_engine(trade_setup, levels)
        trigger_plan = trigger_engine(entry_plan, levels)
        alerts = alert_engine(levels["Current Price"], entry_plan, trigger_plan)
        regime_plan = regime_warning_engine(final_bias, trade_setup, levels, fibs, options_data)

        quote = quote_snapshot(ticker, df)

        score = trade_setup["score"]

        if risk_plan["mode"] == "Deploy":
            score += 2
        elif risk_plan["mode"] == "Reduced Size":
            score += 1

        if trigger_plan["trigger"] in ["BREAKOUT", "BREAKDOWN"]:
            score += 2
        elif trigger_plan["trigger"] == "ACTIVE":
            score += 1

        if "Avoid" in regime_plan["action"]:
            score -= 2

        if score >= 6:
            action = "READY"
        elif score >= 3:
            action = "WAIT"
        else:
            action = "AVOID"

        contract = "CALL" if "Long" in trade_setup["setup"] else "PUT" if "Short" in trade_setup["setup"] else "WAIT"

        price = levels["Current Price"]
        entry_zone = entry_plan.get("entry_zone")
        timing = "WAIT"

        try:
            if entry_zone and " - " in str(entry_zone):
                low, high = map(float, entry_zone.split(" - "))
                zone_size = high - low

                if zone_size > 0:
                    position = (price - low) / zone_size

                    if position <= 0.33:
                        timing = "EARLY"
                    elif position <= 0.66:
                        timing = "MID"
                    else:
                        timing = "LATE"
        except Exception:
            timing = "WAIT"

        rows.append({
            "Symbol": ticker,
            "Last": quote["last"],
            "Net Chg": quote["net_change"],
            "% Chg": quote["pct_change"],
            "Bid": quote["bid"],
            "Ask": quote["ask"],
            "ATR": quote["atr"],
            "Bias": final_bias,
            "Bull %": bull_prob,
            "Bear %": bear_prob,
            "Setup": trade_setup["setup"],
            "Trigger": trigger_plan["trigger"],
            "Mode": risk_plan["mode"],
            "Score": trade_setup["score"],
            "Action": action,
            "Contract": contract,
            "Timing": timing,
            "Rank Score": score
        })

    return pd.DataFrame(rows)


def make_chart(df, fibs, levels, poc, hvns, lvns, options_data, timeframe, final_bias, trigger_plan, entry_plan, risk_plan):
    x_col = "Date" if "Date" in df.columns else "Datetime"

    from plotly.subplots import make_subplots

    fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.78, 0.22]
)
    # -----------------------------
    # WHAT = BIAS BACKGROUND
    # -----------------------------
    bias_color = (
        "rgba(0,100,0,0.08)" if "Bullish" in final_bias else
        "rgba(150,0,0,0.08)" if "Bearish" in final_bias else
        "rgba(150,150,0,0.08)"
    )

    fig.update_layout(
        plot_bgcolor=bias_color,
        paper_bgcolor="#050505"
    )
    fig.add_trace(go.Candlestick(
    x=df[x_col],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price",
    hovertemplate=
    "Time: %{x}<br>" +
    "Open: %{open:.2f}<br>" +
    "High: %{high:.2f}<br>" +
    "Low: %{low:.2f}<br>" +
    "Close: %{close:.2f}<extra></extra>",
    increasing=dict(line=dict(color="#00ff66", width=1), fillcolor="#00ff66"),
    decreasing=dict(line=dict(color="#ff3b3b", width=1), fillcolor="#ff3b3b")
), row=1, col=1)
    
    fig.add_trace(
    go.Bar(
        x=df[x_col],
        y=df["Volume"],
        name="Volume",
        marker_color="rgba(120,120,120,0.35)",
        hovertemplate=
        "Time: %{x}<br>" +
        "Open: %{open:.2f}<br>" +
        "High: %{high:.2f}<br>" +
        "Low: %{low:.2f}<br>" +
        "Close: %{close:.2f}<extra></extra>"
    ),
    row=2,
    col=1
)
    fig.update_layout(
    height=550,
    width=1325,
    dragmode="pan",
    hovermode="x",
    uirevision="keep",
    xaxis_rangeslider_visible=False,
    plot_bgcolor=bias_color,
    paper_bgcolor="#050505",
    font=dict(color="white"),
    margin=dict(l=10, r=40, t=30, b=10),
    spikedistance=-1,
  
)
    fig.add_trace(
    go.Scatter(x=df[x_col], y=df["EMA_20"], name="EMA 20", line=dict(color="white", width=1)),
    row=1, col=1
)

    fig.add_trace(
    go.Scatter(x=df[x_col], y=df["EMA_50"], name="EMA 50", line=dict(color="#00BFFF", width=1)),
    row=1, col=1
)

    fig.add_trace(
    go.Scatter(x=df[x_col], y=df["EMA_200"], name="EMA 200", line=dict(color="#FF00FF", width=1)),
    row=1, col=1
)

    fig.add_trace(
    go.Scatter(x=df[x_col], y=df["VWAP"], name="VWAP", line=dict(color="yellow", width=2)),
    row=1, col=1
)
    
    fig.add_trace(
    go.Scatter(x=df[x_col], y=df["VWAP_Upper_1"], name="VWAP +1σ", line=dict(color="rgba(255,255,0,0.5)", width=1, dash="dot")),
    row=1, col=1
)

    fig.add_trace(
    go.Scatter(x=df[x_col], y=df["VWAP_Lower_1"], name="VWAP -1σ", line=dict(color="rgba(255,255,0,0.5)", width=1, dash="dot")),
    row=1, col=1
)

    fig.add_trace(
    go.Scatter(x=df[x_col], y=df["VWAP_Upper_2"], name="VWAP +2σ", line=dict(color="rgba(255,165,0,0.6)", width=1, dash="dash")),
    row=1, col=1
)

    fig.add_trace(
    go.Scatter(x=df[x_col], y=df["VWAP_Lower_2"], name="VWAP -2σ", line=dict(color="rgba(255,165,0,0.6)", width=1, dash="dash")),
    row=1, col=1
)

    for label, level in fibs.items():
        fig.add_hline(
        y=level,
        line_dash="dot",
        line_color="white",
        line_width=1,
        annotation_text=label,
        annotation_font_color="white"
    )

    for label, level in levels.items():
        fig.add_hline(
        y=level,
        line_dash="dash",
        line_color="white",
        line_width=1,
        annotation_text=label,
        annotation_font_color="white"
    )

        fig.add_hline(
        y=poc,
        line_color="white",
        line_width=3,
        annotation_text="POC",
        annotation_font_color="white"
    )

    for level in hvns:
        fig.add_hline(
        y=level,
        line_dash="dashdot",
        line_color="white",
        line_width=1,
        opacity=0.4
    )

    for level in lvns:
        fig.add_hline(
        y=level,
        line_dash="longdash",
        line_color="white",
        line_width=1,
        opacity=0.35
    )

    if options_data is not None:
        fig.add_hline(
        y=options_data["call_wall"],
        line_color="lime",
        line_width=3,
        annotation_text="CALL WALL",
        annotation_font_color="lime"
    )

    fig.add_hline(
        y=options_data["put_wall"],
        line_color="red",
        line_width=3,
        annotation_text="PUT WALL",
        annotation_font_color="red"
    )
    fig.update_layout(
    height=550,
    width=1325,
    dragmode="pan",
    hovermode="x unified",
    uirevision="keep",
    xaxis_rangeslider_visible=False,
    plot_bgcolor=bias_color,
    paper_bgcolor="#050505",
    font=dict(color="white"),
    margin=dict(l=10, r=40, t=30, b=10),
    spikedistance=-1,
    legend=dict(
        font=dict(color="white"),
        bgcolor="rgba(0,0,0,0.35)",
        bordercolor="rgba(255,255,255,0.25)",
        borderwidth=1
    )
)


    fig.update_xaxes(
    showspikes=True,
    spikemode="across",
    spikesnap="cursor",   # 🔑 THIS IS THE KEY
    spikecolor="white",
    spikethickness=1,
    showline=True,
    showgrid=True,
    gridcolor="#333",
    color="white",
    hoverformat="%m/%d %I:%M %p"
)

    fig.update_yaxes(
    showspikes=True,
    spikemode="across",
    spikesnap="cursor",   # 🔑 THIS FIXES YOUR PROBLEM
    spikecolor="white",
    spikethickness=1,
    showline=True,
    showgrid=True,
    gridcolor="#333",
    color="white",
    tickformat=".2f"
)
    fig.update_yaxes(
    row=2,
    col=1,
    range=[0, df["Volume"].max() * 1.2],
    visible=False,
    fixedrange=True
)
    # -----------------------------
    # PRICE LINE (BUBBLE STYLE)
    # -----------------------------
    current_price = df.iloc[-1]["Close"]

    fig.add_hline(
        y=current_price,
        line_color="lime",
        line_width=2,
        annotation_text=f"{round(current_price, 2)}",
        annotation_position="right",
        annotation_font=dict(size=16, color="white"),
        annotation_bgcolor="green"
    )

    # -----------------------------
    # HIGH MARKER
    # -----------------------------
    day_high = df["High"].tail(20).max()

    fig.add_hline(
        y=day_high,
        line_color="white",
        line_width=2,
        annotation_text=f"Hi {round(day_high, 2)}",
        annotation_position="right",
        annotation_font=dict(size=14, color="black"),
        annotation_bgcolor="white"
    )

    # -----------------------------
    # AUTO ZOOM (CLEAN VERSION)
    # -----------------------------
    bars_map = {
    "1d": 120,
    "1h": 80,
    "30m": 80,
    "15m": 70,
    "5m": 60
    }

    bars = bars_map.get(timeframe, 120)

    if len(df) > 1:
        bars = min(bars, len(df) - 1)
    else:
        bars = 1

    fig.update_xaxes(
        range=[df[x_col].iloc[-bars], df[x_col].iloc[-1]]
    )
    visible_df = df.tail(bars)

    y_low = visible_df["Low"].min()
    y_high = visible_df["High"].max()

    padding = (y_high - y_low) * 0.15

    fig.update_yaxes(
    range=[y_low - padding, y_high + padding],
    showgrid=True,
    gridcolor="#333",
    color="white",
    fixedrange=False
    )
    if timeframe != "1d":
        fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[20, 4], pattern="hour")
        ]
    )
    else:
        fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"])
        ]
    )
    # -----------------------------
    # WHEN = TRIGGER MARKER
    # -----------------------------
    if trigger_plan["trigger"] in ["BREAKOUT", "BREAKDOWN", "ACTIVE"]:
        fig.add_annotation(
        x=df[x_col].iloc[-1],
        y=levels["Current Price"],
        text=trigger_plan["trigger"],
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="white", size=12),
        bgcolor="green" if "BREAKOUT" in trigger_plan["trigger"] else
                 "red" if "BREAKDOWN" in trigger_plan["trigger"] else
                 "orange"
    )
        
# -----------------------------
# ALWAYS SHOW ENTRY / STOP / TARGET ZONES
# -----------------------------
    entry_zone = entry_plan.get("entry_zone")

    if entry_zone and " - " in str(entry_zone):
        entry_low, entry_high = map(float, entry_zone.split(" - "))
        zone = entry_high - entry_low

        early_high = entry_low + zone * 0.33
        mid_high = entry_low + zone * 0.66

        fig.add_hrect(y0=entry_low, y1=early_high, fillcolor="green", opacity=0.18, line_width=0, annotation_text="EARLY / BEST", row=1, col=1)
        fig.add_hrect(y0=early_high, y1=mid_high, fillcolor="yellow", opacity=0.18, line_width=0, annotation_text="MID / OK", row=1, col=1)
        fig.add_hrect(y0=mid_high, y1=entry_high, fillcolor="red", opacity=0.18, line_width=0, annotation_text="LATE / AVOID", row=1, col=1)


    if risk_plan.get("invalidation") is not None:
        fig.add_hline(
        y=risk_plan["invalidation"],
        line_color="red",
        line_width=3,
        annotation_text="STOP / INVALIDATION",
        annotation_font_color="red"
    )

    if risk_plan.get("target") is not None:
        fig.add_hline(
        y=risk_plan["target"],
        line_color="lime",
        line_width=3,
        annotation_text="TARGET",
        annotation_font_color="lime"
    )
    bear_flip = (
    "Bearish" in final_bias
    or df.iloc[-1]["Close"] < df.iloc[-1]["VWAP"]
    and trigger_plan["trigger"] == "BREAKDOWN"
)

    if bear_flip:
        fig.add_annotation(
        x=df[x_col].iloc[-1],
        y=df["High"].iloc[-1],
        text="🔴 BEARISH FLIP",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-60,
        font=dict(color="white", size=14),
        bgcolor="red",
        row=1,
        col=1
    )
    return fig

    
WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.csv")

def load_watchlist_symbols():
    if os.path.exists(WATCHLIST_FILE):
        df = pd.read_csv(WATCHLIST_FILE)
        if "Symbol" in df.columns:
            df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
            df = df[df["Symbol"] != ""].drop_duplicates("Symbol")
            return df
    return pd.DataFrame({"Symbol": ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "META", "AMD"]})

def save_watchlist_symbols(df):
    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df = df[df["Symbol"] != ""].drop_duplicates("Symbol")
    df.to_csv(WATCHLIST_FILE, index=False)

if "watchlist_symbols" not in st.session_state:
    st.session_state.watchlist_symbols = load_watchlist_symbols()


# -----------------------------
# INPUTS (MUST COME FIRST)
# -----------------------------
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "SPY"

ticker = st.sidebar.text_input(
    "Main Ticker",
    value=st.session_state.selected_ticker,
    key="main_ticker"
).upper()
timeframe = st.sidebar.selectbox(
    "Chart Timeframe",
    ["1d", "4h", "1h", "30m", "15m", "5m", "2m", "1m"],
    index=2,
    key="chart_timeframe"
)

st.session_state.selected_ticker = ticker


# ✅ Period mapping (how much data to pull)
period_map = {
    "1d": "1y",
    "4h": "6mo",
    "1h": "3mo",
    "30m": "1mo",
    "15m": "1mo",
    "5m": "5d",
    "2m": "5d",
    "1m": "5d"
}

# ✅ Interval mapping (what yfinance actually supports)
interval_map = {
    "1d": "1d",
    "4h": "1h",   # simulate 4H using 1H data
    "1h": "1h",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
    "2m": "2m",
    "1m": "1m"
}


auto_refresh = st.sidebar.checkbox("Auto Refresh Quotes", value=True)
if auto_refresh:
    st_autorefresh(interval=15000, key="quote_refresh")

main_df = load_data(
    ticker,
    period_map[timeframe],
    interval_map[timeframe]
)

if main_df.empty:
    st.error("No data found.")
    st.stop()

main_df = add_indicators(main_df)
if main_df.empty:
    st.error("Not enough data after indicators. Try a higher timeframe like 15m, 30m, or 1h.")
    st.stop()
quote = quote_snapshot(ticker, main_df)

fibs = calculate_fibs(main_df)
levels = liquidity_levels(main_df)
poc, hvns, lvns = volume_profile(main_df)

current_price = levels["Current Price"]
options_data = options_positioning(ticker, current_price)
gex = gamma_exposure_engine(ticker, current_price)
delta_flow = delta_flow_engine(ticker, current_price)
mtf = multi_timeframe_analysis(ticker)
final_bias, bull_prob, bear_prob, neutral_prob = overall_bias(mtf)
regime_color = get_regime_color(final_bias)

st.markdown(
f"""
<div style="
    display:inline-block;
    padding:8px 16px;
    border-radius:12px;
    background:rgba(0,0,0,0.05);
    font-size:16px;
    font-weight:600;
    color:{regime_color};
    border:1px solid rgba(0,0,0,0.08);
    margin-top:10px;
    margin-bottom:12px;
">
Environment: {final_bias}
</div>
""",
unsafe_allow_html=True
)
historical_returns = find_similar_setups(main_df)
pattern_stats = analyze_forward_returns(historical_returns)

trade_setup = trade_setup_engine(
    final_bias,
    levels,
    options_data,
    pattern_stats
)

risk_plan = risk_management_engine(trade_setup, levels)
entry_plan = entry_engine(trade_setup, levels)
trigger_plan = trigger_engine(entry_plan, levels)
regime_plan = regime_warning_engine(
    final_bias,
    trade_setup,
    levels,
    fibs,
    options_data
)
probability = probability_engine(
    final_bias,
    trade_setup,
    levels,
    options_data,
    delta_flow,
    gex,
    pattern_stats
)
alerts = alert_engine(levels["Current Price"], entry_plan, trigger_plan)

# 🔥 ADD EXIT ALERTS
exit_alerts = exit_alert_engine(
    levels["Current Price"],
    risk_plan,
    trade_setup
)

position_size = position_size_engine(
    probability,
    account_size=10000,
    max_risk_pct=1
)

trade_signal = auto_trade_signal(
    probability,
    final_bias,
    trigger_plan
)
st.markdown("### 📊 Position Size Engine")

s1, s2, s3 = st.columns(3)

s1.metric("Suggested Size", position_size["size"])
s2.metric("Risk %", f"{position_size['risk_pct']}%")
s3.metric("Risk $", f"${position_size['risk_dollars']}")
position_size = position_size_engine(
    probability,
    account_size=10000,
    max_risk_pct=1
)
mm = market_maker_engine(main_df, levels, options_data, ticker)
orb = opening_range_break_engine(main_df, mm["intraday"])
panel = stage15_final_panel(
    final_bias,
    levels,
    options_data,
    entry_plan,
    risk_plan,
    mm,
    orb
)
trade_label, trade_note = pro_trade_label_engine(
    final_bias,
    delta_flow,
    orb,
    probability
)
trade_context = trade_context_engine(
    levels,
    delta_flow,
    orb,
    final_bias
)
contract = contract_selection_engine(
    levels["Current Price"],
    mm,
    probability,
    panel,
    final_bias
)
pro_alerts = []

if "SCALP LONG" in trade_label:
    pro_alerts.append("⚡ SQUEEZE STARTING / COUNTER-TREND LONG")

if "SCALP SHORT" in trade_label:
    pro_alerts.append("⚠️ LONG TRAP / COUNTER-TREND SHORT")

if probability["confidence"] < 60 and orb["trigger"] != "INSIDE RANGE":
    pro_alerts.append("⚠️ CONFLICT: Delta vs Price Action — Trap Risk HIGH")

all_alerts = alerts + exit_alerts + pro_alerts
if all_alerts:
    alert_text = f"{ticker} ALERT ({probability['confidence']}%)\n\n" + "\n".join(all_alerts)

    last_alert_key = f"last_alert_{ticker}"

    if st.session_state.get(last_alert_key) != alert_text:
        send_telegram_alert(alert_text)
        st.session_state[last_alert_key] = alert_text
panel["control"] = trade_context["control"]

price_color = "#16a34a" if quote["net_change"] >= 0 else "#dc2626"

components.html(f"""
<div style="
    display:inline-flex;
    align-items:center;
    gap:18px;
    background:#ffffff;
    border:1px solid #d9dee8;
    border-radius:12px;
    padding:10px 16px;
    box-shadow:0 2px 6px rgba(0,0,0,0.05);
    font-family:Arial, sans-serif;
    font-size:14px;
">
    <b style="color:#0b1f3a;">{ticker}</b>
    <b style="color:{price_color}; font-size:16px;">{quote["last"]}</b>
    <span style="color:{price_color};">{quote["net_change"]} ({quote["pct_change"]}%)</span>
    <span style="color:#64748b;">B:{quote["bid"]}</span>
    <span style="color:#64748b;">A:{quote["ask"]}</span>
    <span style="color:#64748b;">ATR:{quote["atr"]}</span>
</div>
""", height=55)

st.markdown("### 🧾 Best Contract")

bc1, bc2, bc3, bc4 = st.columns(4)

bc1.metric("Contract", contract["contract"])
bc2.metric("Strike", contract["strike"])
bc3.metric("Expiry", contract["expiry"])
bc4.metric("Target", contract["target"])

st.info(contract["note"])

st.markdown("## 🧠 Trading Desk Terminal")

st.markdown(
f"""
<div style="
background: linear-gradient(135deg, #0b1220, #111827);
border:1px solid rgba(255,255,255,0.08);
border-left:5px solid {panel['color']};
border-radius:12px;
padding:8px 10px;
margin:6px 0;
color:white;
">

<div style="
font-size:18px;
font-weight:700;
color:{panel['color']};
letter-spacing:0.5px;
">
{panel['decision']}
</div>

<div style="
font-size:12px;
color:#9ca3af;
margin-top:4px;
">
Trade Permission
</div>

</div>
""",
unsafe_allow_html=True
)
st.markdown(
    f"""
    <div style="
        background:#111827;
        border-left:4px solid {trade_signal['color']};
        color:white;
        padding:8px 10px;
        border-radius:10px;
        margin:6px 0;
        line-height:1.2;
    ">
        <div style="font-size:18px; font-weight:700; color:{trade_signal['color']};">
            {trade_signal['signal']}
        </div>
        <div style="font-size:12px; margin-top:2px;">
            {trade_signal['message']}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("### 🧭 Trade Context")

st.info(
    f"{trade_context['context']}\n\n{trade_context['note']}"
)


c1, c2, c3, c4 = st.columns(4)

c1.metric("Control", panel["control"])
c2.metric("Environment", panel["environment"])
c3.metric("Positioning", panel["positioning"])
c4.metric("Expected Range", panel["expected_range"])

l1, l2, l3 = st.columns(3)

l1.info(f"UPSIDE\n\n{panel['upside']}")
l2.warning(f"MAGNET\n\n{panel['magnet']}")
l3.error(f"DOWNSIDE\n\n{panel['downside']}")

e1, e2, e3, e4 = st.columns(4)

e1.success(f"BEST ENTRY\n\n{panel['best_entry']}")
e2.warning(f"AVOID\n\n{panel['avoid']}")
e3.info(f"TARGET\n\n{panel['target']}")
e4.error(f"BEAR FLIP\n\nBelow {panel['bear_flip']}")
st.markdown("### 🧲 Market Maker / Range Engine")

move_pct = (mm["expected"]["range_size"] / levels["Current Price"]) * 100

em1, em2, em3, em4 = st.columns(4)

em1.metric("Expected Low", mm["expected"]["expected_low"])
em2.metric("Expected High", mm["expected"]["expected_high"])
em3.metric("Range Size", mm["expected"]["range_size"])
em4.metric("Expected Move %", f"{round(move_pct, 2)}%")

st.info(
    f"Expected Move Range: {mm['expected']['expected_low']} → {mm['expected']['expected_high']}"
)

st.divider()

# ✅ MAGNETS (WHERE PRICE GOES)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Upper Magnet", panel["upper_magnet"])
m2.metric("Lower Magnet", panel["lower_magnet"])
m3.metric("Premarket H/L", f"{panel['premarket_high']} / {panel['premarket_low']}")
m4.metric("Opening Range", f"{panel['opening_high']} / {panel['opening_low']}")

st.divider()

# ✅ TIMING ENGINE (WHEN IT MOVES)
st.markdown("### ⚡ Opening Range Engine")

o1, o2 = st.columns(2)
o1.metric("ORB Trigger", panel["orb_trigger"])
o2.metric("Volume Strength", panel["orb_strength"])
if gex:
    g1, g2, g3, g4 = st.columns(4)

    g1.metric("GEX Regime", gex["regime"])
    g2.metric("Max Gamma Strike", gex["max_gamma_strike"])
    g3.metric("Net GEX", gex["net_gex"])
    g4.metric("Pressure", gex["pressure"])
else:
    st.warning("No gamma data available")
    st.markdown("### 🧠 Probability Engine")

p1, p2, p3, p4 = st.columns(4)

p1.metric("Confidence", f"{probability['confidence']}%")
p2.metric("Strength", probability["strength"])
p3.metric("First Target Prob", f"{probability['target_1_prob']}%")
p4.metric("Runner Prob", f"{probability['runner_prob']}%")

# 🎯 TARGET PROBABILITY MAP (HEDGE FUND STYLE)

price = levels["Current Price"]

call_wall = options_data.get("call_wall") if options_data else None
put_wall = options_data.get("put_wall") if options_data else None

# 🚨 SANITY FILTER: ignore walls too far from current price
if call_wall and abs(call_wall - price) > price * 0.20:
    call_wall = None

if put_wall and abs(put_wall - price) > price * 0.20:
    put_wall = None

# 🎯 TARGET 1 LOGIC
if "Bearish" in final_bias:
    target_1 = put_wall
else:
    target_1 = call_wall

# fallback if wall is invalid
if target_1 is None:
    target_1 = risk_plan.get("target")

target_2 = risk_plan.get("target")

runner = (
    mm["expected"]["expected_low"]
    if "Bearish" in final_bias
    else mm["expected"]["expected_high"]
)

st.markdown("### 🎯 Target Probability Map")

tp1, tp2, tp3 = st.columns(3)

tp1.metric("Target 1", f"{target_1}", f"{probability['target_1_prob']}%")
tp2.metric("Main Target", f"{target_2}", f"{probability['target_2_prob']}%")
tp3.metric("Runner Target", f"{runner}", f"{probability['runner_prob']}%")

st.markdown("### 🧨 Delta Flow Engine")

if delta_flow:
    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Delta Bias", delta_flow["bias"])
    d2.metric("Net Delta", delta_flow["net_delta"])
    d3.metric("Call Delta", delta_flow["call_delta"])
    d4.metric("Put Delta", delta_flow["put_delta"])

    st.info(delta_flow["read"])
else:
    st.warning("No delta flow data available")
    # 🚨 ALERT SYSTEM
    st.markdown("### 🚨 Live Alerts")

if alerts:
    for a in alerts:
        st.error(a)
else:
    st.info("No active alerts")

# -----------------------------
# TOP DECISION BAR
# -----------------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Ticker", ticker)
col2.metric("Bias", final_bias)
col3.metric("Bull %", f"{bull_prob}%")
col4.metric("Bear %", f"{bear_prob}%")
col5.metric("Setup", trade_setup["setup"])
col6.metric("Mode", risk_plan["mode"])

# -----------------------------
# QUOTRON (TOP BAR)
# -----------------------------
watchlist = (
    st.session_state.watchlist_symbols["Symbol"]
    .dropna()
    .astype(str)
    .str.upper()
    .str.strip()
    .tolist()
)
for symbol in watchlist:
    scan_df = load_data(symbol, "5d", "5m")
    if scan_df.empty:
        continue

    scan_df = add_indicators(scan_df)
    scan_levels = liquidity_levels(scan_df)
    scan_options = options_positioning(symbol, scan_levels["Current Price"])
    scan_mtf = multi_timeframe_analysis(symbol)
    scan_bias, _, _, _ = overall_bias(scan_mtf)

    scan_trade_setup = trade_setup_engine(
        scan_bias,
        scan_levels,
        scan_options,
        None
    )

    scan_risk = risk_management_engine(scan_trade_setup, scan_levels)
    scan_entry = entry_engine(scan_trade_setup, scan_levels)
    scan_trigger = trigger_engine(scan_entry, scan_levels)

    scan_alerts = alert_engine(
        scan_levels["Current Price"],
        scan_entry,
        scan_trigger
    )

    if scan_alerts:
        alert_text = f"{symbol} WATCHLIST ALERT\n\n" + "\n".join(scan_alerts)

        key = f"watchlist_alert_{symbol}"

        if st.session_state.get(key) != alert_text:
            send_telegram_alert(alert_text)
            st.session_state[key] = alert_text
quotes = quotron_bar(watchlist)

ticker_html = ""

for q in quotes:
    color = "lime" if q["net_change"] >= 0 else "red"

    ticker_html += f"""
    <span style="margin-right:22px;">
    <b>{q['ticker']}</b>
    <span style="color:{color};">{q['last']}</span>
    <span style="color:{color};">{q['net_change']} ({q['pct_change']}%)</span>
    B:{q['bid']} A:{q['ask']} ATR:{q['atr']}
    </span>
    """

st.markdown(
    f"""
    <div style="
    background:#000;
    color:white;
    padding:4px 10px;
    border-radius:8px;
    font-size:12px;
    margin-bottom:6px;
    ">
    <marquee behavior="scroll" direction="left" scrollamount="4">
    {ticker_html}
    </marquee>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# FULL WIDTH CHART (TOP PRIORITY)
# -----------------------------
@st.fragment(run_every="15s")
def live_chart():

    fresh_df = load_data(
        ticker,
        period_map[timeframe],
        interval_map[timeframe]
    )

    if fresh_df.empty:
        st.warning("Chart data loading...")
        return

    fresh_df = add_indicators(fresh_df)
    fresh_fibs = calculate_fibs(fresh_df)
    fresh_levels = liquidity_levels(fresh_df)
    fresh_poc, fresh_hvns, fresh_lvns = volume_profile(fresh_df)

    fresh_trade_setup = trade_setup_engine(
        final_bias,
        fresh_levels,
        options_data,
        pattern_stats
    )

    fresh_risk_plan = risk_management_engine(fresh_trade_setup, fresh_levels)
    fresh_entry_plan = entry_engine(fresh_trade_setup, fresh_levels)
    fresh_trigger_plan = trigger_engine(fresh_entry_plan, fresh_levels)

    fig = make_chart(
        fresh_df,
        fresh_fibs,
        fresh_levels,
        fresh_poc,
        fresh_hvns,
        fresh_lvns,
        options_data,
        timeframe,
        final_bias,
        fresh_trigger_plan,
        fresh_entry_plan,
        fresh_risk_plan
    )

    st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
    }
)

live_chart()

st.divider()

a, b, c, d = st.columns(4)

# WHAT / WHERE / FUEL / CONFIDENCE stays SAME
# -----------------------------
# PRO SIDEBAR TERMINAL
# -----------------------------
st.sidebar.markdown("## ⚡ Trade Panel")

# 🧠 REGIME
st.sidebar.markdown("### 🧠 Market Regime")
st.sidebar.markdown(
    f"""
**State:** {regime_plan['regime']}  
**Bias:** {final_bias}  
**Action:** {regime_plan['action']}
"""
)

# 🎯 ENTRY
st.sidebar.markdown("### 🎯 Entry Plan")
st.sidebar.markdown(
    f"""
**Type:** {entry_plan['entry_type']}  
**Zone:** {entry_plan['entry_zone']}  
**Breakout:** {entry_plan['breakout_entry']}
"""
)

# ⚠️ CONFIRMATION / AVOID
st.sidebar.markdown(
    f"""
**Confirm:** {entry_plan['confirmation']}  
**Avoid:** {entry_plan['avoid']}
"""
)

# ⚡ TRIGGER
st.sidebar.markdown("### ⚡ Trigger")
st.sidebar.markdown(
    f"""
**Status:** {trigger_plan['trigger']}  
**Note:** {trigger_plan['message']}
"""
)

# 📊 TRADE QUALITY
st.sidebar.markdown("### 📊 Trade Quality")
st.sidebar.markdown(
    f"""
**Setup:** {trade_setup['setup']}  
**Score:** {trade_setup['score']}  
**Mode:** {risk_plan['mode']}
"""
)

# 💰 RISK
if risk_plan["mode"] != "No Trade":
    st.sidebar.markdown("### 💰 Risk Plan")
    st.sidebar.markdown(
        f"""
**Invalidation:** {risk_plan['invalidation']}  
**Target:** {risk_plan['target']}  
**R/R:** {risk_plan['rr']}
"""
    )

# 🧾 REASONS
st.sidebar.markdown("### 🧾 Why")
for r in trade_setup["reasons"]:
    st.sidebar.markdown(f"• {r}")
# -----------------------------
# SECOND ROW: MTF / LIQUIDITY / OPTIONS / HISTORY
# -----------------------------
a, b, c, d = st.columns(4)

with a:
    st.subheader("🧠 WHAT (Trend)")

    for tf in ["Daily", "4H", "1H", "15m"]:
        result = mtf.get(tf)

        if result:
            if result["bias"] == "Bullish":
                st.success(f"{tf}: {result['state']}")
            elif result["bias"] == "Bearish":
                st.error(f"{tf}: {result['state']}")
            else:
                st.warning(f"{tf}: {result['state']}")
        else:
            st.warning(f"{tf}: No data")

with b:
    st.subheader("📍 WHERE (Levels)")

    st.metric("Price", f"${levels['Current Price']:.2f}")
    st.write(f"**Prior High:** {round(levels['Prior High Liquidity'], 2)}")
    st.write(f"**Prior Low:** {round(levels['Prior Low Liquidity'], 2)}")
    st.write(f"**VWAP:** {round(levels['VWAP Magnet'], 2)}")
    st.write(f"**POC:** {round(poc, 2)}")

with c:
    st.subheader("⚡ FUEL (Options)")

    if options_data:
        st.write(f"**Put/Call:** {options_data['put_call_ratio']}")
        st.write(f"**Call Wall:** {options_data['call_wall']}")
        st.write(f"**Put Wall:** {options_data['put_wall']}")
        st.write(f"**Positioning:** {options_data['positioning']}")
        st.write(f"**Pressure:** {options_data['dealer_pressure']}")
    else:
        st.warning("No options data")

with d:
    st.subheader("📊 CONFIDENCE")

    if pattern_stats:
        st.write(f"**Samples:** {pattern_stats['samples']}")
        st.write(f"**Win Rate:** {pattern_stats['win_rate']}%")
        st.write(f"**Avg Return:** {pattern_stats['avg_return']}%")
        st.write(f"**Avg Up:** {pattern_stats['avg_up']}%")
        st.write(f"**Avg Down:** {pattern_stats['avg_down']}%")
    else:
        st.warning("No strong historical match")

# -----------------------------
# AI ASSISTANT
# -----------------------------
st.subheader("AI Market Assistant")

user_input = st.text_input("Ask about this ticker:")

if user_input:
    insights = ai_market_analysis(
        final_bias,
        levels,
        fibs,
        options_data,
        mtf
    )

    for i in insights:
        st.write(f"- {i}")


st.subheader("Watchlist / Stage 14 Trading Terminal")

cols = [
    "Symbol", "Last", "Net Chg", "% Chg", "Bid", "Ask", "ATR",
    "Bull %", "Bear %", "Contract", "Timing",
    "Setup", "Trigger", "Mode", "Bias",
    "Score", "Action", "Rank Score"
]

watchlist = (
    st.session_state.watchlist_symbols["Symbol"]
    .dropna().astype(str).str.upper().str.strip().tolist()
)

rank_df = stage14_rank_engine(tuple(watchlist))

if rank_df.empty:
    rank_df = pd.DataFrame(columns=cols)

symbol_df = pd.DataFrame({"Symbol": watchlist + [""]})

terminal_df = symbol_df.merge(rank_df, on="Symbol", how="left")

for col in cols:
    if col not in terminal_df.columns:
        terminal_df[col] = None

terminal_df = terminal_df[cols].round(2)
terminal_df = terminal_df.sort_values(
    "Rank Score",
    ascending=False,
    na_position="last"
).reset_index(drop=True)



def style_top3(row):
    action = str(row["Action"])
    contract = str(row["Contract"])
    timing = str(row["Timing"])

    if row.name < 3:
        return ["background-color:#d4af37; color:black; font-weight:bold;"] * len(row)

    if action == "READY":
        bg = "#d9f2e3"
        text = "#063b1f"
    elif action == "AVOID":
        bg = "#f6d6d6"
        text = "#4a0000"
    elif action == "WAIT":
        bg = "#fff3cd"
        text = "#4a3b00"
    else:
        bg = "#eeeeee"
        text = "#222222"

    return [f"background-color:{bg}; color:{text};"] * len(row)
st.divider()
st.subheader("Add Ticker")

new_ticker = st.text_input("Ticker to add", key="add_ticker").upper().strip()

if st.button("Add Ticker"):
    if new_ticker:
        current = st.session_state.watchlist_symbols["Symbol"].tolist()

        if new_ticker not in current:
            current.append(new_ticker)

        st.session_state.watchlist_symbols = pd.DataFrame({"Symbol": current})
        save_watchlist_symbols(st.session_state.watchlist_symbols)
        st.cache_data.clear()
        st.rerun()

st.dataframe(
    terminal_df.style.apply(style_top3, axis=1),
    use_container_width=True,
    height=560,
    hide_index=True,
    column_config={
        "Last": st.column_config.NumberColumn(format="$%,.2f"),
        "Net Chg": st.column_config.NumberColumn(format="$%,.2f"),
        "% Chg": st.column_config.NumberColumn(format="%d%%"),
        "Bid": st.column_config.NumberColumn(format="$%,.2f"),
        "Ask": st.column_config.NumberColumn(format="$%,.2f"),
        "ATR": st.column_config.NumberColumn(format="$%,.2f"),
        "Bull %": st.column_config.NumberColumn(format="%d%%"),
        "Bear %": st.column_config.NumberColumn(format="%d%%"),
        "Score": st.column_config.NumberColumn(format="%.0f"),
        "Rank Score": st.column_config.NumberColumn(format="%.0f"),
    }
)