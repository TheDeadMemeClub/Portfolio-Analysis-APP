
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Investment Analysis App v7", page_icon="📈", layout="wide")


# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.subtle {
    color: #9aa0a6;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 0.4rem;
    margin-bottom: 0.4rem;
}
.signal-bullish {
    padding: 0.75rem 1rem;
    border-radius: 0.9rem;
    background: rgba(0, 180, 80, 0.15);
    border: 1px solid rgba(0, 180, 80, 0.4);
    font-weight: 700;
}
.signal-bearish {
    padding: 0.75rem 1rem;
    border-radius: 0.9rem;
    background: rgba(220, 50, 50, 0.15);
    border: 1px solid rgba(220, 50, 50, 0.4);
    font-weight: 700;
}
.signal-neutral {
    padding: 0.75rem 1rem;
    border-radius: 0.9rem;
    background: rgba(255, 170, 0, 0.12);
    border: 1px solid rgba(255, 170, 0, 0.35);
    font-weight: 700;
}
.small-note {
    font-size: 0.88rem;
    color: #9aa0a6;
}
.value-bullish {
    padding: 0.5rem 0.8rem;
    border-radius: 0.8rem;
    background: rgba(0, 180, 80, 0.15);
    border: 1px solid rgba(0, 180, 80, 0.4);
    font-weight: 700;
    display: inline-block;
}
.value-bearish {
    padding: 0.5rem 0.8rem;
    border-radius: 0.8rem;
    background: rgba(220, 50, 50, 0.15);
    border: 1px solid rgba(220, 50, 50, 0.4);
    font-weight: 700;
    display: inline-block;
}
.value-neutral {
    padding: 0.5rem 0.8rem;
    border-radius: 0.8rem;
    background: rgba(255, 170, 0, 0.12);
    border: 1px solid rgba(255, 170, 0, 0.35);
    font-weight: 700;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
def clean_downloaded_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = [c[0] for c in df.columns]
        except Exception:
            df.columns = [str(c[0]) for c in df.columns]

    df = df.rename(columns=str.title)
    if "Close" in df.columns and "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    return df.dropna(how="all")


@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    ticker = ticker.strip().upper()

    attempts = []

    # Attempt 1: standard download
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        df = clean_downloaded_df(df)
        if not df.empty and "Adj Close" in df.columns:
            return df
        attempts.append("download_auto_adjust_false")
    except Exception:
        attempts.append("download_auto_adjust_false_failed")

    # Attempt 2: adjusted download
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        df = clean_downloaded_df(df)
        if not df.empty and "Adj Close" in df.columns:
            return df
        attempts.append("download_auto_adjust_true")
    except Exception:
        attempts.append("download_auto_adjust_true_failed")

    # Attempt 3: Ticker.history
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=False)
        df = clean_downloaded_df(df)
        if not df.empty and "Adj Close" in df.columns:
            return df
        attempts.append("history_auto_adjust_false")
    except Exception:
        attempts.append("history_auto_adjust_false_failed")

    # Attempt 4: Ticker.history adjusted
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=True)
        df = clean_downloaded_df(df)
        if not df.empty and "Adj Close" in df.columns:
            return df
        attempts.append("history_auto_adjust_true")
    except Exception:
        attempts.append("history_auto_adjust_true_failed")

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_stock_info(ticker: str) -> dict:
    tk = yf.Ticker(ticker.strip().upper())
    merged = {}

    for attr in ["info", "fast_info"]:
        try:
            obj = getattr(tk, attr)
            data = obj if isinstance(obj, dict) else dict(obj)
            for k, v in data.items():
                if k not in merged or merged.get(k) in [None, "", np.nan]:
                    merged[k] = v
        except Exception:
            pass

    return merged


@st.cache_data(show_spinner=False)
def load_ticker_object(ticker: str):
    return yf.Ticker(ticker.strip().upper())


@st.cache_data(show_spinner=False)
def compute_beta_vs_market(ticker: str, market: str = "SPY", period: str = "2y") -> float:
    stock = load_price_data(ticker, period=period)
    mkt = load_price_data(market, period=period)

    if stock.empty or mkt.empty or "Adj Close" not in stock.columns or "Adj Close" not in mkt.columns:
        return np.nan

    stock_ret = stock["Adj Close"].pct_change().dropna()
    mkt_ret = mkt["Adj Close"].pct_change().dropna()
    joined = pd.concat([stock_ret.rename("stock"), mkt_ret.rename("market")], axis=1).dropna()

    if len(joined) < 30:
        return np.nan

    cov = np.cov(joined["stock"], joined["market"])[0, 1]
    var = np.var(joined["market"])
    return np.nan if var == 0 else cov / var


@st.cache_data(show_spinner=False)
def load_multi_close(tickers, period="3y"):
    frames = []
    for t in tickers:
        df = load_price_data(t, period=period)
        if not df.empty and "Adj Close" in df.columns:
            frames.append(df["Adj Close"].rename(t))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="all")


def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"] = df["Adj Close"].rolling(5).mean()
    df["MA10"] = df["Adj Close"].rolling(10).mean()
    df["MA20"] = df["Adj Close"].rolling(20).mean()
    df["MA50"] = df["Adj Close"].rolling(50).mean()
    df["MA100"] = df["Adj Close"].rolling(100).mean()
    df["MA200"] = df["Adj Close"].rolling(200).mean()
    df["Returns"] = df["Adj Close"].pct_change()
    return df


def annualized_return_and_vol(returns: pd.Series):
    mean_daily = returns.mean()
    vol_daily = returns.std()
    ann_return = (1 + mean_daily) ** 252 - 1
    ann_vol = vol_daily * np.sqrt(252)
    return ann_return, ann_vol


def max_drawdown(price_series: pd.Series):
    rolling_max = price_series.cummax()
    drawdown = price_series / rolling_max - 1
    return drawdown.min(), drawdown


def calculate_var_cvar(returns: pd.Series, confidence: float):
    if returns.empty:
        return np.nan, np.nan
    percentile = 100 * (1 - confidence)
    var = np.percentile(returns, percentile)
    tail = returns[returns <= var]
    cvar = tail.mean() if len(tail) > 0 else np.nan
    return var, cvar


def technical_trend_label(latest_price, ma5, ma10, ma20):
    if pd.isna(latest_price) or pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20):
        return "Insufficient data"
    if latest_price > ma10 and ma5 > ma10 > ma20:
        return "Bullish"
    if latest_price < ma5 and latest_price < ma10 and latest_price < ma20 and ma5 < ma10 < ma20:
        return "Bearish"
    return "Not Trending"


def signal_box(label: str):
    if label == "Bullish":
        st.markdown(f'<div class="signal-bullish">Trend Signal: {label}</div>', unsafe_allow_html=True)
    elif label == "Bearish":
        st.markdown(f'<div class="signal-bearish">Trend Signal: {label}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="signal-neutral">Trend Signal: {label}</div>', unsafe_allow_html=True)


def valuation_tag(implied_price, current_price):
    if pd.isna(implied_price) or pd.isna(current_price) or current_price == 0:
        return "No Data", "value-neutral"
    pct_diff = implied_price / current_price - 1
    if pct_diff >= 0.15:
        return "Undervalued", "value-bullish"
    if pct_diff <= -0.15:
        return "Overvalued", "value-bearish"
    return "Fairly Valued", "value-neutral"


def render_value_tag(label, css_class):
    st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)


def blended_fair_value(values, weights):
    valid_values = []
    valid_weights = []
    for value, weight in zip(values, weights):
        if not pd.isna(value):
            valid_values.append(value)
            valid_weights.append(weight)
    if not valid_values:
        return np.nan
    valid_weights = np.array(valid_weights, dtype=float)
    total = valid_weights.sum()
    if total <= 0:
        valid_weights = np.repeat(1 / len(valid_values), len(valid_values))
    else:
        valid_weights = valid_weights / total
    return float(np.dot(valid_values, valid_weights))


def analyst_summary_text(current_price, blended_value, trend_label):
    if pd.isna(blended_value) or pd.isna(current_price) or current_price == 0:
        valuation_view = "insufficient valuation data"
    else:
        pct_diff = blended_value / current_price - 1
        if pct_diff >= 0.15:
            valuation_view = f"the blended fair value suggests upside of about {pct_diff:.1%}"
        elif pct_diff <= -0.15:
            valuation_view = f"the blended fair value suggests downside of about {abs(pct_diff):.1%}"
        else:
            valuation_view = f"the blended fair value is close to the current price, with a gap of about {abs(pct_diff):.1%}"

    if trend_label == "Bullish":
        trend_view = "The technical trend is bullish."
    elif trend_label == "Bearish":
        trend_view = "The technical trend is bearish."
    else:
        trend_view = "The technical trend is not strongly trending."

    return f"{trend_view} On valuation, {valuation_view}."


def safe_get(info: dict, key: str, default=np.nan):
    value = info.get(key, default)
    if value is None:
        return default
    return value


def format_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2%}"


def format_num(x):
    return "N/A" if pd.isna(x) else f"{x:,.2f}"


def relative_pe_value(eps, target_pe):
    if pd.isna(eps) or eps <= 0 or pd.isna(target_pe) or target_pe <= 0:
        return np.nan
    return eps * target_pe


def intrinsic_value_gordon(dividend_rate, required_return, growth_rate):
    if pd.isna(dividend_rate) or dividend_rate <= 0:
        return np.nan
    if pd.isna(required_return) or required_return <= growth_rate:
        return np.nan
    d1 = dividend_rate * (1 + growth_rate)
    return d1 / (required_return - growth_rate)


def get_statement_value(df: pd.DataFrame, label_candidates):
    if df is None or len(df) == 0:
        return np.nan
    lower_index_map = {str(idx).strip().lower(): idx for idx in df.index}
    for candidate in label_candidates:
        key = candidate.strip().lower()
        if key in lower_index_map:
            row_label = lower_index_map[key]
            row = df.loc[row_label]
            if isinstance(row, pd.Series):
                valid = row.dropna()
                if len(valid) > 0:
                    return float(valid.iloc[0])
    return np.nan


def get_better_fundamentals(ticker: str, info: dict) -> dict:
    tk = load_ticker_object(ticker)

    result = {
        "trailing_eps": safe_get(info, "trailingEps", np.nan),
        "current_pe": safe_get(info, "trailingPE", np.nan),
        "forward_pe": safe_get(info, "forwardPE", np.nan),
        "dividend_rate": safe_get(info, "dividendRate", np.nan),
        "shares_outstanding": safe_get(info, "sharesOutstanding", np.nan),
        "free_cash_flow": safe_get(info, "freeCashflow", np.nan),
        "market_cap": safe_get(info, "marketCap", np.nan),
        "enterprise_value": safe_get(info, "enterpriseValue", np.nan),
        "total_debt": safe_get(info, "totalDebt", 0.0),
        "total_cash": safe_get(info, "totalCash", 0.0),
        "book_value": safe_get(info, "bookValue", np.nan),
        "price_to_book": safe_get(info, "priceToBook", np.nan),
        "analyst_target_price": safe_get(info, "targetMeanPrice", np.nan),
    }

    # Shares outstanding fallback
    try:
        fast_info = tk.fast_info
        if pd.isna(result["shares_outstanding"]):
            result["shares_outstanding"] = fast_info.get("shares")
        if pd.isna(result["market_cap"]):
            result["market_cap"] = fast_info.get("market_cap")
    except Exception:
        pass

    # Financial statements fallbacks
    quarterly_cf = pd.DataFrame()
    annual_cf = pd.DataFrame()
    quarterly_bs = pd.DataFrame()
    annual_bs = pd.DataFrame()
    quarterly_is = pd.DataFrame()
    annual_is = pd.DataFrame()

    for attr_name, var_name in [
        ("quarterly_cashflow", "quarterly_cf"),
        ("cashflow", "annual_cf"),
        ("quarterly_balance_sheet", "quarterly_bs"),
        ("balance_sheet", "annual_bs"),
        ("quarterly_income_stmt", "quarterly_is"),
        ("income_stmt", "annual_is"),
    ]:
        try:
            locals()[var_name] = getattr(tk, attr_name)
        except Exception:
            pass

    # FCF fallback: operating cash flow - capex
    if pd.isna(result["free_cash_flow"]):
        ocf = get_statement_value(
            quarterly_cf,
            ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"]
        )
        capex = get_statement_value(
            quarterly_cf,
            ["Capital Expenditure", "Capital Expenditures"]
        )
        if pd.isna(ocf):
            ocf = get_statement_value(
                annual_cf,
                ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"]
            )
            capex = get_statement_value(
                annual_cf,
                ["Capital Expenditure", "Capital Expenditures"]
            )
        if not pd.isna(ocf) and not pd.isna(capex):
            result["free_cash_flow"] = ocf - abs(capex)

    # Debt/cash fallback
    if pd.isna(result["total_debt"]) or result["total_debt"] == 0:
        td = get_statement_value(quarterly_bs, ["Total Debt", "Current Debt And Capital Lease Obligation"])
        if pd.isna(td):
            td = get_statement_value(annual_bs, ["Total Debt", "Current Debt And Capital Lease Obligation"])
        if not pd.isna(td):
            result["total_debt"] = td

    if pd.isna(result["total_cash"]) or result["total_cash"] == 0:
        cash = get_statement_value(quarterly_bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
        if pd.isna(cash):
            cash = get_statement_value(annual_bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
        if not pd.isna(cash):
            result["total_cash"] = cash

    # EPS fallback using Net Income / shares
    if pd.isna(result["trailing_eps"]) and not pd.isna(result["shares_outstanding"]) and result["shares_outstanding"] > 0:
        ni = get_statement_value(quarterly_is, ["Net Income", "Net Income Common Stockholders"])
        if pd.isna(ni):
            ni = get_statement_value(annual_is, ["Net Income", "Net Income Common Stockholders"])
        if not pd.isna(ni):
            result["trailing_eps"] = ni / result["shares_outstanding"]

    return result


def dcf_share_value(fcf, discount_rate, terminal_growth, shares_outstanding, net_debt=0.0, years=5, growth_rate=0.08):
    if pd.isna(fcf) or fcf <= 0:
        return np.nan
    if pd.isna(discount_rate) or discount_rate <= terminal_growth or discount_rate <= -0.99:
        return np.nan
    if pd.isna(shares_outstanding) or shares_outstanding <= 0:
        return np.nan

    projected_fcfs = []
    current_fcf = float(fcf)
    for year in range(1, years + 1):
        current_fcf *= (1 + growth_rate)
        projected_fcfs.append(current_fcf / ((1 + discount_rate) ** year))

    terminal_fcf = current_fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)

    enterprise_value = sum(projected_fcfs) + pv_terminal
    equity_value = enterprise_value - net_debt
    return equity_value / shares_outstanding


def pb_valuation(current_price, book_value_per_share, target_pb_multiple):
    if pd.isna(current_price) or pd.isna(book_value_per_share) or book_value_per_share <= 0 or pd.isna(target_pb_multiple):
        return np.nan
    return book_value_per_share * target_pb_multiple


def normalize_weights(weights):
    arr = np.array(weights, dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.repeat(1 / len(arr), len(arr))
    return arr / total


def portfolio_stats(returns_df: pd.DataFrame, weights: np.ndarray, rf_rate: float):
    weights = np.asarray(weights, dtype=float)
    mean_daily = returns_df.mean()
    cov_daily = returns_df.cov()
    port_daily_return = np.dot(weights, mean_daily)
    port_ann_return = (1 + port_daily_return) ** 252 - 1
    port_ann_vol = np.sqrt(np.dot(weights.T, np.dot(cov_daily * 252, weights)))
    sharpe = (port_ann_return - rf_rate) / port_ann_vol if port_ann_vol != 0 else np.nan
    return port_ann_return, port_ann_vol, sharpe


def build_portfolio_growth(returns_df: pd.DataFrame, weights: np.ndarray, initial_value: float):
    port_daily = returns_df.dot(weights)
    growth = initial_value * (1 + port_daily).cumprod()
    return growth


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Controls")

with st.sidebar.expander("Primary Inputs", expanded=True):
    ticker = st.text_input("Primary ticker", value="AAPL").upper().strip()
    benchmark = st.text_input("Benchmark", value="SPY").upper().strip()

    period = st.selectbox(
        "Stock history period",
        ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"],
        index=3
    )

    custom_period = st.text_input(
        "Optional custom lookback (override)",
        value="",
        help="Examples: 1mo, 2mo, 3mo, 6mo, 18mo, 2y, 5y, max"
    ).strip()
    if custom_period:
        period = custom_period

with st.sidebar.expander("Valuation Inputs", expanded=True):
    rf_rate = st.slider("Risk-free rate (%)", 0.0, 10.0, 4.0, 0.1) / 100
    market_premium = st.slider("Market risk premium (%)", 1.0, 12.0, 5.5, 0.1) / 100
    growth_rate = st.slider("Dividend growth rate (%)", 0.0, 12.0, 4.0, 0.1) / 100
    dcf_growth_rate = st.slider("FCF growth rate (%)", 0.0, 25.0, 8.0, 0.5) / 100
    terminal_growth_rate = st.slider("Terminal growth rate (%)", 0.0, 6.0, 3.0, 0.1) / 100
    dcf_years = st.slider("DCF forecast years", 3, 10, 5, 1)
    target_pe = st.slider("Target P/E", 5.0, 50.0, 18.0, 0.5)
    target_pb = st.slider("Target P/B", 0.5, 15.0, 3.0, 0.1)
    relative_pe_weight = st.slider("Relative P/E weight", 0.0, 1.0, 0.30, 0.05)
    ddm_weight = st.slider("DDM weight", 0.0, 1.0, 0.15, 0.05)
    dcf_weight = st.slider("DCF weight", 0.0, 1.0, 0.35, 0.05)
    pb_weight = st.slider("P/B weight", 0.0, 1.0, 0.10, 0.05)
    analyst_target_weight = st.slider("Analyst target weight", 0.0, 1.0, 0.10, 0.05)
    var_conf = st.slider("Historical VaR confidence", 0.90, 0.99, 0.95, 0.01)

with st.sidebar.expander("Chart Settings", expanded=True):
    show_ma5 = st.checkbox("Show MA 5", value=True)
    show_ma10 = st.checkbox("Show MA 10", value=True)
    show_ma20 = st.checkbox("Show MA 20", value=True)
    show_ma50 = st.checkbox("Show MA 50", value=False)
    show_ma100 = st.checkbox("Show MA 100", value=True)
    show_ma200 = st.checkbox("Show MA 200", value=True)
    show_volume = st.checkbox("Show volume in data table", value=True)
    recent_candles = st.slider("Recent candles in technical chart", 20, 120, 40, 5)

with st.sidebar.expander("Scanner", expanded=False):
    watchlist_text = st.text_area(
        "Watchlist scanner (comma separated)",
        value="AAPL,MSFT,NVDA,TSLA,AMD",
        height=110
    )

st.sidebar.markdown("---")
st.sidebar.caption("Trend rule:")
st.sidebar.caption("Bullish = Price > 10 MA and 5MA > 10MA > 20MA")
st.sidebar.caption("Bearish = Price < 5, 10, 20 MA and 5MA < 10MA < 20MA")
st.sidebar.caption("Else = Not Trending")


# -----------------------------
# Main
# -----------------------------
st.markdown('<div class="main-header">Investment and Finance App v7</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">More robust Yahoo data handling, improved valuation inputs, and better fallback logic for price and fundamentals.</div>',
    unsafe_allow_html=True
)

if not ticker:
    st.warning("Enter a ticker to begin.")
    st.stop()

price_df = load_price_data(ticker, period=period)
info = load_stock_info(ticker)

if price_df.empty or "Adj Close" not in price_df.columns:
    st.error("No market data was returned for that ticker.")
    st.stop()

price_df = add_mas(price_df)
fund = get_better_fundamentals(ticker, info)

latest = price_df.iloc[-1]
latest_price = latest["Adj Close"]
ma5 = latest["MA5"]
ma10 = latest["MA10"]
ma20 = latest["MA20"]
trend_label = technical_trend_label(latest_price, ma5, ma10, ma20)

returns = price_df["Returns"].dropna()
ann_return, ann_vol = annualized_return_and_vol(returns)
mdd, drawdown_series = max_drawdown(price_df["Adj Close"])
var_daily, cvar_daily = calculate_var_cvar(returns, var_conf)

beta_info = safe_get(info, "beta", np.nan)
beta_calc = compute_beta_vs_market(ticker, benchmark, "2y")
beta_used = beta_info if not pd.isna(beta_info) else beta_calc
required_return = rf_rate + beta_used * market_premium if not pd.isna(beta_used) else np.nan
sharpe = (ann_return - rf_rate) / ann_vol if ann_vol and not pd.isna(ann_vol) and ann_vol != 0 else np.nan

dividend_rate = fund["dividend_rate"]
trailing_eps = fund["trailing_eps"]
current_pe = fund["current_pe"]
forward_pe = fund["forward_pe"]
book_value = fund["book_value"]
price_to_book = fund["price_to_book"]
analyst_target_price = fund["analyst_target_price"]

ddm_value = intrinsic_value_gordon(dividend_rate, required_return, growth_rate)
relative_value = relative_pe_value(trailing_eps, target_pe)

net_debt = (fund["total_debt"] if not pd.isna(fund["total_debt"]) else 0.0) - (fund["total_cash"] if not pd.isna(fund["total_cash"]) else 0.0)
dcf_value = dcf_share_value(
    fcf=fund["free_cash_flow"],
    discount_rate=required_return,
    terminal_growth=terminal_growth_rate,
    shares_outstanding=fund["shares_outstanding"],
    net_debt=net_debt,
    years=dcf_years,
    growth_rate=dcf_growth_rate,
)
pb_value = pb_valuation(latest_price, book_value, target_pb)

blended_value = blended_fair_value(
    [relative_value, ddm_value, dcf_value, pb_value, analyst_target_price],
    [relative_pe_weight, ddm_weight, dcf_weight, pb_weight, analyst_target_weight]
)
blended_label, blended_class = valuation_tag(blended_value, latest_price)
summary_text = analyst_summary_text(latest_price, blended_value, trend_label)

company_name = safe_get(info, "longName", ticker)
sector = safe_get(info, "sector", "N/A")
industry = safe_get(info, "industry", "N/A")
market_cap = fund["market_cap"]

top1, top2, top3, top4, top5 = st.columns(5)
top1.metric("Ticker", ticker)
top2.metric("Price", f"${latest_price:,.2f}")
top3.metric("CAPM Return", format_pct(required_return))
top4.metric("Beta", format_num(beta_used))
top5.metric("Sharpe", format_num(sharpe))

signal_box(trend_label)
st.markdown(f'<div class="small-note">{company_name} | {sector} | {industry}</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Overview", "Technical", "Valuation", "Risk", "Portfolio", "Watchlist Scanner", "Raw Data"]
)

def add_selected_mas(fig, df, x_col):
    ma_settings = [
        ("MA5", show_ma5, "MA 5"),
        ("MA10", show_ma10, "MA 10"),
        ("MA20", show_ma20, "MA 20"),
        ("MA50", show_ma50, "MA 50"),
        ("MA100", show_ma100, "MA 100"),
        ("MA200", show_ma200, "MA 200"),
    ]
    for col, enabled, name in ma_settings:
        if enabled and col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=x_col, y=df[col], mode="lines", name=name))

with tab1:
    c1, c2 = st.columns([2.2, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df["Adj Close"], mode="lines", name="Adj Close"))
        add_selected_mas(fig, price_df, price_df.index)
        fig.update_layout(title=f"{ticker} Price and Selected Moving Averages", height=520, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Quick Stats</div>', unsafe_allow_html=True)
        st.metric("Annualized Return", format_pct(ann_return))
        st.metric("Annualized Volatility", format_pct(ann_vol))
        st.metric("Max Drawdown", format_pct(mdd))
        st.metric("Market Cap", "N/A" if pd.isna(market_cap) else f"${market_cap:,.0f}")
        st.metric("Current P/E", "N/A" if pd.isna(current_pe) else f"{current_pe:,.2f}")
        st.metric("Forward P/E", "N/A" if pd.isna(forward_pe) else f"{forward_pe:,.2f}")
        st.metric("Dividend Rate", "N/A" if pd.isna(dividend_rate) else f"${dividend_rate:,.2f}")

    st.markdown(
        '<div class="small-note">This version uses more fallback paths for Yahoo price downloads and more statement-based fallbacks for valuation inputs.</div>',
        unsafe_allow_html=True
    )

with tab2:
    st.markdown('<div class="section-header">Technical Trend Classification</div>', unsafe_allow_html=True)

    tech_left, tech_right = st.columns([1, 1.4])
    with tech_left:
        decision_df = pd.DataFrame({
            "Metric": [
                "Latest Price", "5 Day MA", "10 Day MA", "20 Day MA",
                "Price > 5MA", "Price > 10MA", "Price > 20MA",
                "5MA > 10MA", "10MA > 20MA", "Bullish Rule Met", "Bearish Rule Met", "Final Label"
            ],
            "Value": [
                round(float(latest_price), 2),
                round(float(ma5), 2) if not pd.isna(ma5) else np.nan,
                round(float(ma10), 2) if not pd.isna(ma10) else np.nan,
                round(float(ma20), 2) if not pd.isna(ma20) else np.nan,
                bool(latest_price > ma5) if not pd.isna(ma5) else None,
                bool(latest_price > ma10) if not pd.isna(ma10) else None,
                bool(latest_price > ma20) if not pd.isna(ma20) else None,
                bool(ma5 > ma10) if not pd.isna(ma5) and not pd.isna(ma10) else None,
                bool(ma10 > ma20) if not pd.isna(ma10) and not pd.isna(ma20) else None,
                bool((latest_price > ma10) and (ma5 > ma10 > ma20)) if not pd.isna(ma5) and not pd.isna(ma10) and not pd.isna(ma20) else None,
                bool((latest_price < ma5) and (latest_price < ma10) and (latest_price < ma20) and (ma5 < ma10 < ma20)) if not pd.isna(ma5) and not pd.isna(ma10) and not pd.isna(ma20) else None,
                trend_label
            ]
        })
        st.dataframe(decision_df, use_container_width=True, hide_index=True)

    with tech_right:
        signal_box(trend_label)
        st.write(
            "A stock is **Bullish** when price is above the 10 day moving average "
            "and the moving averages are stacked in the order **5MA > 10MA > 20MA**. "
            "It is **Bearish** only when price is below the 5, 10, and 20 day moving averages and the reverse stack is true. "
            "Anything else is labeled **Not Trending**."
        )

        recent = price_df.tail(recent_candles)
        candle_fig = go.Figure()
        candle_fig.add_trace(go.Candlestick(
            x=recent.index,
            open=recent["Open"],
            high=recent["High"],
            low=recent["Low"],
            close=recent["Close"],
            name="OHLC"
        ))
        add_selected_mas(candle_fig, recent, recent.index)
        candle_fig.update_layout(height=520, title=f"{ticker} Recent Candles and Selected MAs", xaxis_rangeslider_visible=False)
        st.plotly_chart(candle_fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Valuation</div>', unsafe_allow_html=True)

    summary_col1, summary_col2, summary_col3 = st.columns([1.1, 1, 1.2])
    with summary_col1:
        st.metric("Current Price", f"${latest_price:,.2f}")
        st.metric("Blended Fair Value", "N/A" if pd.isna(blended_value) else f"${blended_value:,.2f}")
    with summary_col2:
        if not pd.isna(blended_value):
            st.metric("Blended % Difference", f"{(blended_value / latest_price - 1):.2%}")
        else:
            st.metric("Blended % Difference", "N/A")
        render_value_tag(blended_label, blended_class)
    with summary_col3:
        st.markdown("### Analyst-Style Summary")
        st.write(summary_text)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Risk-Free Rate", format_pct(rf_rate))
    v2.metric("Market Premium", format_pct(market_premium))
    v3.metric("Required Return", format_pct(required_return))
    v4.metric("Target P/E", format_num(target_pe))

    left, right = st.columns(2)
    with left:
        st.markdown("### CAPM Inputs")
        capm_df = pd.DataFrame({
            "Input": ["Risk-free rate", "Beta", "Market risk premium", "Required return"],
            "Value": [rf_rate, beta_used, market_premium, required_return]
        })
        st.dataframe(capm_df, use_container_width=True, hide_index=True)

        st.markdown("### Relative P/E")
        rel_label, rel_class = valuation_tag(relative_value, latest_price)
        render_value_tag(rel_label, rel_class)
        st.write(f"Model weight: **{relative_pe_weight:.0%}**")
        st.write(f"Trailing EPS: **{'N/A' if pd.isna(trailing_eps) else f'${trailing_eps:,.2f}'}**")
        st.write(f"Current trailing P/E: **{'N/A' if pd.isna(current_pe) else f'{current_pe:,.2f}'}**")
        st.write(f"Relative P/E value: **{'N/A' if pd.isna(relative_value) else f'${relative_value:,.2f}'}**")

        st.markdown("### Price / Book Model")
        pb_label, pb_class = valuation_tag(pb_value, latest_price)
        render_value_tag(pb_label, pb_class)
        st.write(f"Model weight: **{pb_weight:.0%}**")
        st.write(f"Book value per share: **{'N/A' if pd.isna(book_value) else f'${book_value:,.2f}'}**")
        st.write(f"Current P/B: **{'N/A' if pd.isna(price_to_book) else f'{price_to_book:,.2f}'}**")
        st.write(f"Target P/B: **{target_pb:,.2f}**")
        st.write(f"P/B implied price: **{'N/A' if pd.isna(pb_value) else f'${pb_value:,.2f}'}**")

    with right:
        st.markdown("### Dividend Discount Model")
        ddm_label, ddm_class = valuation_tag(ddm_value, latest_price)
        render_value_tag(ddm_label, ddm_class)
        st.write(f"Model weight: **{ddm_weight:.0%}**")
        st.write(f"Dividend rate: **{'N/A' if pd.isna(dividend_rate) else f'${dividend_rate:,.2f}'}**")
        st.write(f"Dividend growth assumption: **{growth_rate:.2%}**")
        st.write(f"DDM value: **{'N/A' if pd.isna(ddm_value) else f'${ddm_value:,.2f}'}**")

        st.markdown("### Free Cash Flow / DCF Model")
        dcf_label, dcf_class = valuation_tag(dcf_value, latest_price)
        render_value_tag(dcf_label, dcf_class)
        st.write(f"Model weight: **{dcf_weight:.0%}**")
        st.write(f"Free cash flow: **{'N/A' if pd.isna(fund['free_cash_flow']) else f'${fund['free_cash_flow']:,.0f}'}**")
        st.write(f"Shares outstanding: **{'N/A' if pd.isna(fund['shares_outstanding']) else f'{fund['shares_outstanding']:,.0f}'}**")
        st.write(f"FCF growth assumption: **{dcf_growth_rate:.2%}**")
        st.write(f"Terminal growth assumption: **{terminal_growth_rate:.2%}**")
        st.write(f"DCF value per share: **{'N/A' if pd.isna(dcf_value) else f'${dcf_value:,.2f}'}**")

        st.markdown("### Analyst Target Price")
        analyst_label, analyst_class = valuation_tag(analyst_target_price, latest_price)
        render_value_tag(analyst_label, analyst_class)
        st.write(f"Model weight: **{analyst_target_weight:.0%}**")
        st.write(f"Analyst target price: **{'N/A' if pd.isna(analyst_target_price) else f'${analyst_target_price:,.2f}'}**")

    st.markdown("### Model Weights")
    weights_df = pd.DataFrame({
        "Model": ["Relative P/E", "Dividend Discount Model", "Free Cash Flow / DCF", "Price / Book", "Analyst Target"],
        "Weight": [relative_pe_weight, ddm_weight, dcf_weight, pb_weight, analyst_target_weight]
    })
    st.dataframe(weights_df.style.format({"Weight": "{:.0%}"}), use_container_width=True)

    st.markdown("### Valuation Comparison vs Current Price")
    comparison_rows = []
    models = [
        ("Current Price", latest_price),
        ("Relative P/E", relative_value),
        ("Dividend Discount Model", ddm_value),
        ("Free Cash Flow / DCF", dcf_value),
        ("Price / Book", pb_value),
        ("Analyst Target", analyst_target_price),
        ("Blended Fair Value", blended_value),
    ]
    for name, value in models:
        difference = np.nan if pd.isna(value) else value - latest_price
        pct_diff = np.nan if pd.isna(value) or latest_price == 0 else (value / latest_price - 1)
        status, _ = valuation_tag(value, latest_price)
        comparison_rows.append({
            "Model": name,
            "Implied Price": value,
            "Dollar Difference": difference,
            "Percent Difference": pct_diff,
            "Status": status,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison_df.style.format({
            "Implied Price": "${:,.2f}",
            "Dollar Difference": "${:,.2f}",
            "Percent Difference": "{:.2%}",
        }),
        use_container_width=True
    )

    st.markdown('<div class="small-note">This version adds more valuation inputs with fallbacks from Yahoo statement data. Some companies will still show N/A on certain models if Yahoo does not expose enough data.</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-header">Risk Metrics</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Daily VaR", format_pct(var_daily))
    r2.metric("Daily CVaR", format_pct(cvar_daily))
    r3.metric("Annual Volatility", format_pct(ann_vol))
    r4.metric("Max Drawdown", format_pct(mdd))

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, mode="lines", name="Drawdown"))
    dd_fig.update_layout(height=470, title=f"{ticker} Drawdown History", xaxis_title="Date", yaxis_title="Drawdown")
    st.plotly_chart(dd_fig, use_container_width=True)

with tab5:
    st.markdown('<div class="section-header">Portfolio Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Enter up to 5 tickers and adjust weights. The app normalizes the weights automatically.</div>', unsafe_allow_html=True)

    pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
    pt1 = pcol1.text_input("Ticker 1", value="AAPL").upper().strip()
    pt2 = pcol2.text_input("Ticker 2", value="MSFT").upper().strip()
    pt3 = pcol3.text_input("Ticker 3", value="NVDA").upper().strip()
    pt4 = pcol4.text_input("Ticker 4", value="AMD").upper().strip()
    pt5 = pcol5.text_input("Ticker 5", value="SPY").upper().strip()

    wcol1, wcol2, wcol3, wcol4, wcol5 = st.columns(5)
    w1 = wcol1.slider("Weight 1", 0, 100, 25, 1)
    w2 = wcol2.slider("Weight 2", 0, 100, 25, 1)
    w3 = wcol3.slider("Weight 3", 0, 100, 20, 1)
    w4 = wcol4.slider("Weight 4", 0, 100, 15, 1)
    w5 = wcol5.slider("Weight 5", 0, 100, 15, 1)

    initial_value = st.number_input("Starting portfolio value ($)", min_value=1000, value=100000, step=1000)
    portfolio_period = st.selectbox("Portfolio history period", ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y"], index=5)

    tickers = [t for t in [pt1, pt2, pt3, pt4, pt5] if t]
    raw_weights = [w1, w2, w3, w4, w5][:len(tickers)]
    weights = normalize_weights(raw_weights)

    close_df = load_multi_close(tickers, period=portfolio_period)
    if not close_df.empty:
        returns_df = close_df.pct_change().dropna()
        aligned_weights = normalize_weights(weights[:len(close_df.columns)])

        port_return, port_vol, port_sharpe = portfolio_stats(returns_df.dropna(), aligned_weights, rf_rate)
        port_growth = build_portfolio_growth(returns_df.dropna(), aligned_weights, initial_value)
        port_mdd, port_dd = max_drawdown(port_growth)

        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Portfolio Ann. Return", format_pct(port_return))
        pm2.metric("Portfolio Ann. Vol", format_pct(port_vol))
        pm3.metric("Portfolio Sharpe", format_num(port_sharpe))
        pm4.metric("Portfolio Max Drawdown", format_pct(port_mdd))

        weights_df = pd.DataFrame({
            "Ticker": close_df.columns,
            "Normalized Weight": aligned_weights
        })
        st.dataframe(weights_df, use_container_width=True, hide_index=True)

        port_fig = go.Figure()
        norm_prices = close_df / close_df.iloc[0] * initial_value
        for col in norm_prices.columns:
            port_fig.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[col], mode="lines", name=col))
        port_fig.add_trace(go.Scatter(x=port_growth.index, y=port_growth, mode="lines", name="Portfolio"))
        port_fig.update_layout(height=520, title="Portfolio vs Components", xaxis_title="Date", yaxis_title="Growth of Initial Value")
        st.plotly_chart(port_fig, use_container_width=True)

        st.markdown("### Correlation Matrix")
        st.dataframe(returns_df.corr().style.format("{:.2f}"), use_container_width=True)
    else:
        st.warning("Portfolio data could not be loaded for the selected tickers.")

with tab6:
    st.markdown('<div class="section-header">Watchlist Trend Scanner</div>', unsafe_allow_html=True)
    watchlist = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]

    scanner_period = st.selectbox("Scanner lookback", ["1mo", "3mo", "6mo", "1y"], index=2)

    results = []
    for sym in watchlist:
        df = load_price_data(sym, period=scanner_period)
        if df.empty or "Adj Close" not in df.columns:
            results.append({"Ticker": sym, "Price": np.nan, "5MA": np.nan, "10MA": np.nan, "20MA": np.nan, "Trend": "No Data"})
            continue

        df = add_mas(df)
        last = df.iloc[-1]
        label = technical_trend_label(last["Adj Close"], last["MA5"], last["MA10"], last["MA20"])

        results.append({
            "Ticker": sym,
            "Price": round(float(last["Adj Close"]), 2),
            "5MA": round(float(last["MA5"]), 2) if not pd.isna(last["MA5"]) else np.nan,
            "10MA": round(float(last["MA10"]), 2) if not pd.isna(last["MA10"]) else np.nan,
            "20MA": round(float(last["MA20"]), 2) if not pd.isna(last["MA20"]) else np.nan,
            "Trend": label
        })

    scanner_df = pd.DataFrame(results)
    if not scanner_df.empty:
        trend_filter = st.multiselect(
            "Filter scanner by trend",
            ["Bullish", "Bearish", "Not Trending", "No Data"],
            default=["Bullish", "Bearish", "Not Trending", "No Data"]
        )
        scanner_df = scanner_df[scanner_df["Trend"].isin(trend_filter)]

    st.dataframe(scanner_df, use_container_width=True, hide_index=True)

    bullish_count = (scanner_df["Trend"] == "Bullish").sum() if not scanner_df.empty else 0
    bearish_count = (scanner_df["Trend"] == "Bearish").sum() if not scanner_df.empty else 0
    neutral_count = (scanner_df["Trend"] == "Not Trending").sum() if not scanner_df.empty else 0

    s1, s2, s3 = st.columns(3)
    s1.metric("Bullish", int(bullish_count))
    s2.metric("Bearish", int(bearish_count))
    s3.metric("Not Trending", int(neutral_count))

    csv = scanner_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download scanner results as CSV", data=csv, file_name="watchlist_scanner.csv", mime="text/csv")

with tab7:
    st.markdown('<div class="section-header">Raw Data</div>', unsafe_allow_html=True)
    display_cols = ["Open", "High", "Low", "Close", "Adj Close", "MA5", "MA10", "MA20", "MA50", "MA100", "MA200", "Returns"]
    if show_volume:
        display_cols.insert(5, "Volume")
    available_cols = [c for c in display_cols if c in price_df.columns]
    rows_to_show = st.slider("Rows to display", 20, 250, 60, 10)
    st.dataframe(price_df[available_cols].tail(rows_to_show), use_container_width=True)

st.markdown("---")
st.caption("Educational use only. Data may contain errors or delays. This app is not investment advice.")
