"""
File to map the names of assets to user-friendly ones, used in the dashboard for display.
Several helpers are also here as well
"""

#Name, asset class and one-line description for every asset, keyed by the ticker used in the files
ASSETS = {
    "BTCUSDT": ("Bitcoin", "Crypto", "Largest cryptocurrency, priced in USDT (a dollar-pegged token)"),
    "ETHUSDT": ("Ethereum", "Crypto", "Second-largest cryptocurrency, used for smart contracts"),
    "BNBUSDT": ("BNB", "Crypto", "Native token of the Binance exchange"),
    "SOLUSDT": ("Solana", "Crypto", "Cryptocurrency of the Solana blockchain"),
    "XRPUSDT": ("XRP", "Crypto", "Cryptocurrency used for cross-border payments (Ripple)"),
    "AAPL": ("Apple", "Equity", "Apple Inc. stock, consumer electronics and software"),
    "NVDA": ("NVIDIA", "Equity", "NVIDIA Corp. stock, graphics and AI chips"),
    "JPM": ("JPMorgan Chase", "Equity", "Largest US bank by assets"),
    "JNJ": ("Johnson & Johnson", "Equity", "Healthcare and pharmaceutical company"),
    "XOM": ("Exxon Mobil", "Equity", "Large oil and gas company"),
    "SPY": ("S&P 500 ETF", "ETF", "Fund tracking the 500 largest US companies"),
    "QQQ": ("Nasdaq-100 ETF", "ETF", "Fund tracking the 100 largest Nasdaq companies, tech-heavy"),
    "EEM": ("Emerging Markets ETF", "ETF", "Fund tracking stocks of emerging-market countries"),
    "GLD": ("Gold ETF", "ETF", "Fund backed by physical gold"),
    "TLT": ("Long-term Treasury ETF", "ETF", "Fund of US government bonds with 20+ years to maturity"),
    "EURUSD_X": ("EUR/USD", "Currency", "Euro against the US dollar"),
    "GBPUSD_X": ("GBP/USD", "Currency", "British pound against the US dollar"),
    "USDJPY_X": ("USD/JPY", "Currency", "US dollar against the Japanese yen"),
    "USDCHF_X": ("USD/CHF", "Currency", "US dollar against the Swiss franc"),
    "AUDUSD_X": ("AUD/USD", "Currency", "Australian dollar against the US dollar"),
    "CL_F": ("Crude oil", "Commodity", "WTI crude oil futures"),
    "NG_F": ("Natural gas", "Commodity", "US natural gas futures"),
    "GC_F": ("Gold", "Commodity", "Gold futures"),
    "SI_F": ("Silver", "Commodity", "Silver futures"),
    "HG_F": ("Copper", "Commodity", "Copper futures"),
}


#Method to give the text shown in the dropdowns for a ticker
def label(ticker):
    name = ASSETS.get(ticker, (ticker,))[0]
    return f"{name} ({ticker})"


#Method to give the one-line description shown under the asset dropdown
def describe(ticker):
    if ticker not in ASSETS:
        return ""
    return f"{ASSETS[ticker][1]}: {ASSETS[ticker][2]}"


#Method to sort the tickers by asset class and then by name
def sort_key(ticker):
    info = ASSETS.get(ticker, (ticker, "", ""))
    return (info[1], info[0])