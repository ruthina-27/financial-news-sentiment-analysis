import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import yfinance as yf

ticker = 'AAPL'
data = yf.download(ticker, start='1998-01-01', end='2024-05-31')

# Check number of rows and date range
print(f"Data shape: {data.shape}")
print(f"Start date: {data.index.min()}")
print(f"End date: {data.index.max()}")

# Show last few rows
print(data.tail())
# Add indicators with pandas-ta
data['SMA20'] = ta.sma(data['Close'], length=20)
data['RSI'] = ta.rsi(data['Close'], length=14)
macd = ta.macd(data['Close'])
data = pd.concat([data, macd], axis=1)
data.tail()
import numpy as np

# Daily returns
data['Returns'] = data['Close'].pct_change()

# Annualized volatility (assuming 252 trading days/year)
volatility = np.std(data['Returns'].dropna()) * np.sqrt(252)
print(f"Annualized volatility: {volatility:.2%}")
import numpy as np

# Daily returns
data['Returns'] = data['Close'].pct_change()

# Annualized volatility (assuming 252 trading days/year)
volatility = np.std(data['Returns'].dropna()) * np.sqrt(252)
print(f"Annualized volatility: {volatility:.2%}")
import pandas_ta as ta

# This adds MACD columns directly into your DataFrame
data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)

# Now you should have these columns:
print([col for col in data.columns if 'MACD' in col])
import pandas_ta as ta
print(ta.version)
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt

# Download data
ticker = 'AAPL'
data = yf.download(ticker, start='1998-01-01', end='2024-05-31')

# Calculate indicators
data['SMA20'] = ta.sma(data['Close'], length=20)
data['RSI'] = ta.rsi(data['Close'], length=14)

# METHOD 1: Try the standard pandas_ta approach
try:
    data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    macd_cols = [col for col in data.columns if 'MACD' in col]
    if not macd_cols:
        raise ValueError("MACD columns not created")
        
except Exception as e:
    print(f"Standard MACD failed: {e}")
    # METHOD 2: Manual calculation fallback
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_line'] = ema12 - ema26
    data['MACD_signal'] = data['MACD_line'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD_line'] - data['MACD_signal']
    macd_cols = ['MACD_line', 'MACD_signal', 'MACD_hist']

# Verify columns
print("Available MACD columns:", macd_cols)

# Plotting
plt.figure(figsize=(14, 10))

# Price and SMA
plt.subplot(3, 1, 1)
plt.plot(data['Close'], label='Close Price')
plt.plot(data['SMA20'], label='SMA 20')
plt.title(f'{ticker} Close Price & SMA20')
plt.legend()

# RSI
plt.subplot(3, 1, 2)
plt.plot(data['RSI'], label='RSI')
plt.axhline(70, color='red', linestyle='--', alpha=0.5)
plt.axhline(30, color='green', linestyle='--', alpha=0.5)
plt.title('RSI (14-day)')
plt.legend()

# MACD (using whichever columns exist)
plt.subplot(3, 1, 3)
if 'MACD_12_26_9' in data.columns:
    plt.plot(data['MACD_12_26_9'], label='MACD', color='blue')
    plt.plot(data['MACDs_12_26_9'], label='Signal', color='orange')
    plt.bar(data.index, data['MACDh_12_26_9'], label='Histogram', color='gray', alpha=0.5)
elif 'MACD_line' in data.columns:
    plt.plot(data['MACD_line'], label='MACD', color='blue')
    plt.plot(data['MACD_signal'], label='Signal', color='orange')
    plt.bar(data.index, data['MACD_hist'], label='Histogram', color='gray', alpha=0.5)
else:
    print("No valid MACD columns found")
plt.title('MACD (12,26,9)')
plt.legend()

plt.tight_layout()
plt.show()
