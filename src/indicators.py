import pandas as pd
import pandas_ta as ta

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators (SMA, RSI, MACD) to the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Stock price DataFrame with a 'Close' column.
    
    Returns:
        pd.DataFrame: DataFrame with new indicator columns added.
    """
    # Simple Moving Average - 20 days
    df['SMA_20'] = ta.sma(df['Close'], length=20)

    # Relative Strength Index - 14 days
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # MACD - Moving Average Convergence Divergence
    macd = ta.macd(df['Close'])  # Returns MACD, Signal, Histogram
    df = pd.concat([df, macd], axis=1)

    return df
