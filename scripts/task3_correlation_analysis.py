import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import matplotlib.pyplot as plt

print("🔁 Starting correlation analysis...")

# 1. Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# 2. Load news data (replace with your actual news loading code)
print("📄 Loading news...")
try:
    news_df = pd.read_csv("data/cleaned_news.csv")
    news_df['date'] = pd.to_datetime(news_df['date']).dt.date
except FileNotFoundError:
    print("❌ Error: 'data/cleaned_news.csv' not found")
    exit()

# 3. Perform sentiment analysis
print("🧠 Performing sentiment analysis...")
analyzer = SentimentIntensityAnalyzer()
news_df['sentiment'] = news_df['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()

# 4. Download and prepare stock data
print("📈 Downloading stock data...")
try:
    stock_df = yf.download("AAPL", start="2024-01-01", end="2024-12-31", auto_adjust=False)
    
    # Convert MultiIndex columns to single level
    stock_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stock_df.columns]
    stock_df = stock_df.reset_index()
    
    # Prepare columns
    stock_df['date'] = pd.to_datetime(stock_df['Date_']).dt.date
    stock_df['daily_return'] = stock_df['Adj Close_AAPL'].pct_change()
    
    # 5. Merge datasets
    print("\n🔗 Merging sentiment with stock returns...")
    merged_df = pd.merge(
        daily_sentiment, 
        stock_df[['date', 'daily_return']], 
        on='date', 
        how='inner'
    )

    if merged_df.empty:
        print("⚠️ Warning: No matching dates between news and stock data")
        print("News date range:", daily_sentiment['date'].min(), "to", daily_sentiment['date'].max())
        print("Stock date range:", stock_df['date'].min(), "to", stock_df['date'].max())
    else:
        # 6. Correlation analysis
        correlation = merged_df['sentiment'].corr(merged_df['daily_return'])
        print(f"\n✅ Pearson correlation: {correlation:.4f}")

        # 7. Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(merged_df['sentiment'], merged_df['daily_return'], alpha=0.6)
        plt.title("Sentiment vs. Daily Returns (AAPL)")
        plt.xlabel("Average Daily Sentiment Score")
        plt.ylabel("Daily Return (Adj Close)")
        plt.grid(True)
        
        # Add regression line
        m, b = np.polyfit(merged_df['sentiment'], merged_df['daily_return'], 1)
        plt.plot(merged_df['sentiment'], m*merged_df['sentiment'] + b, color='red')
        
        plt.tight_layout()
        plt.savefig("outputs/sentiment_vs_return.png")
        plt.show()

        # 8. Save results
        merged_df.to_csv("outputs/merged_sentiment_stock.csv", index=False)
        print("\n💾 Saved results to outputs/merged_sentiment_stock.csv")

except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    if 'stock_df' in locals():
        print("\nStock data sample:")
        print(stock_df.head())