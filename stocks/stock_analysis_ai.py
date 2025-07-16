
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
import requests
from bs4 import BeautifulSoup
import time
import random
from openai import AzureOpenAI, OpenAI
import base64
from matplotlib import ticker
from torch import mode
import websockets
import asyncio
import yfinance as yf
import numpy as np
from scipy.signal import argrelextrema

def query_openai_with_prompt_v1(prompt_text, image_path, api_key, endpoint, deployment_name):
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-06-01",
        azure_endpoint=endpoint,
    )
    # client = OpenAI(
    #     base_url = 'http://localhost:11434/v1',
    #     api_key='ollama', # required, but unused
    #     model='' # model name
    # )

    # קידוד תמונה ל־base64
    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # קריאה ל־chat completions
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a professional swing trader and technical analyst."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ],
        temperature=0.3,
        max_tokens=800
    )

    return response.choices[0].message.content


def fetch_data(ticker, period='1y'):
    # Use 2 years of data to ensure we have enough for 150-day moving average
    data = yf.download(ticker, period='2y', interval='1d')
    
    # Handle multi-index columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        # Extract the data for this ticker if it's multi-index
        if len(data.columns.levels) > 1 and ticker in data.columns.levels[1]:
            data = data.xs(ticker, level=1, axis=1)
        else:
            # If we can't extract by ticker, just flatten the columns
            data.columns = [col[1] if isinstance(col, tuple) else col for col in data.columns]
    
    # Make sure all price columns are numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop any rows with NaN values in the OHLC columns
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    # Calculate moving averages
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA150'] = data['Close'].rolling(window=150).mean()
    
    return data

def fit_trendline_on_extrema(df, column, direction='low'):
    """
    Fits a trendline on recent extrema points.
    direction: 'low' for support (higher lows), 'high' for resistance (lower highs)
    """
    extrema_func = np.less_equal if direction == 'low' else np.greater_equal
    check_valid = lambda arr: all(x < y for x, y in zip(arr, arr[1:])) if direction == 'low' else all(x > y for x, y in zip(arr, arr[1:]))

    extrema_idx = argrelextrema(df[column].values, extrema_func, order=5)[0]
    extrema_df = df.iloc[extrema_idx]

    for n in range(10, 1, -1):
        recent = extrema_df.tail(n)
        if len(recent) >= 2 and check_valid(recent[column]):
            x = recent.index.map(lambda d: df.index.get_loc(d)).values
            y = recent[column].values
            coef = np.polyfit(x, y, 1)
            trend = coef[0] * np.arange(len(df)) + coef[1]
            return pd.Series(trend[x[0]:], index=df.index[x[0]:])
    return None

def plot_chart_with_adaptive_trendlines(ticker='TSLA'):
    df = yf.download(ticker, period="1y", interval="1d", group_by='ticker')
    filename = f"{ticker}_trendlines_adaptive.png"
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.levels[0]:
            df = df[ticker]
        else:
            df.columns = df.columns.droplevel(0)

    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.dropna(subset=required, inplace=True)
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA150'] = df['Close'].rolling(window=150).mean()
    df['SupportTrend'] = np.nan
    df['ResistanceTrend'] = np.nan

    # Apply trendline fitting
    support = fit_trendline_on_extrema(df, 'Low', direction='low')
    if support is not None:
        df.loc[support.index, 'SupportTrend'] = support

    resistance = fit_trendline_on_extrema(df, 'High', direction='high')
    if resistance is not None:
        df.loc[resistance.index, 'ResistanceTrend'] = resistance

    # Plot chart
    addplots = []
    if df['SupportTrend'].notna().sum() > 0:
        addplots.append(mpf.make_addplot(df['SupportTrend'], color='blue', linestyle='--'))
    if df['ResistanceTrend'].notna().sum() > 0:
        addplots.append(mpf.make_addplot(df['ResistanceTrend'], color='red', linestyle='--'))

    mpf.plot(
        df,
        type='candle',
        style='yahoo',
        mav=(20, 150),
        volume=True,
        addplot=addplots,
        title=f"{ticker} with Adaptive Trendlines",
        tight_layout=True,
        savefig=dict(fname=filename, dpi=200)
    )

    return filename



def plot_chart(data, ticker):
    save_path = f"{ticker}_chart.png"
    
    # Create a copy of the data with only the required columns for mplfinance
    # Use only the most recent 365 days for better visualization
    if len(data) > 365:
        plot_data = data.iloc[-365:][['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    else:
        plot_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Verify that all values are numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if not pd.api.types.is_numeric_dtype(plot_data[col]):
            plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
    
    # Drop any NaN values that might remain
    plot_data = plot_data.dropna()
    
    # Enhance the chart style with colors for better visibility
    mc = mpf.make_marketcolors(
        up='green',
        down='red',
        edge='inherit',
        wick={'up':'green', 'down':'red'},
        volume={'up':'green', 'down':'red'},
    )
    
    # Create a custom style with improved visibility
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=':', 
        gridaxis='both',
        y_on_right=False,
        rc={'font.size': 12}  # Increased font size
    )
    
    # Ensure we have enough data to plot
    if len(plot_data) > 0:
        # Define figure size for higher quality
        fig_config = dict(
            figsize=(14, 10),  # Larger figure for better visibility
            panel_ratios=(4, 1),
            tight_layout=True,
            # dpi=300  # Higher DPI for better resolution
        )
        
        # Get latest values for annotations
        last_price = plot_data['Close'].iloc[-1]
        last_date = plot_data.index[-1].strftime('%Y-%m-%d')
        sma20_val = data['SMA20'].iloc[-1] if 'SMA20' in data else None
        sma150_val = data['SMA150'].iloc[-1] if 'SMA150' in data else None
        
        # Add support and resistance levels
        support = detect_support(data)
        resistance = detect_resistance(data)
        
        # Add horizontal lines for support and resistance
        hlines = []
        colors = []
        labels = []
        
        if not pd.isna(support):
            hlines.append(support)
            colors.append('green')
            labels.append(f'Support: ${support:.2f}')
            
        if not pd.isna(resistance):
            hlines.append(resistance)
            colors.append('red')
            labels.append(f'Resistance: ${resistance:.2f}')
        
        # Create a figure first to add custom annotations
        fig, axes = mpf.plot(
            plot_data,
            type='candle',
            style=s,
            volume=True,
            title=f'{ticker} - Technical Analysis Chart',
            ylabel='Price ($)',
            ylabel_lower='Volume',
            mav=(20, 150),
            mavcolors=['blue', 'red'],
            figratio=(16, 9),
            figscale=1.5,
            hlines=dict(hlines=hlines, colors=colors, linestyle='--', linewidths=1.5),
            returnfig=True,  # Return the figure to add annotations
            **fig_config
        )
        
        # Add explanatory text for the chart elements
        legend_text = (
            f"Chart Legend:\n"
            f"- Candlesticks: Green = Up day, Red = Down day\n"
            f"- Blue Line: 20-day Moving Average (${sma20_val:.2f})\n"
            f"- Red Line: 150-day Moving Average (${sma150_val:.2f})\n"
            f"- Green Dashed Line: Support Level (${support:.2f})\n"
            f"- Red Dashed Line: Resistance Level (${resistance:.2f})\n"
            f"- Bottom Panel: Volume\n\n"
            f"Current Price: ${last_price:.2f} (as of {last_date})"
        )
        
        # Add the legend text to the chart
        axes[0].text(
            0.01, 0.01, legend_text,
            transform=axes[0].transAxes,
            fontsize=9,
            verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5')
        )
        
        # Save the figure with the added annotations
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        print(f"Warning: Not enough valid data to plot chart for {ticker}")
        
    return save_path


def detect_trend(data):
    sma20 = data['SMA20'].iloc[-1]
    sma150 = data['SMA150'].iloc[-1]
    price_now = data['Close'].iloc[-1]
    price_past = data['Close'].iloc[-30]
    if sma20 > sma150 and price_now > price_past:
        return "Uptrend"
    elif sma20 < sma150 and price_now < price_past:
        return "Downtrend"
    else:
        return "Sideways / Unclear"

def detect_candlestick(data):
    last = data.iloc[-1]
    prev = data.iloc[-2]
    open_ = last['Open']
    close = last['Close']
    high = last['High']
    low = last['Low']
    body = abs(close - open_)
    range_ = high - low
    if body / range_ < 0.3 and (min(open_, close) - low) > 2 * body:
        return 'Hammer'
    if prev['Close'] < prev['Open'] and last['Close'] > last['Open'] and last['Close'] > prev['Open'] and last['Open'] < prev['Close']:
        return 'Bullish Engulfing'
    if abs(close - open_) / range_ < 0.05:
        return 'Doji'
    return 'None'

def detect_support(data, lookback=30):
    return round(data['Low'][-lookback:].min(), 2)

def detect_resistance(data, lookback=30):
    return round(data['High'][-lookback:].max(), 2)

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return round(macd.iloc[-1], 2), round(signal.iloc[-1], 2)

def generate_prompt(ticker, data):
    current_price = round(data['Close'].iloc[-1], 2)
    sma20 = round(data['SMA20'].iloc[-1], 2)
    sma150 = round(data['SMA150'].iloc[-1], 2)
    volume = int(data['Volume'].iloc[-1])
    avg_volume = int(data['Volume'].rolling(20).mean().iloc[-1])
    trend = detect_trend(data)
    support_price = detect_support(data)
    resistance = detect_resistance(data)
    pattern = detect_candlestick(data)
    rsi = calculate_rsi(data)
    macd, macd_signal = calculate_macd(data)
    
    # Calculate price relationship to moving averages
    pct_from_sma20 = round(((current_price - sma20) / sma20) * 100, 2)
    pct_from_sma150 = round(((current_price - sma150) / sma150) * 100, 2) if not pd.isna(sma150) else "N/A"
    
    # Check if SMA20 crossed above/below SMA150 recently (bullish/bearish crossover)
    crossover = "None"
    if len(data) > 30:
        sma20_past = data['SMA20'].iloc[-30]
        sma150_past = data['SMA150'].iloc[-30]
        if not pd.isna(sma20_past) and not pd.isna(sma150_past) and not pd.isna(sma20) and not pd.isna(sma150):
            if sma20_past < sma150_past and sma20 > sma150:
                crossover = "Recent Bullish Crossover (Golden Cross)"
            elif sma20_past > sma150_past and sma20 < sma150:
                crossover = "Recent Bearish Crossover (Death Cross)"

    prompt = f"""
You are a professional and experienced technical analyst and swing trader.

Based on the following stock data, analyze whether this is a valid entry point for a short- to medium-term trade (lasting from a few days up to a few weeks). 

Use your analysis of candlestick patterns, moving averages, RSI, MACD, and support/resistance levels to justify your recommendation. 

If the trade is favorable, provide:
- A clear entry rationale
- Suggested stop-loss level (price)
- Target profit 1 (TP1) and optionally Target profit 2 (TP2)
- A concise but professional technical explanation (no fluff).

---

Ticker: {ticker}  
Current Price: ${current_price}  

MOVING AVERAGES:
- SMA20: ${sma20} ({pct_from_sma20}% from current price)
- SMA150: ${sma150} ({pct_from_sma150}% from current price)
- MA Crossover Status: {crossover}

CHART INDICATORS:
- Overall Trend: {trend}  
- Last Candlestick Pattern: {pattern}  
- Support Level (last 30 days): ${support_price}  
- Resistance Level (last 30 days): ${resistance}  

TECHNICAL INDICATORS:
- RSI (14 days): {rsi} {'(Oversold)' if rsi < 30 else '(Overbought)' if rsi > 70 else '(Neutral)'}
- MACD: {macd} | Signal Line: {macd_signal} | {'Bullish' if macd > macd_signal else 'Bearish'} Momentum

VOLUME:
- Daily Volume: {volume:,} 
- 20-day Average: {avg_volume:,}
- Volume Ratio: {round(volume/avg_volume, 2)}x average

CHART EXPLANATION:
- Green candles represent days where price closed higher than it opened
- Red candles represent days where price closed lower than it opened
- Blue line is the 20-day moving average (short-term trend)
- Red line is the 150-day moving average (long-term trend)
- Green dashed line shows the support level (${support_price})
- Red dashed line shows the resistance level (${resistance})

Is this currently a good entry point to long trade? Respond as a seasoned technical trader would.
please answer in Hebrew, and use a professional tone and telegram-friendly format with icons, emojis and markdown.
"""
    return prompt.strip()


def get_trending_stocks(api_key, endpoint, deployment):
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-15-preview",
        azure_endpoint=endpoint
    )

    prompt = (
    "Simulate a list of 30 U.S. stocks that are typically popular, trending, or high-growth based on Reddit discussion trends "
    "(such as those commonly found in r/wallstreetbets), using general market knowledge and historic sentiment patterns. "
    "Return only the stock tickers, comma-separated."
)

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a financial analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )

    # שליפת רשימת המניות והפיכה לרשימת פייתון
    content = response.choices[0].message.content
    tickers = [ticker.strip().upper() for ticker in content.split(",") if ticker.strip()]
    return tickers

def send_telegram(message,bot_token,chat_id, image_path=None):
    """Send message and optional image to Telegram"""

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    response = requests.post(url, data=data)
    
    # If we have an image, send it as a photo
    if image_path and os.path.exists(image_path):
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        files = {"photo": open(image_path, 'rb')}
        data = {"chat_id": chat_id, "caption": "Chart Analysis"}
        response = requests.post(url, data=data, files=files)
    
    return response.json()


if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Stock Technical Analysis Tool')
    parser.add_argument('ticker', nargs='?', default='AFRM', help='Ticker symbol to analyze')
    parser.add_argument('--trending_stocks', action='store_true', help='Use trending stocks from Reddit')
    parser.add_argument('--endpoint', help='Azure OpenAI endpoint')
    parser.add_argument('--deployment_name', default='gpt-4o-mini', help='Deployment name for Azure OpenAI')
    parser.add_argument('--telegram_token', help='Telegram bot token for sending messages')
    parser.add_argument('--telegram_chat_id', help='Telegram chat ID to send messages')
    parser.add_argument('--azure_token')
    args = parser.parse_args()
    
    api_key = args.azure_token
    endpoint = args.endpoint
    deployment = args.deployment_name
    telegram_token = args.telegram_token
    telegram_chat_id = args.telegram_chat_id
    
    # Determine which stocks to analyze
    if args.trending_stocks:
        print("Fetching trending stocks")
        tickers = get_trending_stocks(api_key, endpoint, deployment)
    else:
        tickers = [args.ticker.upper()]
    
    # Track analysis stats
    analyzed = 0
    good_entry_points = 0
    
    # Process each ticker
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Analyzing {ticker}...")
        
        try:
            data = fetch_data(ticker)
            
            # Check if we have enough data for analysis and visualization
            if len(data) < 150:
                print(f"Warning: Not enough data for {ticker} to calculate 150-day moving average. Found {len(data)} rows.")
                if len(data) < 20:
                    print(f"Error: Insufficient data for analysis. Need at least 20 days of data.")
                    continue
            
            #chart_path = plot_chart(data, ticker)
            chart_path = plot_chart_with_adaptive_trendlines(ticker)  # Use adaptive trendlines
            prompt = generate_prompt(ticker, data)
            
            analyzed += 1
            print(prompt)
            print(f"\n📊 Chart saved to: {chart_path}")
            
            # Simple criteria for "good entry point" detection
            # (You can customize this based on your trading strategy)
            price = data['Close'].iloc[-1]
            sma150 = data['SMA150'].iloc[-1]
            rsi = calculate_rsi(data)
            
            # Example criteria: Price above 150MA with RSI not overbought
            if not pd.isna(sma150) and price > sma150 and rsi < 70 and rsi > 30:
                print("✅ Potential good entry point detected!")
                good_entry_points += 1
            
            # Add a delay between API calls to avoid rate limiting
            if args.trending_stocks and ticker != tickers[-1]:
                time.sleep(1)

            response = query_openai_with_prompt_v1(
            prompt_text=prompt,
            image_path=chart_path,
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment)

            print(response)
            # Send the response to a telegram channel with the ticker and chart
            message = f"Analysis for {ticker}:\n{response}"
            send_telegram(message, telegram_token, telegram_chat_id, image_path=chart_path)
            

                
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()

    

    # מתוך הקוד שלך:
    # prompt = generate_prompt(ticker, data)
    # chart_path = plot_chart(data, ticker)




