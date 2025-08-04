#!/usr/bin/env python3
"""
Stock Technical Analysis Tool with AI Analysis

A comprehensive stock analysis tool that:
- Fetches real-time stock data using Yahoo Finance
- Generates technical analysis charts with moving averages
- Uses Google Gemini AI to provide professional trading insights
- Supports analysis of trending stocks or specific tickers
- Sends analysis and charts to Telegram channels

Author: Stock Analysis Bot
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
import requests
from bs4 import BeautifulSoup
import time
import random
import google.generativeai as genai
import base64
from matplotlib import ticker
from torch import mode
import websockets
import asyncio
import yfinance as yf
import numpy as np
from PIL import Image
import sys
import argparse
import traceback

# ======================== CONFIGURATION ========================

# Chart settings
CHART_CONFIG = {
    'figsize': (14, 10),
    'panel_ratios': (4, 1),
    'dpi': 200,
    'style': 'yahoo',
    'volume': True,
    'tight_layout': True
}

# Technical analysis settings
TECHNICAL_CONFIG = {
    'sma_periods': [20, 150],
    'rsi_window': 14,
    'ema_periods': [12, 26],
    'macd_signal_period': 9,
    'trend_lookback': 30,
    # הוספת אינדיקטורים חדשים
    'atr_period': 14,
    'volume_sma_period': 20,
    'momentum_period': 10
}

# Color scheme
COLORS = {
    'sma20': 'blue',
    'sma150': 'purple',
    'current_price': 'orange',
    'up_candle': 'green',
    'down_candle': 'red'
}

# ======================== AI AND DATA FUNCTIONS ========================

def query_gemini_with_prompt_and_image(prompt_text, image_path, api_key):
    """
    Query Google Gemini with text prompt and image
    """
    genai.configure(api_key=api_key)
    
    # Load and process the image
    image = Image.open(image_path)
    
    # Initialize the Gemini model (using gemini-2.5-flash for vision capabilities)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Create the prompt with system instructions
    full_prompt = f"""You are a professional swing trader and technical analyst.

{prompt_text}"""
    
    try:
        # Generate content with both text and image
        response = model.generate_content([full_prompt, image])
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"Error generating analysis: {str(e)}"


# ======================== DATA FETCHING FUNCTIONS ========================

def fetch_data(ticker, period='1y'):
    """
    Fetch stock data from Yahoo Finance with proper data cleaning
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period (default: '1y')
    
    Returns:
        pandas.DataFrame: Clean stock data with moving averages
    """
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
    data['SMA20'] = data['Close'].rolling(window=TECHNICAL_CONFIG['sma_periods'][0]).mean()
    data['SMA150'] = data['Close'].rolling(window=TECHNICAL_CONFIG['sma_periods'][1]).mean()
    
    return data

def get_trending_stocks(api_key):
    """
    Get trending stocks using Gemini API
    
    Args:
        api_key (str): Google Gemini API key
    
    Returns:
        list: List of stock ticker symbols
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = (
        "Simulate a list of 30 U.S. stocks that are typically popular, trending, or high-growth based on Reddit discussion trends "
        "(such as those commonly found in r/wallstreetbets), using general market knowledge and historic sentiment patterns. "
        "Return only the stock tickers, comma-separated."
    )

    try:
        response = model.generate_content(prompt)
        # Extract the list of stocks and convert to Python list
        content = response.text
        tickers = [ticker.strip().upper() for ticker in content.split(",") if ticker.strip()]
        return tickers
    except Exception as e:
        print(f"Error getting trending stocks: {e}")
        # Return a default list of popular stocks if API fails
        return ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'GOOGL', 'MSFT', 'META', 'AMD', 'NFLX', 'PLTR']

# ======================== CHART GENERATION FUNCTIONS ========================
def plot_enhanced_chart(data, ticker):
    """
    Generate an enhanced chart with detailed formatting and annotations
    
    Args:
        data (pandas.DataFrame): Stock data with moving averages
        ticker (str): Stock ticker symbol
    
    Returns:
        str: Path to the generated chart file
    """
    save_path = f"{ticker}_chart_enhanced.png"
    
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
        up=COLORS['up_candle'],
        down=COLORS['down_candle'],
        edge='inherit',
        wick={'up': COLORS['up_candle'], 'down': COLORS['down_candle']},
        volume={'up': COLORS['up_candle'], 'down': COLORS['down_candle']},
    )
    
    # Create a custom style with improved visibility
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=':', 
        gridaxis='both',
        y_on_right=False,
        rc={'font.size': 12}
    )
    
    # Ensure we have enough data to plot
    if len(plot_data) > 0:
        # Define figure size for higher quality
        fig_config = dict(
            figsize=CHART_CONFIG['figsize'],
            panel_ratios=CHART_CONFIG['panel_ratios'],
            tight_layout=CHART_CONFIG['tight_layout'],
        )
        
        # Get latest values for annotations
        last_price = plot_data['Close'].iloc[-1]
        last_date = plot_data.index[-1].strftime('%Y-%m-%d')
        sma20_val = data['SMA20'].iloc[-1] if 'SMA20' in data else None
        sma150_val = data['SMA150'].iloc[-1] if 'SMA150' in data else None
        
        # Create a figure first to add custom annotations
        fig, axes = mpf.plot(
            plot_data,
            type='candle',
            style=s,
            volume=True,
            title=f'{ticker} - Technical Analysis Chart',
            ylabel='Price ($)',
            ylabel_lower='Volume',
            mav=tuple(TECHNICAL_CONFIG['sma_periods']),
            mavcolors=[COLORS['sma20'], COLORS['sma150']],
            figratio=(16, 9),
            figscale=1.5,
            returnfig=True,
            **fig_config
        )
        
        # Improve Y-axis formatting for better price readability
        if axes and len(axes) > 0:
            _format_chart_axes(axes, plot_data)
        
        # Add explanatory text for the chart elements
        legend_text = _create_chart_legend(sma20_val, sma150_val, last_price, last_date)
        
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

def _format_chart_axes(axes, plot_data):
    """Helper function to format chart axes for better readability"""
    # Format Y-axis to show more price levels with dollar signs
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
    # Add more tick marks for better price reference
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=12, prune='lower'))
    axes[0].yaxis.set_minor_locator(ticker.MaxNLocator(nbins=24, prune='lower'))
    
    # Add grid lines for better price level visibility
    axes[0].grid(True, linestyle=':', alpha=0.6, which='major')
    axes[0].grid(True, linestyle=':', alpha=0.3, which='minor')
    
    # Improve Y-axis label positioning and formatting
    axes[0].tick_params(axis='y', labelsize=10, colors='black')
    
    # Add current price as a horizontal line with label
    current_price = plot_data['Close'].iloc[-1]
    axes[0].axhline(y=current_price, color=COLORS['current_price'], linestyle='-', linewidth=2, alpha=0.8)
    axes[0].text(len(plot_data)*0.02, current_price, f'Current: ${current_price:.2f}', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['current_price'], alpha=0.8),
                fontsize=9, fontweight='bold')

def _create_chart_legend(sma20_val, sma150_val, last_price, last_date):
    """Helper function to create chart legend text"""
    return (
        f"Chart Legend:\n"
        f"- Candlesticks: Green = Up day, Red = Down day\n"
        f"- Blue Line: 20-day Moving Average (${sma20_val:.2f})\n"
        f"- Purple Line: 150-day Moving Average (${sma150_val:.2f})\n"
        f"- Bottom Panel: Volume\n\n"
        f"Current Price: ${last_price:.2f} (as of {last_date})"
    )

# ======================== TECHNICAL ANALYSIS FUNCTIONS ========================


def detect_trend(data):
    """
    Detect overall market trend based on moving averages and price movement
    
    Args:
        data (pandas.DataFrame): Stock data with moving averages
    
    Returns:
        str: Trend description ('Uptrend', 'Downtrend', or 'Sideways / Unclear')
    """
    sma20 = data['SMA20'].iloc[-1]
    sma150 = data['SMA150'].iloc[-1]
    price_now = data['Close'].iloc[-1]
    price_past = data['Close'].iloc[-TECHNICAL_CONFIG['trend_lookback']]
    
    if sma20 > sma150 and price_now > price_past:
        return "Uptrend"
    elif sma20 < sma150 and price_now < price_past:
        return "Downtrend"
    else:
        return "Sideways / Unclear"

def analyze_candlestick_patterns(data):
    """
    Comprehensive candlestick pattern analysis
    
    Args:
        data (pandas.DataFrame): Stock data
    
    Returns:
        dict: Detected patterns with signals and descriptions
    """
    patterns = []
    signals = []
    
    if len(data) < 3:
        return {'patterns': [], 'signals': [], 'description': 'Insufficient data'}
    
    # Get recent candles
    last = data.iloc[-1]
    prev = data.iloc[-2] if len(data) >= 2 else None
    prev2 = data.iloc[-3] if len(data) >= 3 else None
    
    # Calculate candle properties
    def get_candle_info(candle):
        open_price = candle['Open']
        close_price = candle['Close']
        high_price = candle['High']
        low_price = candle['Low']
        
        body = abs(close_price - open_price)
        total_range = high_price - low_price
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        is_bullish = close_price > open_price
        body_ratio = body / total_range if total_range > 0 else 0
        
        return {
            'body': body, 'range': total_range, 'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow, 'is_bullish': is_bullish,
            'body_ratio': body_ratio, 'open': open_price, 'close': close_price,
            'high': high_price, 'low': low_price
        }
    
    last_info = get_candle_info(last)
    
    # 1. DOJI - גוף קטן, אי החלטיות
    if last_info['body_ratio'] < 0.1:  # גוף קטן מ-10% מהטווח
        patterns.append("Doji")
        signals.append("Indecision")
    
    # 2. HAMMER - פטיש (bullish reversal)
    if (last_info['lower_shadow'] > 2 * last_info['body'] and 
        last_info['upper_shadow'] < last_info['body'] * 0.5 and
        last_info['body_ratio'] > 0.1):
        patterns.append("Hammer")
        signals.append("Potential Bullish Reversal")
    
    # 3. SHOOTING STAR - כוכב יורה (bearish reversal)
    if (last_info['upper_shadow'] > 2 * last_info['body'] and 
        last_info['lower_shadow'] < last_info['body'] * 0.5 and
        last_info['body_ratio'] > 0.1):
        patterns.append("Shooting Star")
        signals.append("Potential Bearish Reversal")
    
    # 4. SPINNING TOP - סביבון (indecision)
    if (last_info['body_ratio'] < 0.3 and 
        last_info['upper_shadow'] > last_info['body'] and
        last_info['lower_shadow'] > last_info['body']):
        patterns.append("Spinning Top")
        signals.append("Indecision")
    
    # Multi-candle patterns (need previous candles)
    if prev is not None:
        prev_info = get_candle_info(prev)
        
        # 5. ENGULFING PATTERNS - בליעה
        # Bullish Engulfing
        if (not prev_info['is_bullish'] and last_info['is_bullish'] and
            last_info['open'] < prev_info['close'] and 
            last_info['close'] > prev_info['open']):
            patterns.append("Bullish Engulfing")
            signals.append("Strong Bullish Signal")
        
        # Bearish Engulfing
        if (prev_info['is_bullish'] and not last_info['is_bullish'] and
            last_info['open'] > prev_info['close'] and 
            last_info['close'] < prev_info['open']):
            patterns.append("Bearish Engulfing")
            signals.append("Strong Bearish Signal")
        
        # 6. PIERCING PATTERN - חדירה (bullish)
        if (not prev_info['is_bullish'] and last_info['is_bullish'] and
            last_info['open'] < prev_info['low'] and 
            last_info['close'] > (prev_info['open'] + prev_info['close']) / 2):
            patterns.append("Piercing Pattern")
            signals.append("Bullish Signal")
        
        # 7. DARK CLOUD COVER - כיסוי ענן כהה (bearish)
        if (prev_info['is_bullish'] and not last_info['is_bullish'] and
            last_info['open'] > prev_info['high'] and 
            last_info['close'] < (prev_info['open'] + prev_info['close']) / 2):
            patterns.append("Dark Cloud Cover")
            signals.append("Bearish Signal")
    
    # Three-candle patterns
    if prev is not None and prev2 is not None:
        prev2_info = get_candle_info(prev2)
        
        # 8. MORNING STAR - כוכב בוקר (bullish reversal)
        if (not prev2_info['is_bullish'] and  # First candle bearish
            prev_info['body_ratio'] < 0.3 and  # Middle candle small
            last_info['is_bullish'] and  # Last candle bullish
            last_info['close'] > prev2_info['close']):  # Close above first candle close
            patterns.append("Morning Star")
            signals.append("Strong Bullish Reversal")
        
        # 9. EVENING STAR - כוכב ערב (bearish reversal)
        if (prev2_info['is_bullish'] and  # First candle bullish
            prev_info['body_ratio'] < 0.3 and  # Middle candle small
            not last_info['is_bullish'] and  # Last candle bearish
            last_info['close'] < prev2_info['close']):  # Close below first candle close
            patterns.append("Evening Star")
            signals.append("Strong Bearish Reversal")
    
    # Create description
    if patterns:
        description = f"זוהו {len(patterns)} תבניות נרות: {', '.join(patterns)}"
    else:
        description = "לא זוהו תבניות נרות מיוחדות"
    
    return {
        'patterns': patterns,
        'signals': signals,
        'description': description,
        'last_candle_type': 'Bullish' if last_info['is_bullish'] else 'Bearish'
    }

def detect_chart_patterns(data):
    """
    Detect major chart patterns like head & shoulders, cup and handle, etc.
    
    Args:
        data (pandas.DataFrame): Stock data
    
    Returns:
        dict: Detected chart patterns with levels and signals
    """
    patterns = []
    signals = []
    key_levels = []
    
    if len(data) < 50:
        return {'patterns': [], 'signals': [], 'key_levels': [], 'description': 'Insufficient data for chart patterns'}
    
    # Use recent 100 days for pattern detection
    recent_data = data.iloc[-100:] if len(data) > 100 else data
    highs = recent_data['High']
    lows = recent_data['Low']
    closes = recent_data['Close']
    
    # 1. SUPPORT AND RESISTANCE LEVELS
    # Find significant highs and lows
    from scipy.signal import find_peaks
    
    try:
        # Find peaks (resistance levels)
        peaks, _ = find_peaks(highs, distance=5, prominence=highs.std()*0.5)
        # Find troughs (support levels)
        troughs, _ = find_peaks(-lows, distance=5, prominence=lows.std()*0.5)
        
        # Get current price
        current_price = closes.iloc[-1]
        
        # Check for support/resistance near current price
        if len(peaks) > 0:
            resistance_levels = highs.iloc[peaks].values
            nearby_resistance = [r for r in resistance_levels if abs(r - current_price) / current_price < 0.05]
            if nearby_resistance:
                key_levels.extend([f"Resistance: ${r:.2f}" for r in nearby_resistance])
        
        if len(troughs) > 0:
            support_levels = lows.iloc[troughs].values
            nearby_support = [s for s in support_levels if abs(s - current_price) / current_price < 0.05]
            if nearby_support:
                key_levels.extend([f"Support: ${s:.2f}" for s in nearby_support])
    
    except Exception:
        pass  # Skip if scipy not available or other errors
    
    # 2. DOUBLE TOP/BOTTOM
    if len(recent_data) >= 20:
        # Simple double top detection
        max_high = highs.max()
        second_max = highs.nlargest(2).iloc[1]
        if abs(max_high - second_max) / max_high < 0.02:  # Within 2%
            patterns.append("Potential Double Top")
            signals.append("Bearish Pattern")
            key_levels.append(f"Double Top Level: ${max_high:.2f}")
        
        # Simple double bottom detection
        min_low = lows.min()
        second_min = lows.nsmallest(2).iloc[1]
        if abs(min_low - second_min) / min_low < 0.02:  # Within 2%
            patterns.append("Potential Double Bottom")
            signals.append("Bullish Pattern")
            key_levels.append(f"Double Bottom Level: ${min_low:.2f}")
    
    # 3. TRIANGLE PATTERNS (simplified)
    if len(recent_data) >= 30:
        recent_highs = highs.iloc[-30:]
        recent_lows = lows.iloc[-30:]
        
        # Check if highs are declining and lows are rising (symmetrical triangle)
        high_trend = recent_highs.iloc[-1] < recent_highs.iloc[0]
        low_trend = recent_lows.iloc[-1] > recent_lows.iloc[0]
        
        if high_trend and low_trend:
            patterns.append("Symmetrical Triangle")
            signals.append("Breakout Expected")
            key_levels.append(f"Triangle Apex Near: ${closes.iloc[-1]:.2f}")
    
    # 4. CUP AND HANDLE (simplified detection)
    if len(recent_data) >= 50:
        mid_point = len(recent_data) // 2
        left_high = highs.iloc[:mid_point].max()
        cup_low = lows.iloc[mid_point//2:mid_point+mid_point//2].min()
        right_high = highs.iloc[mid_point:].max()
        
        # Check if we have a cup shape
        if (abs(left_high - right_high) / left_high < 0.05 and  # Similar highs
            (left_high - cup_low) / left_high > 0.15):  # Significant dip
            patterns.append("Potential Cup Pattern")
            signals.append("Watch for Handle Formation")
            key_levels.append(f"Cup Rim: ${max(left_high, right_high):.2f}")
    
    # 5. FLAGS AND PENNANTS (simplified)
    if len(recent_data) >= 20:
        # Check for consolidation after strong move
        recent_20 = recent_data.iloc[-20:]
        range_pct = (recent_20['High'].max() - recent_20['Low'].min()) / recent_20['Close'].mean()
        
        if range_pct < 0.1:  # Tight consolidation
            patterns.append("Consolidation/Flag Pattern")
            signals.append("Potential Breakout Setup")
    
    # Create description
    if patterns:
        description = f"זוהו {len(patterns)} תבניות גרף: {', '.join(patterns)}"
    else:
        description = "לא זוהו תבניות גרף בולטות"
    
    return {
        'patterns': patterns,
        'signals': signals,
        'key_levels': key_levels,
        'description': description
    }

def detect_candlestick(data):
    """
    Legacy function - replaced by analyze_candlestick_patterns
    """
    result = analyze_candlestick_patterns(data)
    if result['patterns']:
        return result['patterns'][0]  # Return first pattern for compatibility
    return 'None'

def analyze_moving_average_crossover(data):
    """
    Analyze moving average crossover patterns
    
    Args:
        data (pandas.DataFrame): Stock data with SMA20 and SMA150 columns
    
    Returns:
        str: Crossover status description
    """
    try:
        # Ensure we have SMA data
        if 'SMA20' not in data.columns or 'SMA150' not in data.columns:
            # Calculate SMAs if they don't exist
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['SMA150'] = data['Close'].rolling(window=150).mean()
        
        # Get recent data points
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) >= 2 else latest
        
        current_sma20 = latest['SMA20']
        current_sma150 = latest['SMA150']
        prev_sma20 = previous['SMA20']
        prev_sma150 = previous['SMA150']
        
        # Check for valid data
        if pd.isna(current_sma20) or pd.isna(current_sma150):
            return "Insufficient data for crossover analysis"
        
        # Determine crossover status
        if current_sma20 > current_sma150:
            if prev_sma20 <= prev_sma150:
                return "Golden Cross (Bullish)"
            else:
                return "SMA20 above SMA150 (Bullish)"
        elif current_sma20 < current_sma150:
            if prev_sma20 >= prev_sma150:
                return "Death Cross (Bearish)"
            else:
                return "SMA20 below SMA150 (Bearish)"
        else:
            return "SMAs converging (Neutral)"
            
    except Exception as e:
        return f"Error in crossover analysis: {str(e)}"

def calculate_rsi(data, window=None):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data (pandas.DataFrame): Stock data
        window (int): RSI calculation window (default from config)
    
    Returns:
        float: RSI value
    """
    if window is None:
        window = TECHNICAL_CONFIG['rsi_window']
        
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calculate_macd(data):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data (pandas.DataFrame): Stock data
    
    Returns:
        tuple: (MACD value, Signal line value)
    """
    ema12 = data['Close'].ewm(span=TECHNICAL_CONFIG['ema_periods'][0], adjust=False).mean()
    ema26 = data['Close'].ewm(span=TECHNICAL_CONFIG['ema_periods'][1], adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=TECHNICAL_CONFIG['macd_signal_period'], adjust=False).mean()
    return round(macd.iloc[-1], 2), round(signal.iloc[-1], 2)

def calculate_atr(data):
    """
    Calculate Average True Range (ATR) for volatility measurement
    
    Args:
        data (pandas.DataFrame): Stock data
    
    Returns:
        float: ATR value
    """
    period = TECHNICAL_CONFIG['atr_period']
    
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return round(atr.iloc[-1], 2)

def calculate_volume_indicators(data):
    """
    Calculate volume-based indicators
    
    Args:
        data (pandas.DataFrame): Stock data
    
    Returns:
        tuple: (Volume SMA, Volume ratio)
    """
    volume_sma = data['Volume'].rolling(window=TECHNICAL_CONFIG['volume_sma_period']).mean()
    current_volume = data['Volume'].iloc[-1]
    avg_volume = volume_sma.iloc[-1]
    volume_ratio = round(current_volume / avg_volume, 2)
    
    return round(avg_volume, 0), volume_ratio

def calculate_momentum(data):
    """
    Calculate price momentum
    
    Args:
        data (pandas.DataFrame): Stock data
    
    Returns:
        tuple: (Momentum value, Momentum percentage)
    """
    period = TECHNICAL_CONFIG['momentum_period']
    
    current_price = data['Close'].iloc[-1]
    past_price = data['Close'].iloc[-period]
    
    momentum = current_price - past_price
    momentum_pct = ((current_price - past_price) / past_price) * 100
    
    return round(momentum, 2), round(momentum_pct, 2)

def calculate_risk_management(data, entry_price=None):
    """
    Calculate advanced risk management parameters
    
    Args:
        data (pandas.DataFrame): Stock data
        entry_price (float, optional): Proposed entry price (default: current price)
    
    Returns:
        dict: Risk management recommendations
    """
    if entry_price is None:
        entry_price = data['Close'].iloc[-1]
    
    atr = calculate_atr(data)
    sma20 = data['SMA20'].iloc[-1]
    
    # חישוב stop-loss בשיטות שונות
    atr_stop = entry_price - (2 * atr)  # ATR-based stop
    sma20_stop = sma20 * 0.95  # 5% מתחת ל-SMA20
    
    # בחירת stop-loss הקרוב ביותר למחיר הכניסה (הכי שמרני)
    suggested_stop = max(atr_stop, sma20_stop)
    stop_loss_pct = ((entry_price - suggested_stop) / entry_price) * 100
    
    # חישוב יעדי רווח
    tp1 = entry_price + (2 * atr)  # 2:1 Risk-Reward
    tp2 = entry_price + (3 * atr)  # 3:1 Risk-Reward
    
    # חישוב גודל פוזיציה (2% סיכון מהחשבון)
    risk_per_trade = 0.02  # 2% מהחשבון
    account_size = 10000  # ברירת מחדל - ניתן להתאמה
    risk_amount = account_size * risk_per_trade
    position_size = risk_amount / (entry_price - suggested_stop)
    
    return {
        'entry_price': round(entry_price, 2),
        'stop_loss': round(suggested_stop, 2),
        'stop_loss_pct': round(stop_loss_pct, 2),
        'tp1': round(tp1, 2),
        'tp2': round(tp2, 2),
        'position_size': round(position_size, 0),
        'risk_reward_1': round((tp1 - entry_price) / (entry_price - suggested_stop), 2),
        'risk_reward_2': round((tp2 - entry_price) / (entry_price - suggested_stop), 2),
        'atr': atr
    }

def detect_advanced_patterns(data):
    """
    Detect advanced candlestick and chart patterns
    
    Args:
        data (pandas.DataFrame): Stock data
    
    Returns:
        dict: Detected patterns and their signals
    """
    patterns = []
    signals = []
    
    if len(data) < 10:
        return {'patterns': [], 'signals': []}
    
    # קשר עם 5 הנרות האחרונים
    recent_data = data.iloc[-10:]
    
    # Morning Star / Evening Star
    if len(recent_data) >= 3:
        first = recent_data.iloc[-3]
        middle = recent_data.iloc[-2]
        last = recent_data.iloc[-1]
        
        # Morning Star (בולי)
        if (first['Close'] < first['Open'] and  # נר אדום
            abs(middle['Close'] - middle['Open']) < (first['Open'] - first['Close']) * 0.3 and  # נר קטן
            last['Close'] > last['Open'] and  # נר ירוק
            last['Close'] > first['Close']):  # סגירה מעל הנר הראשון
            patterns.append("Morning Star")
            signals.append("Bullish")
        
        # Evening Star (דובי)
        elif (first['Close'] > first['Open'] and  # נר ירוק
              abs(middle['Close'] - middle['Open']) < (first['Close'] - first['Open']) * 0.3 and  # נר קטן
              last['Close'] < last['Open'] and  # נר אדום
              last['Close'] < first['Close']):  # סגירה מתחת לנר הראשון
            patterns.append("Evening Star")
            signals.append("Bearish")
    
    # Double Bottom / Double Top (פשטות)
    lows = recent_data['Low'].rolling(window=3, center=True).min()
    highs = recent_data['High'].rolling(window=3, center=True).max()
    
    current_low = recent_data['Low'].iloc[-1]
    current_high = recent_data['High'].iloc[-1]
    
    # בדיקה פשוטה לדאבל בוטום
    if len(lows.dropna()) > 0:
        min_low = lows.min()
        if abs(current_low - min_low) / min_low < 0.02:  # 2% דמיון
            patterns.append("Potential Double Bottom")
            signals.append("Bullish")
    
    # תמיכה/התנגדות דינמית
    sma20 = data['SMA20'].iloc[-1]
    current_price = data['Close'].iloc[-1]
    
    if abs(current_price - sma20) / sma20 < 0.01:  # 1% מהממוצע הנע
        patterns.append("SMA20 Support/Resistance")
        signals.append("Key Level")
    
    return {
        'patterns': patterns,
        'signals': signals
    }

# ======================== AI ANALYSIS FUNCTIONS ========================
def generate_prompt(ticker, data):
    """
    Generate comprehensive analysis prompt for AI focused on price action and chart patterns
    
    Args:
        ticker (str): Stock ticker symbol
        data (pandas.DataFrame): Stock data with technical indicators
    
    Returns:
        str: Formatted prompt text for AI analysis
    """
    current_price = round(data['Close'].iloc[-1], 2)
    sma20 = round(data['SMA20'].iloc[-1], 2)
    sma150 = round(data['SMA150'].iloc[-1], 2)
    volume = int(data['Volume'].iloc[-1])
    avg_volume = int(data['Volume'].rolling(20).mean().iloc[-1])
    trend = detect_trend(data)
    
    # Get comprehensive pattern analysis
    candlestick_analysis = analyze_candlestick_patterns(data)
    chart_patterns = detect_chart_patterns(data)
    
    # Calculate price relationship to moving averages
    pct_from_sma20 = round(((current_price - sma20) / sma20) * 100, 2)
    pct_from_sma150 = round(((current_price - sma150) / sma150) * 100, 2) if not pd.isna(sma150) else "N/A"
    
    # Analyze crossover status
    crossover = analyze_moving_average_crossover(data)
    
    # Volume analysis
    volume_sma, volume_ratio = calculate_volume_indicators(data)
    
    # Format candlestick patterns
    candle_patterns = ', '.join(candlestick_analysis['patterns']) if candlestick_analysis['patterns'] else 'אין תבניות מיוחדות'
    candle_signals = ', '.join(candlestick_analysis['signals']) if candlestick_analysis['signals'] else 'ניטרלי'
    
    # Format chart patterns
    chart_pattern_list = ', '.join(chart_patterns['patterns']) if chart_patterns['patterns'] else 'אין תבניות גרף מיוחדות'
    key_levels = ', '.join(chart_patterns['key_levels']) if chart_patterns['key_levels'] else 'אין רמות מפתח קריטיות'

    prompt = f"""
You are a highly experienced swing trader and technical analyst.

Your job is to analyze the following stock data for a potential **swing trade entry** (holding period: a few days to several weeks).

Focus your analysis only on:
- Price action and candlestick patterns (e.g. doji, engulfing, hammer, shooting star)
- Chart patterns (e.g. head & shoulders, cup and handle, double bottom/top, flags, triangles)  
- Key moving averages: 20-day (short-term trend) and 150-day (long-term trend)
- Volume and volume anomalies

Provide a **clear and concise recommendation** on whether this is a good entry point **now**, or if it's better to wait. Mention exact levels (entry, stop, TP1/TP2) based on the technical structure. Keep your reasoning **short and sharp**.

Respond in **HEBREW** with a professional tone and friendly style for **Telegram**, using icons and emojis appropriately.

---

🎯 **מניה**: {ticker}  
💰 **מחיר נוכחי**: ${current_price}  

📈 **ממוצעים נעים**:
- SMA20: ${sma20} ({pct_from_sma20}% מהמחיר הנוכחי)
- SMA150: ${sma150} ({pct_from_sma150}% מהמחיר הנוכחי)
- מצב Crossover: {crossover}
- מגמה כללית: {trend}

🕯️ **תבניות נרות**:
- תבניות זוהו: {candle_patterns}
- סיגנלים: {candle_signals}
- סוג נר אחרון: {candlestick_analysis['last_candle_type']}

📊 **תבניות גרף**:
- תבניות זוהו: {chart_pattern_list}
- רמות מפתח: {key_levels}

📊 **נפח**:
- נפח נוכחי: {volume:,}
- ממוצע נפח (20 ימים): {volume_sma:,}
- יחס נפח: {volume_ratio}x

📋 **הסבר גרף**:
- נרות ירוקים = ימי עלייה (סגירה > פתיחה)
- נרות אדומים = ימי ירידה (סגירה < פתיחה)  
- קו כחול = ממוצע נע 20 ימים (מגמה קצרה)
- קו סגול = ממוצע נע 150 ימים (מגמה ארוכה)

🎯 **השאלה המרכזית**: האם זו נקודת כניסה טובה למסחר swing כרגע, או שעדיף לחכות?

ענה כסוחר טכני מנוסה, בטון מקצועי וידידותי לטלגרם עם אייקונים ואמוג'ים בעברית. תן המלצה ברורה עם רמות כניסה/עצירה/יעדים מדויקות.
"""
    return prompt.strip()

# ======================== COMMUNICATION FUNCTIONS ========================

def send_telegram(message, bot_token, chat_id, image_path=None):
    """
    Send message and optional image to Telegram
    
    Args:
        message (str): Text message to send
        bot_token (str): Telegram bot token
        chat_id (str): Telegram chat ID
        image_path (str, optional): Path to image file to send
    
    Returns:
        dict: Response from Telegram API
    """
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

def cleanup_chart_file(chart_path):
    """
    Delete chart file locally after sending to Telegram
    
    Args:
        chart_path (str): Path to chart file to delete
    """
    try:
        if os.path.exists(chart_path):
            os.remove(chart_path)
            print(f"🗑️ Deleted chart file: {chart_path}")
    except Exception as delete_error:
        print(f"Warning: Could not delete chart file {chart_path}: {delete_error}")

# ======================== ANALYSIS WORKFLOW FUNCTIONS ========================

def analyze_single_stock(ticker, api_key, send_telegram_flag=False, telegram_token=None, telegram_chat_id=None):
    """
    Analyze a single stock with complete workflow
    
    Args:
        ticker (str): Stock ticker symbol
        api_key (str): Google Gemini API key
        send_telegram_flag (bool): Whether to send results to Telegram
        telegram_token (str): Telegram bot token
        telegram_chat_id (str): Telegram chat ID
    
    Returns:
        dict: Analysis results including AI response and stats
    """
    print(f"Analyzing {ticker}...")
    
    try:
        # Fetch stock data
        data = fetch_data(ticker)
        
        # Check if we have enough data for analysis
        if len(data) < 150:
            print(f"Warning: Not enough data for {ticker} to calculate 150-day moving average. Found {len(data)} rows.")
            if len(data) < 20:
                print(f"Error: Insufficient data for analysis. Need at least 20 days of data.")
                return {'success': False, 'error': 'Insufficient data'}
        
        # Generate chart
        chart_path = plot_enhanced_chart(data, ticker)
        
        # Generate analysis prompt
        prompt = generate_prompt(ticker, data)
        
        # Get AI analysis
        ai_response = query_gemini_with_prompt_and_image(
            prompt_text=prompt,
            image_path=chart_path,
            api_key=api_key
        )
        
        # Evaluate if it's a good entry point
        entry_evaluation = evaluate_entry_point(data)
        
        # Calculate risk management
        risk_mgmt = calculate_risk_management(data)
        
        # Get candlestick and chart patterns
        candlestick_analysis = analyze_candlestick_patterns(data)
        chart_patterns = detect_chart_patterns(data)
        
        # Detect advanced patterns (legacy)
        advanced_patterns = detect_advanced_patterns(data)
        
        print(f"📊 Chart saved to: {chart_path}")
        print(f"🎯 Entry Score: {entry_evaluation['score']}% (Risk: {entry_evaluation['risk_level']})")
        print(f"💰 Risk Management: Entry ${risk_mgmt['entry_price']} | Stop ${risk_mgmt['stop_loss']} | TP1 ${risk_mgmt['tp1']}")
        if candlestick_analysis['patterns']:
            print(f"🕯️ Candlestick Patterns: {', '.join(candlestick_analysis['patterns'])}")
        if chart_patterns['patterns']:
            print(f"📈 Chart Patterns: {', '.join(chart_patterns['patterns'])}")
        if advanced_patterns['patterns']:
            print(f"📈 Advanced Patterns: {', '.join(advanced_patterns['patterns'])}")
        print(f"AI Analysis:\n{ai_response}")
        
        # Send to Telegram if requested
        if send_telegram_flag and telegram_token and telegram_chat_id:
            # Create comprehensive message with all information
            patterns_info = ""
            if candlestick_analysis['patterns']:
                patterns_info += f"\n🕯️ תבניות נרות: {', '.join(candlestick_analysis['patterns'])}"
            if chart_patterns['patterns']:
                patterns_info += f"\n📊 תבניות גרף: {', '.join(chart_patterns['patterns'])}"
            if advanced_patterns['patterns']:
                patterns_info += f"\n📈 תבניות נוספות: {', '.join(advanced_patterns['patterns'])}"
            
            message = f"""📈 ניתוח עבור {ticker}:

🎯 ציון כניסה: {entry_evaluation['score']}% (סיכון: {entry_evaluation['risk_level']})

💰 ניהול סיכון:
• כניסה: ${risk_mgmt['entry_price']}
• עצירה: ${risk_mgmt['stop_loss']} (-{risk_mgmt['stop_loss_pct']:.1f}%)
• יעד 1: ${risk_mgmt['tp1']} (R:R {risk_mgmt['risk_reward_1']:.1f}:1)
• יעד 2: ${risk_mgmt['tp2']} (R:R {risk_mgmt['risk_reward_2']:.1f}:1)
{patterns_info}

{ai_response}"""
            
            send_telegram(message, telegram_token, telegram_chat_id, image_path=chart_path)
            print("📱 Sent to Telegram successfully")
        
        # Clean up chart file
        cleanup_chart_file(chart_path)
        
        # Evaluate if it's a good entry point
        is_good_entry = evaluate_entry_point(data)
        
        return {
            'success': True,
            'ticker': ticker,
            'ai_response': ai_response,
            'is_good_entry': entry_evaluation['is_good_entry'],
            'entry_score': entry_evaluation['score'],
            'risk_level': entry_evaluation['risk_level'],
            'risk_management': risk_mgmt,
            'candlestick_patterns': candlestick_analysis,
            'chart_patterns': chart_patterns,
            'patterns': advanced_patterns,
            'chart_path': chart_path
        }
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        traceback.print_exc()
        return {'success': False, 'error': str(e), 'ticker': ticker}

def evaluate_entry_point(data):
    """
    Evaluate if the current price represents a good entry point using price action and chart patterns
    
    Args:
        data (pandas.DataFrame): Stock data with technical indicators
    
    Returns:
        dict: Detailed evaluation with score and reasoning
    """
    score = 0
    max_score = 100
    reasons = []
    
    # בדיקת המחיר ביחס לממוצעים נעים (40 נקודות מקסימום)
    price = data['Close'].iloc[-1]
    sma20 = data['SMA20'].iloc[-1]
    sma150 = data['SMA150'].iloc[-1]
    
    if not pd.isna(sma20) and not pd.isna(sma150):
        if price > sma20 > sma150:  # טרנד עולה חזק
            score += 25
            reasons.append("מחיר מעל שני הממוצעים הנעים - טרנד עולה חזק")
        elif price > sma20:  # מעל ממוצע קצר
            score += 15
            reasons.append("מחיר מעל ממוצע נע 20")
        elif price < sma20 < sma150:  # טרנד יורד
            score -= 10
            reasons.append("מחיר מתחת לשני הממוצעים - טרנד יורד")
    
    # בדיקת תבניות נרות (25 נקודות מקסימום)
    candlestick_analysis = analyze_candlestick_patterns(data)
    if candlestick_analysis['patterns']:
        bullish_patterns = ['Hammer', 'Bullish Engulfing', 'Piercing Pattern', 'Morning Star']
        bearish_patterns = ['Shooting Star', 'Bearish Engulfing', 'Dark Cloud Cover', 'Evening Star']
        
        for i, pattern in enumerate(candlestick_analysis['patterns']):
            if pattern in bullish_patterns:
                score += 15
                reasons.append(f"תבנית נרות בוליש: {pattern}")
            elif pattern in bearish_patterns:
                score -= 10
                reasons.append(f"תבנית נרות דוביש: {pattern}")
            elif pattern in ['Doji', 'Spinning Top']:
                score += 5
                reasons.append(f"אי החלטיות בשוק: {pattern}")
    
    # בדיקת תבניות גרף (20 נקודות מקסימום)
    chart_patterns = detect_chart_patterns(data)
    if chart_patterns['patterns']:
        bullish_chart_patterns = ['Double Bottom', 'Cup Pattern', 'Symmetrical Triangle']
        bearish_chart_patterns = ['Double Top']
        
        for pattern in chart_patterns['patterns']:
            if any(bp in pattern for bp in bullish_chart_patterns):
                score += 15
                reasons.append(f"תבנית גרף בוליש: {pattern}")
            elif any(bp in pattern for bp in bearish_chart_patterns):
                score -= 10
                reasons.append(f"תבנית גרף דוביש: {pattern}")
            else:
                score += 5
                reasons.append(f"תבנית גרף: {pattern}")
    
    # בדיקת נפח (15 נקודות מקסימום)
    avg_volume, volume_ratio = calculate_volume_indicators(data)
    if volume_ratio > 1.5:  # נפח גבוה מאוד
        score += 15
        reasons.append(f"נפח גבוה מאוד מהממוצע ({volume_ratio}x)")
    elif volume_ratio > 1.2:  # נפח גבוה מהממוצע
        score += 10
        reasons.append(f"נפח גבוה מהממוצע ({volume_ratio}x)")
    elif volume_ratio < 0.5:  # נפח נמוך
        score -= 5
        reasons.append(f"נפח נמוך ({volume_ratio}x)")
    
    # חישוב אחוז הציון הסופי
    final_score = max(0, min(100, round((score / max_score) * 100, 1)))
    
    return {
        'is_good_entry': final_score >= 60,  # סף 60% או יותר
        'score': final_score,
        'reasons': reasons,
        'risk_level': 'Low' if final_score >= 75 else 'Medium' if final_score >= 60 else 'High'
    }

def run_trending_analysis(api_key, telegram_token=None, telegram_chat_id=None):
    """
    Analyze trending stocks and report statistics
    
    Args:
        api_key (str): Google Gemini API key
        telegram_token (str): Telegram bot token
        telegram_chat_id (str): Telegram chat ID
    
    Returns:
        dict: Analysis statistics
    """
    print("🔥 Fetching trending stocks...")
    tickers = get_trending_stocks(api_key)
    print(f"Found {len(tickers)} trending stocks: {', '.join(tickers[:10])}...")
    
    # Track analysis stats
    analyzed = 0
    good_entry_points = 0
    results = []  # Store results for summary report
    
    for ticker in tickers:
        print(f"\n{'='*50}")
        
        result = analyze_single_stock(
            ticker=ticker,
            api_key=api_key,
            send_telegram_flag=bool(telegram_token and telegram_chat_id),
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id
        )
        
        results.append(result)  # Collect result for this ticker
        
        if result['success']:
            analyzed += 1
            if result.get('is_good_entry', False):
                print("✅ Potential good entry point detected!")
                good_entry_points += 1
        
        # Add delay between API calls to avoid rate limiting
        if ticker != tickers[-1]:
            time.sleep(1)
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"📊 ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Stocks analyzed: {analyzed}")
    print(f"Good entry points found: {good_entry_points}")
    print(f"Success rate: {(good_entry_points/analyzed*100):.1f}%" if analyzed > 0 else "N/A")
    
    # Generate and print summary report
    summary_report = generate_summary_report(results)
    print(f"\n{summary_report}")
    
    return {
        'analyzed': analyzed,
        'good_entry_points': good_entry_points,
        'success_rate': (good_entry_points/analyzed*100) if analyzed > 0 else 0
    }

def rank_stocks_by_potential(results):
    """
    Rank analyzed stocks by their trading potential
    
    Args:
        results (list): List of analysis results
    
    Returns:
        list: Sorted list by trading potential (best first)
    """
    valid_results = [r for r in results if r['success']]
    
    # Sort by entry score (highest first)
    sorted_results = sorted(valid_results, 
                          key=lambda x: x.get('entry_score', 0), 
                          reverse=True)
    
    return sorted_results

def generate_summary_report(results):
    """
    Generate a comprehensive summary report
    
    Args:
        results (list): List of analysis results
    
    Returns:
        str: Formatted summary report
    """
    ranked_stocks = rank_stocks_by_potential(results)
    total_analyzed = len([r for r in results if r['success']])
    good_entries = len([r for r in ranked_stocks if r['is_good_entry']])
    
    report = f"""
📊 **Summary Report - Trending Stocks Analysis**
{'='*50}

📈 **General Statistics**:
- Stocks analyzed: {total_analyzed}
- Good entry points: {good_entries}
- Success rate: {(good_entries/total_analyzed*100):.1f}%

🏆 **Top Performing Stocks** (TOP 5):
"""
    
    for i, stock in enumerate(ranked_stocks[:5], 1):
        risk_mgmt = stock.get('risk_management', {})
        patterns = stock.get('patterns', {}).get('patterns', [])
        pattern_text = f" | 📋 {patterns[0]}" if patterns else ""
        
        report += f"""
{i}. **{stock['ticker']}** 
   🎯 Score: {stock.get('entry_score', 0)}% | ⚠️ Risk: {stock.get('risk_level', 'Unknown')}
   💰 Entry: ${risk_mgmt.get('entry_price', 'N/A')} | 🛑 Stop: ${risk_mgmt.get('stop_loss', 'N/A')} | 🎯 TP1: ${risk_mgmt.get('tp1', 'N/A')}{pattern_text}
"""
    
    if good_entries > 0:
        report += f"\n💡 **Recommendation**: Focus on stocks with score above 70% and low-medium risk"
    else:
        report += f"\n⚠️ **Recommendation**: Market currently doesn't present excellent swing trading opportunities"
    
    return report

# ======================== MAIN EXECUTION ========================


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description='Stock Technical Analysis Tool with AI Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stock_analysis_ai.py AAPL --gemini_api_key YOUR_KEY
  python stock_analysis_ai.py --trending_stocks --gemini_api_key YOUR_KEY
  python stock_analysis_ai.py TSLA --gemini_api_key YOUR_KEY --telegram_token BOT_TOKEN --telegram_chat_id CHAT_ID
        """
    )
    
    parser.add_argument('ticker', nargs='?', default='AMZN', 
                       help='Ticker symbol to analyze (default: AAPL)')
    parser.add_argument('--trending_stocks', action='store_true', 
                       help='Analyze trending stocks instead of specific ticker')
    parser.add_argument('--gemini_api_key', required=True,
                       help='Google Gemini API key (required)')
    parser.add_argument('--telegram_token', 
                       help='Telegram bot token for sending messages')
    parser.add_argument('--telegram_chat_id', 
                       help='Telegram chat ID to send messages')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.gemini_api_key:
        print("❌ Error: --gemini_api_key is required")
        sys.exit(1)
    
    if (args.telegram_token and not args.telegram_chat_id) or (args.telegram_chat_id and not args.telegram_token):
        print("❌ Error: Both --telegram_token and --telegram_chat_id are required for Telegram functionality")
        sys.exit(1)
    
    # Show startup information
    print("🚀 Stock Technical Analysis Tool")
    print("=" * 50)
    print(f"Mode: {'Trending Stocks Analysis' if args.trending_stocks else f'Single Stock Analysis ({args.ticker})'}")
    print(f"Telegram: {'Enabled' if args.telegram_token else 'Disabled'}")
    print("=" * 50)
    
    try:
        if args.trending_stocks:
            # Run trending stocks analysis
            stats = run_trending_analysis(
                api_key=args.gemini_api_key,
                telegram_token=args.telegram_token,
                telegram_chat_id=args.telegram_chat_id
            )
            
        else:
            # Run single stock analysis
            result = analyze_single_stock(
                ticker=args.ticker.upper(),
                api_key=args.gemini_api_key,
                send_telegram_flag=bool(args.telegram_token and args.telegram_chat_id),
                telegram_token=args.telegram_token,
                telegram_chat_id=args.telegram_chat_id
            )
            
            if result['success']:
                print(f"\n✅ Analysis completed successfully for {result['ticker']}")
                if result.get('is_good_entry'):
                    print("🎯 This appears to be a good entry point!")
            else:
                print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        print("\n🎉 Analysis completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)