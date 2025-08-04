# Stock Analysis AI Tool

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An intelligent stock analysis tool that combines technical analysis with AI-powered insights for swing trading decisions.

## 🎯 Features

- **Real-time Data**: Fetches live stock data using Yahoo Finance API
- **Technical Analysis**: Advanced charting with 20+ indicators including:
  - Moving averages (SMA 20, SMA 150)
  - RSI, MACD, ATR indicators
  - Candlestick pattern recognition (9 major patterns)
  - Chart pattern detection (triangles, double tops/bottoms, etc.)
  - Volume analysis and momentum indicators
- **AI Analysis**: Google Gemini AI provides professional trading insights in Hebrew
- **Risk Management**: Automatic calculation of entry points, stop losses, and profit targets
- **Telegram Integration**: Sends analysis reports and charts directly to Telegram channels
- **Multiple Analysis Modes**:
  - Single stock analysis
  - Trending stocks analysis
  - Batch analysis with ranking

## 🔧 Technical Stack

- **Python 3.8+**
- **Data Sources**: Yahoo Finance API
- **AI Model**: Google Gemini 1.5 Flash
- **Visualization**: matplotlib, mplfinance
- **Data Processing**: pandas, numpy, scipy
- **Communication**: Telegram Bot API

## 📦 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/personal-apps.git
cd personal-apps/financial-tools
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
export GEMINI_API_KEY="your_gemini_api_key"
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"  # Optional
export TELEGRAM_CHAT_ID="your_chat_id"              # Optional
```

## 🚀 Usage

### Basic Stock Analysis
```bash
# Analyze Apple stock
python stock_analysis_ai.py AAPL --gemini_api_key "your_api_key"

# Analyze with Telegram notifications
python stock_analysis_ai.py TSLA \
  --gemini_api_key "your_api_key" \
  --telegram_token "bot_token" \
  --telegram_chat_id "chat_id"
```

### Trending Stocks Analysis
```bash
# Analyze top trending stocks
python stock_analysis_ai.py --trending_stocks \
  --gemini_api_key "your_api_key" \
  --telegram_token "bot_token" \
  --telegram_chat_id "chat_id"
```

### Command Line Options
```
positional arguments:
  ticker                Ticker symbol to analyze (default: AAPL)

options:
  --trending_stocks     Analyze trending stocks instead of specific ticker
  --gemini_api_key      Google Gemini API key (required)
  --telegram_token      Telegram bot token for sending messages
  --telegram_chat_id    Telegram chat ID to send messages
```

## 📊 Output

The tool generates:

1. **Technical Analysis Chart**: High-resolution PNG chart with:
   - Candlestick price data
   - Moving averages (SMA 20, SMA 150)
   - Volume bars
   - Current price indicators

2. **AI Analysis Report**: Comprehensive Hebrew report including:
   - Market trend analysis
   - Entry point evaluation (0-100 score)
   - Risk management recommendations
   - Candlestick and chart pattern analysis
   - Support and resistance levels

3. **Telegram Notification**: Formatted message with:
   - Entry score and risk assessment
   - Key technical levels
   - Pattern recognition results
   - Risk management details

## 🎯 Technical Indicators

### Price Action Analysis
- **Moving Averages**: SMA 20 (short-term), SMA 150 (long-term)
- **Crossover Detection**: Golden cross and death cross identification
- **Trend Analysis**: Multi-timeframe trend confirmation

### Pattern Recognition
- **Candlestick Patterns**: Doji, Hammer, Shooting Star, Engulfing patterns, Morning/Evening Star
- **Chart Patterns**: Double tops/bottoms, triangles, cup and handle, support/resistance levels

### Volume & Momentum
- **Volume Analysis**: Volume ratio vs 20-day average
- **Momentum Indicators**: RSI, MACD for trend confirmation
- **Volatility**: ATR for risk management calculations

### Risk Management
- **Entry Points**: Score-based evaluation (0-100)
- **Stop Losses**: ATR-based dynamic stops
- **Profit Targets**: Multiple TP levels based on technical analysis

## 🔐 API Keys Required

1. **Google Gemini API**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Telegram Bot Token**: Create bot via [@BotFather](https://t.me/botfather) (optional)

## 📝 Configuration

The script includes configurable parameters:
- Chart settings (size, DPI, style)
- Technical indicator periods
- Risk management ratios
- Color schemes and formatting

## 🚦 Status

✅ **Production Ready** - Fully functional with comprehensive error handling and logging.

## 📜 License

MIT License - see LICENSE file for details.
