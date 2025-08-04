# Personal DevOps & Automation Projects

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A collection of practical automation tools and utilities built to streamline workflows, enhance productivity, and demonstrate modern infrastructure techniques. This repository contains various personal projects showcasing DevOps methodologies, API integrations, and automated data processing systems.

## 📂 Repository Structure

```
personal-apps/
├── financial-tools/           # Stock analysis & trading tools
│   ├── stock_analysis_ai.py   # AI-powered technical analysis tool
│   ├── requirements.txt       # Dependencies for financial tools
│   └── README.md              # Stock analysis documentation
├── youtube-automation/        # Video content processing tools
│   ├── youtube-summerizer.py  # AI video summarization tool
│   ├── requirements.txt       # Dependencies for YouTube tools
│   └── README.md              # YouTube automation documentation
├── docs/                      # Project documentation
└── README.md                  # This file
```

## 🚀 Projects Overview

### 📊 Financial Tools
**Location**: `financial-tools/`

Advanced stock analysis and trading automation tools:

- **Stock Analysis AI** (`stock_analysis_ai.py`): 
  - Real-time technical analysis with 20+ indicators
  - AI-powered trading insights using Google Gemini
  - Candlestick and chart pattern recognition
  - Risk management with automated stop-loss calculations
  - Telegram integration for alerts and reports
  - Support for single stocks and trending stock analysis

### 🎥 YouTube Automation
**Location**: `youtube-automation/`

Intelligent video content processing and distribution:

- **YouTube Summarizer** (`youtube-summerizer.py`):
  - Automated video transcript extraction
  - AI-powered content summarization
  - Multi-channel Telegram distribution
  - RSS feed monitoring for new uploads
  - Multi-language support (Hebrew, English, etc.)
  - Comprehensive logging and error handling

## 🔧 Technologies Demonstrated

### Core Technologies
- **Python 3.8+**: Modern Python development with type hints and best practices
- **AI & Machine Learning**: Google Gemini AI integration for content analysis
- **API Integration**: Multi-platform API consumption (YouTube, Telegram, Yahoo Finance)
- **Data Processing**: Advanced data manipulation with pandas, numpy, scipy
- **Visualization**: Professional charting with matplotlib and mplfinance

### DevOps & Infrastructure
- **Configuration Management**: Environment-based configuration and secrets management
- **Logging & Monitoring**: Comprehensive logging with rotation and structured output
- **Error Handling**: Robust error handling with retry logic and graceful degradation
- **Code Organization**: Modular architecture with separation of concerns
- **Documentation**: Comprehensive README files and inline documentation

### Financial Technology
- **Technical Analysis**: 20+ indicators including RSI, MACD, moving averages
- **Pattern Recognition**: Candlestick and chart pattern detection algorithms
- **Risk Management**: Automated stop-loss and profit target calculations
- **Real-time Data**: Live market data integration and processing
- **Multi-timeframe Analysis**: Support for different trading timeframes

### Automation & Communication
- **Process Automation**: Automated video monitoring and content processing
- **Multi-channel Distribution**: Simultaneous delivery to multiple Telegram channels
- **Intelligent Filtering**: Smart content filtering to avoid duplicates
- **Scheduled Operations**: Support for continuous monitoring and batch processing

## 🏆 Key Features

- **Modular Architecture**: Each tool is self-contained with its own dependencies and documentation
- **Production Ready**: Comprehensive error handling, logging, and monitoring capabilities
- **API-First Design**: Clean integration with multiple external services and APIs
- **Real-time Processing**: Live data analysis and immediate notification delivery
- **Multi-language Support**: Hebrew and English content processing capabilities
- **Scalable Design**: Tools designed for both single-use and continuous operation modes

## 🚦 Getting Started

Each project directory contains its own README with specific setup instructions. Generally:

### Quick Start for Financial Tools
```bash
cd financial-tools
pip install -r requirements.txt
export GEMINI_API_KEY="your_gemini_key"
python stock_analysis_ai.py AAPL --gemini_api_key "$GEMINI_API_KEY"
```

### Quick Start for YouTube Automation
```bash
cd youtube-automation
pip install -r requirements.txt
export GEMINI_API_KEY="your_gemini_key"
python youtube-summerizer.py --video-url "https://youtu.be/VIDEO_ID"
```

### Environment Variables Setup
Create a `.env` file or export the following variables:
```bash
# Required for AI features
export GEMINI_API_KEY="your_google_gemini_api_key"

# Optional for Telegram integration
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id_or_comma_separated_ids"

# Optional for YouTube monitoring
export YOUTUBE_CHANNEL_ID="channel_id_to_monitor"
```

## 🔄 Future Enhancements

### Planned Features
- **Containerization**: Docker containers for all tools with multi-arch support
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Web Interface**: Flask/FastAPI web interface for tool management
- **Database Integration**: PostgreSQL/SQLite for data persistence and analytics
- **Additional Automation**: More DevOps workflow automation tools

### Infrastructure Improvements
- **Kubernetes Deployment**: Helm charts for container orchestration
- **Monitoring Dashboard**: Grafana dashboards for tool performance monitoring
- **Alert Management**: Comprehensive alerting system with multiple notification channels
- **Configuration Management**: Centralized configuration with environment-specific overrides

## 📊 Project Statistics

| Tool | Lines of Code | Dependencies | Status |
|------|---------------|--------------|--------|
| Stock Analysis AI | ~1,400 | 10 packages | ✅ Production |
| YouTube Summarizer | ~185 | 4 packages | ✅ Production |

## 🛠️ Development

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git for version control

### Local Development Setup
```bash
# Clone the repository
git clone https://github.com/your-username/personal-apps.git
cd personal-apps

# Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install tool-specific dependencies
cd financial-tools && pip install -r requirements.txt
cd ../youtube-automation && pip install -r requirements.txt
```

### Code Quality
- **Type Hints**: Python type annotations where applicable
- **Error Handling**: Comprehensive exception handling with meaningful messages
- **Logging**: Structured logging with different levels and rotation
- **Documentation**: Inline comments and comprehensive README files

## 📖 Documentation

- **[API Setup Guide](docs/API_SETUP.md)**: Step-by-step guide to obtain and configure API keys
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[Financial Tools README](financial-tools/README.md)**: Detailed documentation for stock analysis tools
- **[YouTube Automation README](youtube-automation/README.md)**: Detailed documentation for video processing tools

## 🧪 Testing Your Installation

Run the setup verification script:
```bash
python verify_setup.py
```

This will check:
- Python version compatibility
- Required packages installation
- API key configuration
- Basic functionality tests

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
