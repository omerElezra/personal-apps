# Personal Projects Repository

![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A collection of practical automation tools and utilities showcasing modern infrastructure, data processing, and integration techniques.

## 📋 Projects Overview

### 📊 YouTube Summarizer
An intelligent automation tool that:
- Monitors YouTube channels for new content
- Extracts and processes video transcripts 
- Uses Gemini AI to generate concise summaries
- Distributes summaries to multiple Telegram channels simultaneously

```bash
python youtube-summerizer.py --video-url https://youtu.be/video_id
```

### 🔮 Future Projects
More automation tools and utilities are in development. Stay tuned!

## 🚀 Technical Skills Demonstrated

- **Automation**: Streamlined processes for content monitoring and distribution
- **API Integration**: YouTube API, Telegram Bot API, Google Gemini AI API
- **Python Development**: Structured code with proper error handling and logging
- **NLP & AI**: Intelligent content summarization using advanced AI models
- **Multi-channel Communication**: Distribution to multiple endpoints simultaneously

## 🛠️ Technologies Used

- Python 3.x
- Google Generative AI (Gemini)
- Telegram Bot API
- YouTube Transcript API
- Feedparser
- Logging with rotation
- Environment-based configuration

## 🏁 Getting Started

1. Clone the repository
2. Install dependencies:
```bash
pip install google-generativeai youtube-transcript-api requests feedparser
```

3. Set environment variables:
```bash
export GEMINI_API_KEY="your_gemini_api_key"
export TELEGRAM_BOT_TOKEN="your_telegram_token"
export TELEGRAM_CHAT_ID="id1,id2,id3"  # Comma-separated for multiple channels
export YOUTUBE_CHANNEL_ID="channel_id_to_monitor"
```

4. Run the project:
```bash
# Process a specific video
python youtube-summerizer.py --video-url https://youtu.be/video_id

# Or monitor a channel for new videos
python youtube-summerizer.py
```

## 🔄 CI/CD Pipeline (Coming Soon)
- Automated testing and deployment
- Docker containerization
- Kubernetes integration

## ⚙️ Architecture
This repository follows infrastructure-as-code and automation-first principles, applying DevOps methodologies to personal projects.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
