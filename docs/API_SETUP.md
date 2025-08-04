# API Keys and Setup Guide

This guide helps you obtain and configure the necessary API keys for all tools in this repository.

## 📋 Required API Keys

### 1. Google Gemini API Key (Required for all tools)
**Purpose**: AI-powered analysis and content generation

**How to get it**:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

**Usage**:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

**Free Tier**: 60 requests per minute, suitable for personal use

---

### 2. Telegram Bot Token (Optional - for notifications)
**Purpose**: Send analysis reports and summaries to Telegram

**How to get it**:
1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/newbot` command
3. Follow the prompts to name your bot
4. Copy the provided HTTP API token

**Get Chat ID**:
1. Add your bot to a chat or channel
2. Send a message to the bot
3. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Look for "chat":{"id": in the response

**Usage**:
```bash
export TELEGRAM_BOT_TOKEN="1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="your_chat_id"
```

---

### 3. YouTube Channel ID (Optional - for monitoring)
**Purpose**: Monitor specific YouTube channels for new videos

**How to get it**:
1. Go to the YouTube channel you want to monitor
2. View page source (Ctrl+U)
3. Search for "channelId" or "browse_id"
4. Copy the ID (starts with UC)

**Alternative method**:
1. Use online tools like "YouTube Channel ID Finder"
2. Enter the channel URL
3. Copy the generated ID

**Usage**:
```bash
export YOUTUBE_CHANNEL_ID="UCxxxxxxxxxxxxxxxxxxxx"
```

---

## 🔧 Environment Setup

### Option 1: Environment Variables
Create a file called `.env` in the project root:
```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
YOUTUBE_CHANNEL_ID=your_channel_id
```

Then load it:
```bash
export $(cat .env | xargs)
```

### Option 2: Shell Profile
Add to your `~/.zshrc` or `~/.bash_profile`:
```bash
export GEMINI_API_KEY="your_gemini_api_key"
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export YOUTUBE_CHANNEL_ID="your_channel_id"
```

### Option 3: Command Line Arguments
Pass keys directly to the scripts:
```bash
python stock_analysis_ai.py AAPL --gemini_api_key "your_key"
python youtube-summerizer.py --gemini-key "your_key" --telegram-token "bot_token"
```

---

## 🔐 Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables or secure vaults**
3. **Regularly rotate API keys**
4. **Use read-only permissions where possible**
5. **Monitor API usage and set up alerts**

---

## 🧪 Testing Your Setup

### Test Gemini API
```bash
python -c "
import google.generativeai as genai
genai.configure(api_key='your_api_key')
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content('Hello, world!')
print(response.text)
"
```

### Test Telegram Bot
```bash
curl -X GET "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getMe"
```

---

## 📞 Support

If you encounter issues:
1. Check the API key format and validity
2. Verify network connectivity
3. Review rate limits and quotas
4. Check the tool-specific README files
5. Enable debug logging for detailed error messages
