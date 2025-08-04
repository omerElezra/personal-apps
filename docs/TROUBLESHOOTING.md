# Troubleshooting Guide

Common issues and solutions for the personal automation tools.

## 🔧 Installation Issues

### Python Version Compatibility
**Problem**: `ModuleNotFoundError` or compatibility errors
**Solution**:
```bash
# Check Python version (3.8+ required)
python --version

# Use specific Python version if needed
python3.8 -m pip install -r requirements.txt
python3.8 stock_analysis_ai.py
```

### Package Installation Failures
**Problem**: Failed to install packages like `mplfinance` or `scipy`
**Solution**:
```bash
# Update pip first
pip install --upgrade pip

# Install with verbose output to see errors
pip install -r requirements.txt -v

# For macOS users with ARM chips
pip install --no-cache-dir -r requirements.txt
```

### Missing System Dependencies
**Problem**: Compilation errors during installation
**Solution** (macOS):
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python-tk
```

---

## 🔑 API Issues

### Gemini API Authentication
**Problem**: `401 Unauthorized` or `Invalid API key`
**Solutions**:
1. Verify API key format (should start with `AIza`)
2. Check if API is enabled in Google Cloud Console
3. Ensure no extra spaces or quotes in the key
4. Test with a simple request:
```bash
curl -H "Content-Type: application/json" \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
     -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=YOUR_API_KEY"
```

### Gemini API Rate Limits
**Problem**: `429 Too Many Requests`
**Solutions**:
1. Reduce request frequency
2. Add delays between requests
3. Check your quota in Google AI Studio
4. Upgrade to paid tier if needed

### Telegram Bot Issues
**Problem**: Bot not responding or messages not sending
**Solutions**:
1. Verify bot token format (should be like `1234567890:ABC...`)
2. Check if bot is active:
```bash
curl "https://api.telegram.org/bot<TOKEN>/getMe"
```
3. Ensure bot has permission to send messages to the chat
4. For channels, make sure bot is added as an admin

---

## 📊 Stock Analysis Issues

### Yahoo Finance Data Errors
**Problem**: `No data found` or empty dataframes
**Solutions**:
1. Check if ticker symbol is correct (use uppercase, e.g., `AAPL`)
2. Verify market is open or use a different ticker
3. Add delay between requests:
```python
import time
time.sleep(1)  # Add 1 second delay
```

### Chart Generation Failures
**Problem**: `Cannot save figure` or display issues
**Solutions**:
1. Ensure matplotlib backend is properly configured:
```python
import matplotlib
matplotlib.use('Agg')  # For headless environments
```
2. Check file permissions in the output directory
3. Clear matplotlib cache:
```bash
rm -rf ~/.matplotlib
```

### Technical Indicator Calculation Errors
**Problem**: `NaN` values or calculation errors
**Solutions**:
1. Ensure sufficient data points (need 150+ days for SMA150)
2. Check for missing data in the dataset
3. Verify data types are numeric

---

## 🎥 YouTube Issues

### Transcript Extraction Failures
**Problem**: `TranscriptsDisabled` or `NoTranscriptFound`
**Solutions**:
1. Check if the video has captions enabled
2. Try different language codes: `['en', 'he', 'iw', 'auto']`
3. Some videos may not have transcripts available

### Video URL Format Issues
**Problem**: Invalid URL format errors
**Solutions**:
Supported formats:
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://youtube.com/watch?v=VIDEO_ID`

---

## 🐛 General Debugging

### Enable Debug Logging
Add to the beginning of any script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check File Permissions
```bash
# Ensure scripts are executable
chmod +x *.py

# Check current directory permissions
ls -la
```

### Network Connectivity
```bash
# Test basic connectivity
ping google.com

# Test HTTPS connectivity
curl -I https://api.openai.com
curl -I https://api.telegram.org
```

### Memory Issues
**Problem**: Script crashes with memory errors
**Solutions**:
1. Reduce data timeframe for analysis
2. Process data in smaller chunks
3. Clear variables when done:
```python
del large_dataframe
import gc
gc.collect()
```

---

## 📝 Log File Analysis

### Stock Analysis Logs
Check for errors in:
- Console output
- Chart generation files
- Network request failures

### YouTube Summarizer Logs
Log file location: `youtube_summary.log`
```bash
# View recent logs
tail -f youtube_summary.log

# Search for errors
grep -i error youtube_summary.log
```

---

## 🆘 Getting Help

If issues persist:

1. **Check the issue tracker** for known problems
2. **Enable verbose logging** and capture the full error
3. **Provide system information**:
   - Operating system and version
   - Python version
   - Package versions: `pip list`
4. **Include relevant log files** and error messages
5. **Test with minimal examples** to isolate the problem

### Useful Commands for Debugging
```bash
# System info
python --version
pip --version
pip list | grep -E "(pandas|matplotlib|requests)"

# Test imports
python -c "import pandas, matplotlib, requests; print('All imports successful')"

# Check environment variables
echo $GEMINI_API_KEY | head -c 10  # Show first 10 chars only
env | grep -E "(TELEGRAM|YOUTUBE|GEMINI)"
```
