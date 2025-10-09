# StonkJournal Trade Automation Platform

A comprehensive automated trading journal system that detects Interactive Brokers CSV reports in Gmail, processes them through a webhook service, and automatically enters trades into StonkJournal using browser automation.

## 🎯 System Overview

This platform consists of three main components working together:

```
Gmail (Interactive Brokers Reports)
    ↓ (Google Apps Script detects new emails)
    ↓
Webhook Service (FastAPI + Docker)
    ↓ (Downloads latest automation script)
    ↓
Trade Automation Script (Playwright)
    ↓
StonkJournal (Trade entries created)
```

### Components

1. **Google Apps Script (`detectReportGmail.gs`)**: Monitors Gmail for Interactive Brokers CSV reports
2. **Webhook Service (`app.py`)**: FastAPI service that receives CSV files and orchestrates automation
3. **Trade Automation Script (`main.py`)**: Playwright-based browser automation for StonkJournal
4. **Docker Container (`Dockerfile`)**: Containerized deployment for the webhook service

## 🚀 Features

### Gmail Monitoring
- ✅ Automatic detection of Interactive Brokers activity reports
- ✅ Smart deduplication to prevent duplicate processing
- ✅ Concurrent execution prevention with locking
- ✅ Automatic labeling of processed emails
- ✅ Configurable lookback window (default: 24 hours)
- ✅ Retry logic for failed webhook calls
- ✅ **Email alerts on webhook failures** (sends error details to your StonkJournal email)

### Trade Processing
- ✅ Filters trades by criteria: `LevelOfDetail='EXECUTION'` AND `CurrencyPrimary='USD'`
- ✅ Chronological processing (sorted by DateTime)
- ✅ Timezone conversion (EDT/EST → Israel timezone)
- ✅ Commission normalization (negative → positive)
- ✅ Smart duplicate detection (adds actions to existing trades)
- ✅ Batch processing of multiple trades in one session

### Webhook Service
- ✅ Bearer token authentication
- ✅ Health check endpoint
- ✅ Comprehensive logging
- ✅ Timeout protection (30 minutes)
- ✅ Automatic script updates from GitHub
- ✅ Detailed error reporting
- ✅ **Error screenshot capture and forwarding**

### Error Monitoring & Alerts
- ✅ Automatic screenshot capture on errors (login, verification, trade insertion)
- ✅ Email alerts sent to your StonkJournal email with:
  - Error details and HTTP status codes
  - Original CSV filename and email subject
  - **Attached PNG screenshot** for visual debugging
  - Troubleshooting steps
  - Cloud Run log commands
- ✅ Automatic retry on next Gmail monitoring run

## 📋 Prerequisites

### For Gmail Monitoring
- Google account with Gmail access
- Interactive Brokers account sending reports to your Gmail

### For Webhook Service
- Docker (for containerized deployment)
- OR Python 3.11+ (for local development)

### For StonkJournal Automation
- StonkJournal account with valid credentials
- Python 3.11+
- Playwright browser automation library

## 🛠️ Installation & Setup

### 1. Google Apps Script Setup

1. Go to [Google Apps Script](https://script.google.com/)
2. Create a new project
3. Copy the contents of `detectReportGmail.gs` into the script editor
4. Set up Script Properties (Project Settings → Script Properties):
   ```
   USERNAME        = your-stonkjournal-email@example.com
   PASSWORD        = your-stonkjournal-password
   WEBHOOK_URL     = https://your-webhook-service.com/ingest
   BEARER_TOKEN    = your-secret-bearer-token
   ```
5. Configure a time-based trigger:
   - Function: `processNewReports`
   - Deployment: Head
   - Event source: Time-driven
   - Type: Minutes timer
   - Interval: Every 5 minutes (or your preference)

6. Test manually by running `debugOnce()` function

### 2. Webhook Service Deployment

#### Option A: Docker Deployment (Local Development)

```bash
# Clone the repository
git clone https://github.com/omerElezra/personal-apps.git
cd personal-apps/playwright-stonkjornal

# Build the Docker image
docker build -t stonkjournal-automation .

# Run the container
docker run -d \
  -p 8080:8080 \
  -e BEARER_TOKEN="your-secret-bearer-token" \
  --name stonkjournal-service \
  stonkjournal-automation
```

#### Option B: Local Development

```bash
# Install dependencies
pip install fastapi uvicorn requests playwright pytz tenacity

# Install Playwright browsers
python -m playwright install --with-deps chromium

# Set environment variable
export BEARER_TOKEN="your-secret-bearer-token"

# Run the service
uvicorn app:app --host 0.0.0.0 --port 8080
```

#### Option C: Google Cloud Run Deployment (Recommended for Production)

Deploy to Google Cloud Run with automatic scaling and pay-per-use pricing:

**Method 1: Build and Deploy from Source (Easiest)**
```bash
# Deploy directly from source code (Cloud Build handles the image)
gcloud run deploy gmail-csv-webhook \
  --project=stocks-report-474512 \
  --region=me-west1 \
  --source . \
  --allow-unauthenticated \
  --set-env-vars "BEARER_TOKEN=$(openssl rand -hex 32)" \
  --cpu=1 \
  --memory=1Gi \
  --timeout=1800 \
  --max-instances=10 \
  --min-instances=0
```

**Method 2: Build Image First, Then Deploy**
```bash
# Step 1: Build the container image
gcloud builds submit \
  --project=stocks-report-474512 \
  --tag gcr.io/stocks-report-474512/stonkjournal-automation

# Step 2: Deploy the pre-built image
gcloud run deploy gmail-csv-webhook \
  --project=stocks-report-474512 \
  --region=me-west1 \
  --image gcr.io/stocks-report-474512/stonkjournal-automation \
  --allow-unauthenticated \
  --set-env-vars "BEARER_TOKEN=$(openssl rand -hex 32)" \
  --cpu=1 \
  --memory=1Gi \
  --timeout=1800 \
  --max-instances=10 \
  --min-instances=0
```

**Method 3: Deploy Without Rebuilding (Update Existing Service)**
```bash
# Update environment variables or settings without rebuilding
gcloud run services update gmail-csv-webhook \
  --project=stocks-report-474512 \
  --region=me-west1 \
  --update-env-vars "BEARER_TOKEN=your-new-token"

# Or scale resources
gcloud run services update gmail-csv-webhook \
  --project=stocks-report-474512 \
  --region=me-west1 \
  --cpu=2 \
  --memory=2Gi
```

**Configuration Options Explained:**
- `--source .` - Build from current directory
- `--allow-unauthenticated` - Allow webhook calls without Google auth
- `--cpu=1` - 1 vCPU (can be 1, 2, 4, or 8)
- `--memory=1Gi` - 1GB RAM (minimum for Playwright)
- `--timeout=1800` - 30 minutes max execution time (for large CSV files)
- `--max-instances=10` - Scale up to 10 concurrent instances
- `--min-instances=0` - Scale to zero when idle (save costs)
- `$(openssl rand -hex 32)` - Generate secure random token

**Get the Deployed URL:**
```bash
gcloud run services describe gmail-csv-webhook \
  --project=stocks-report-474512 \
  --region=me-west1 \
  --format='value(status.url)'
```

#### Option D: Other Cloud Platforms

**AWS ECS/Fargate:**
```bash
# Use AWS CLI or Console to deploy the Docker image
aws ecs create-service --cluster my-cluster --service-name stonkjournal-automation
```

**Azure Container Instances:**
```bash
az container create --resource-group myResourceGroup \
  --name stonkjournal-automation \
  --image your-registry/stonkjournal-automation:latest \
  --environment-variables BEARER_TOKEN=your-token
```

### Google Cloud Run - Quick Commands

**Deploy from source (one command):**
```bash
gcloud run deploy gmail-csv-webhook --project=stocks-report-474512 --region=me-west1 --source . --allow-unauthenticated --set-env-vars "BEARER_TOKEN=$(openssl rand -hex 32)" --cpu=1 --memory=1Gi --timeout=1800
```

**Update without rebuild:**
```bash
gcloud run services update gmail-csv-webhook --project=stocks-report-474512 --region=me-west1 --update-env-vars "BEARER_TOKEN=new-token"
```

**Get service URL:**
```bash
gcloud run services describe gmail-csv-webhook --project=stocks-report-474512 --region=me-west1 --format='value(status.url)'
```

**View logs:**
```bash
gcloud run services logs tail gmail-csv-webhook --project=stocks-report-474512 --region=me-west1
```

### 3. Standalone Script Usage

You can also run the automation script directly:

#### Single Trade Entry
```bash
python3.11 main.py \
  --username "your-email@example.com" \
  --password "your-password" \
  --symbol TSLA \
  --quantity 10 \
  --price 250.50 \
  --fee 2 \
  --action BUY \
  --datetime "10/08/2025,09:30"
```

#### CSV Batch Processing
```bash
python3.11 main.py \
  --username "your-email@example.com" \
  --password "your-password" \
  --csv-file "/path/to/interactive-brokers-report.csv"
```

## 📊 Data Flow

### 1. Gmail Detection Flow
```
New email arrives → Gmail filter matches → Apps Script triggered
→ Deduplication check → Extract CSV attachment → POST to webhook
→ Mark as processed → Apply label
```

### 2. Webhook Processing Flow
```
Receive CSV → Authenticate request → Download latest main.py
→ Save CSV to temp directory → Execute automation script
→ Return results → Log outcome
```

### 3. Trade Automation Flow
```
Login to StonkJournal → Parse CSV → Filter trades (EXECUTION + USD)
→ Sort chronologically → For each trade:
  → Reload page → Filter OPEN trades → Check for existing trade
  → If exists: Add action | If new: Create trade
→ Report summary
```

## 🔧 Configuration

### Google Apps Script Configuration

Edit the constants at the top of `detectReportGmail.gs`:

```javascript
const FROM_EMAIL = 'Info@inter-il.com';              // Sender email to filter
const SUBJECT_PHRASE = 'Activity Flex for ';         // Subject text to match
const LOOKBACK_HOURS = 24;                           // How far back to search
const LABEL_AFTER_PROCESS = 'Processed/CSV-Reports'; // Label for processed emails
const DEDUP_KEY = 'processedMessageIds';             // Storage key for dedup
```

### Webhook Service Configuration

Environment variables:
- `BEARER_TOKEN`: Secret token for authentication (required)
- `PORT`: Service port (default: 8080)

### Trade Automation Configuration

The script filters trades based on:
- `LevelOfDetail == 'EXECUTION'`
- `CurrencyPrimary == 'USD'`

Modify `parse_csv_file()` in `main.py` to adjust filters.

## 📡 API Documentation

### Health Check
```http
GET /healthz
```

**Response:**
```json
{
  "status": "healthy",
  "service": "stonkjournal-automation"
}
```

### Process CSV Report
```http
POST /ingest
Authorization: Bearer YOUR_TOKEN
X-Username: your-stonkjournal-email@example.com
X-Password: your-stonkjournal-password
X-Filename: report.csv (optional)
Content-Type: text/csv

[CSV file content]
```

**Success Response (200 OK):**
```
Total trades processed: 5
Successful: 5
Failed: 0
```

**No Matching Trades (200 OK):**
```
No trades matching filters found in CSV file.
Filters applied: LevelOfDetail='EXECUTION' AND CurrencyPrimary='USD'
Nothing to process. Exiting successfully.
```

**Error Response (500):**
```json
{
  "detail": "Error message with full output"
}
```

## 📝 CSV File Format

Expected Interactive Brokers Daily Activity CSV format:

| Column | Description | Example |
|--------|-------------|---------|
| LevelOfDetail | Record type | EXECUTION |
| CurrencyPrimary | Currency | USD |
| Symbol | Stock ticker | TSLA |
| DateTime | Trade timestamp | 10/03/2025,09:47:03 EDT |
| Quantity | Number of shares | 10 |
| IBCommission | Commission fee | -2.50 |
| Buy/Sell | Trade action | BUY |
| TradePrice | Price per share | 250.50 |

## 🔒 Security

### Authentication
- Bearer token authentication for webhook service
- Credentials passed securely via headers (not URL parameters)
- Script properties in Google Apps Script are encrypted

### Best Practices
- ⚠️ Never commit credentials to version control
- ⚠️ Use environment variables for sensitive data
- ⚠️ Rotate bearer tokens regularly
- ⚠️ Use HTTPS for webhook URLs
- ⚠️ Keep Script Properties secure (don't share project)

### Recommended: Use Secret Management
```bash
# Example using environment variables
export STONKJOURNAL_USER="your-email@example.com"
export STONKJOURNAL_PASS="your-password"
export BEARER_TOKEN="your-secret-token"
```

## 🐛 Troubleshooting

### Gmail Script Issues

**Problem**: No emails being processed
- Check Script Properties are set correctly
- Verify trigger is enabled
- Check execution logs in Apps Script
- Test manually with `debugOnce()`

**Problem**: Duplicate processing
- Deduplication storage may be full (max 2000 entries)
- Check if Message-ID is unique
- Review PropertiesService quota

### Webhook Issues

**Problem**: 401 Unauthorized
- Verify Bearer token matches in both Apps Script and webhook service
- Check Authorization header format: `Bearer YOUR_TOKEN`

**Problem**: 500 Server Error
- Check webhook service logs
- Verify main.py URL is accessible
- Check CSV file format
- Review execution timeout (30 minutes default)

### Automation Issues

**Problem**: Login fails
- Verify StonkJournal credentials
- Check if website structure changed
- Run in headful mode to observe: `--headful`
- Check error screenshots saved in directory

**Problem**: Trades not found
- Verify CSV has EXECUTION records
- Check CurrencyPrimary is USD
- Review filter criteria in `parse_csv_file()`

**Problem**: Duplicate trades
- Script should detect existing trades automatically
- Check symbol matching logic in `check_symbol_in_open_trades()`

## 📊 Monitoring & Logging

### Error Notifications

**Automatic Email Alerts**: When webhook errors occur, you'll automatically receive an email alert with:
- ✅ HTTP error code and message
- ✅ CSV filename that failed
- ✅ Original email details (subject, date)
- ✅ Webhook URL and timestamp
- ✅ Troubleshooting steps
- ✅ Next steps for resolution

**Email Configuration**: No setup required! The Google Apps Script automatically sends error alerts to your StonkJournal username email.

**Example Alert Email**:
```
Subject: ⚠️ StonkJournal Automation Error - U123456_20231015.csv

ERROR DETAILS:
HTTP Status Code: 500
Error Message: {"detail":"Execution error: ..."}

TROUBLESHOOTING:
1. Check if the webhook service is running
2. Verify BEARER_TOKEN is configured correctly
3. Check Cloud Run logs
```

### Google Apps Script Logs
View execution logs:
1. Open Apps Script project
2. Click "Executions" (left sidebar)
3. Review logs for each run

### Webhook Service Logs
```bash
# Docker logs
docker logs stonkjournal-service

# Follow live logs
docker logs -f stonkjournal-service
```

### Automation Script Output
The script provides detailed logging:
```
[INFO] Navigating to StonkJournal dashboard...
[INFO] ✓ Found email field
[INFO] ✓ Successfully logged in!
[INFO] Found 5 EXECUTION records
[INFO] ✓ Trades sorted chronologically
Processing Trade 1/5
✓ Trade 1 completed successfully!
```

## 🔄 Updates & Maintenance

### Updating the Automation Script
The webhook service automatically downloads the latest `main.py` from GitHub on each request. To update:

1. Push changes to GitHub
2. No need to restart the webhook service
3. Next CSV processing will use the latest version

### Updating the Webhook Service
```bash
# Rebuild Docker image
docker build -t stonkjournal-automation .

# Stop and remove old container
docker stop stonkjournal-service
docker rm stonkjournal-service

# Run new container
docker run -d -p 8080:8080 -e BEARER_TOKEN="your-token" \
  --name stonkjournal-service stonkjournal-automation
```

### Updating Google Apps Script
1. Edit the script in Apps Script editor
2. Save the changes
3. Script will use new version on next trigger

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Gmail Inbox                         │
│  (Interactive Brokers sends daily CSV reports)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Monitored by
                     ↓
┌─────────────────────────────────────────────────────────┐
│            Google Apps Script Trigger                   │
│  • Runs every 5 minutes (configurable)                  │
│  • Searches for new emails matching criteria            │
│  • Deduplication check                                  │
│  • Extracts CSV attachments                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ POST /ingest
                     ↓
┌─────────────────────────────────────────────────────────┐
│           FastAPI Webhook Service (Docker)              │
│  • Authenticates request (Bearer token)                 │
│  • Downloads latest main.py from GitHub                 │
│  • Saves CSV to temp directory                          │
│  • Executes automation script                           │
│  • Returns results                                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ subprocess.run()
                     ↓
┌─────────────────────────────────────────────────────────┐
│          Trade Automation Script (Playwright)           │
│  • Parses CSV and filters trades                        │
│  • Converts timezones (EDT → Israel)                    │
│  • Logs into StonkJournal                               │
│  • Processes each trade sequentially                    │
│  • Creates/updates trades                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Headless browser
                     ↓
┌─────────────────────────────────────────────────────────┐
│                  StonkJournal Website                   │
│  • Trades automatically entered                         │
│  • No manual data entry required                        │
└─────────────────────────────────────────────────────────┘
```

## 📂 File Structure

```
playwright-stonkjornal/
├── app.py                   # FastAPI webhook service
├── main.py                  # Playwright automation script
├── detectReportGmail.gs     # Google Apps Script for Gmail monitoring
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose configuration
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🤝 Contributing

This is a personal automation project. Feel free to fork and adapt for your needs.

## 📄 License

Private use only.

## 🙏 Acknowledgments

- Built for automating Interactive Brokers → StonkJournal workflow
- Uses Playwright for reliable browser automation
- Leverages Google Apps Script for email monitoring
- FastAPI for robust webhook service

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review execution logs
3. Test components individually
4. Run in headful mode to observe browser behavior

---

**Created by Omer Elezra** - Automated trading journal entry system