# StonkJournal Trade Automation

A Python automation script using Playwright to automatically enter trades into StonkJournal from Interactive Brokers CSV reports or single trade entries.

## Features

- 🤖 **Automated Trade Entry**: Automatically log in and enter trades into StonkJournal
- 📊 **CSV Batch Processing**: Parse Interactive Brokers activity reports and process multiple trades
- 🔄 **Smart Trade Management**: Detects existing open trades and adds actions instead of creating duplicates
- 🌍 **Timezone Conversion**: Automatically converts EDT/EST to Israel timezone
- 💰 **Commission Handling**: Converts negative commission values to positive
- 🎯 **Filtering**: Processes only EXECUTION-level records with USD currency
- ⏱️ **Chronological Processing**: Sorts trades by DateTime before execution
- 🔍 **Status Filtering**: Automatically filters to show only OPEN trades

## Prerequisites

- Python 3.11+
- Playwright for Python
- pytz for timezone conversion

## Installation

1. Install required packages:

```bash
pip install playwright pytz
```

2. Install Playwright browsers:

```bash
playwright install chromium
```

## Usage

### Single Trade Mode

Enter a single trade manually:

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

### CSV Batch Mode

Process multiple trades from an Interactive Brokers CSV report:

```bash
python3.11 main.py \
  --username "your-email@example.com" \
  --password "your-password" \
  --csv-file "/path/to/your/flex.csv"
```

### Headful Mode (Show Browser)

By default, the script runs in headless mode. To see the browser:

```bash
python3.11 main.py \
  --username "your-email@example.com" \
  --password "your-password" \
  --csv-file "/path/to/your/flex.csv" \
  --headful
```

## Command Line Arguments

### Required Arguments
- `--username`: Your StonkJournal email/username
- `--password`: Your StonkJournal password

### CSV Mode Arguments
- `--csv-file`: Path to Interactive Brokers CSV report file

### Single Trade Mode Arguments
- `--symbol`: Stock symbol (e.g., TSLA)
- `--quantity`: Number of shares
- `--price`: Price per share
- `--fee`: Trading fee/commission
- `--action`: Trade action (BUY or SELL)
- `--datetime`: Trade date/time in format MM/DD/YYYY,HH:MM (optional)

### Optional Arguments
- `--headful`: Run in headed mode to see the browser (default: headless)

## CSV File Format

The script expects Interactive Brokers Daily Activity CSV reports with the following columns:

- `LevelOfDetail`: Should contain "EXECUTION" for trade records
- `CurrencyPrimary`: Should be "USD" (non-USD trades are filtered out)
- `Symbol`: Stock ticker symbol
- `DateTime`: Format "MM/DD/YYYY,HH:MM:SS EDT" or "MM/DD/YYYY,HH:MM:SS EST"
- `Quantity`: Number of shares
- `IBCommission`: Commission fee (negative values are converted to positive)
- `Buy/Sell`: Trade action (BUY or SELL)
- `TradePrice`: Execution price per share

## How It Works

1. **Login**: Authenticates with StonkJournal using provided credentials
2. **Page Verification**: Confirms the dashboard loaded successfully
3. **Filter Setup**: Applies OPEN trades filter
4. **Trade Processing**: For each trade:
   - Reloads page and reapplies filter
   - Checks if symbol exists in open trades
   - If exists: Adds action to existing trade
   - If new: Creates new trade entry
5. **Summary**: Reports success/failure counts

## Data Transformations

### Timezone Conversion
- **Input**: EDT/EST (America/New_York)
- **Output**: Israel Time (Asia/Jerusalem)
- **Format**: MM/DD/YYYY,HH:MM

### Commission
- Converts negative values to positive (e.g., -2.50 → 2.50)

### Filtering
- Only processes records where:
  - `LevelOfDetail == 'EXECUTION'`
  - `CurrencyPrimary == 'USD'`

### Sorting
- Trades are sorted chronologically by DateTime before processing

## Error Handling

- **Login Retry**: Attempts login up to 5 times with page refresh
- **Page Verification**: Retries page load verification with refresh
- **Multiple Encoding Support**: Tries utf-8, latin-1, iso-8859-1, cp1252 for CSV files
- **OPEN Selection Fallback**: Three different approaches to select OPEN status
- **Screenshots**: Saves error screenshots to help debug issues

## Output

The script provides detailed logging of each step:

```
=============================================================
StonkJournal Trade Automation
=============================================================
Username: your-email@example.com
Processing Mode: CSV Batch
CSV File: /path/to/flex.csv
Total trades to process: 5
=============================================================

[INFO] Found 5 EXECUTION records
[INFO] Trades sorted chronologically
[INFO] Processed 5 trades from CSV

Processing Trade 1/5
=============================================================
Symbol: TSLA
Action: BUY 10 shares @ $250.50
Fee: $2.0
DateTime: 10/08/2025,09:30

✓ Trade 1 completed successfully!

...

=============================================================
EXECUTION SUMMARY
=============================================================
Total trades processed: 5
Successful: 5
Failed: 0
=============================================================

✓ All trades completed successfully!
```

## Troubleshooting

### Login Issues
- Ensure credentials are correct
- Check if StonkJournal site is accessible
- Review error screenshots saved in the directory

### CSV Parsing Issues
- Verify CSV file format matches Interactive Brokers format
- Check for encoding issues (script tries multiple encodings)
- Ensure LevelOfDetail and CurrencyPrimary columns exist

### Trade Entry Issues
- Check if page selectors are still valid (website changes may break automation)
- Run in `--headful` mode to observe browser behavior
- Review console output for specific error messages

## Notes

- The script maintains a single browser session for all trades in batch mode
- Page is reloaded and filtered before each trade insertion to ensure clean state
- Existing trades are detected to avoid duplicates - actions are added instead
- All times are converted to Israel timezone automatically

## Security Warning

⚠️ **Never commit credentials to version control!**

Consider using environment variables:

```bash
export STONKJOURNAL_USER="your-email@example.com"
export STONKJOURNAL_PASS="your-password"

python3.11 main.py \
  --username "$STONKJOURNAL_USER" \
  --password "$STONKJOURNAL_PASS" \
  --csv-file "trades.csv"
```

## License

Private use only.

## Author

Created for automating Interactive Brokers trade entry into StonkJournal.
