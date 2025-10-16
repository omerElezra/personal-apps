
import argparse
from datetime import datetime
import os
from playwright.sync_api import sync_playwright , TimeoutError, Page
import time
import csv
import pytz
from pathlib import Path
URL = "https://app.stonkjournal.com/dashboard"

def click_or_click_here(page: Page, *, open_same_tab: bool = False, timeout: int = 6000) -> Page:

    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(300)  # allow animation

    def _try_click(loc):
        loc.wait_for(state="visible", timeout=min(2500, timeout))
        if open_same_tab:
            # force same tab
            try:
                loc.evaluate("el => el.target = '_self'")
            except Exception:
                pass
            loc.click()
            page.wait_for_load_state("load")
            return page
        else:
            with page.context.expect_page() as pop:
                loc.click()
            newp = pop.value
            newp.wait_for_load_state("domcontentloaded")
            return newp

    selectors = [
        lambda c: c.locator("a.cta-link", has_text="or click here"),
    ]

    for frame in page.frames:
        if frame is page.main_frame:
            continue
        if frame.name == "sleek-widget":
            for build in selectors:
                try:
                    print(f"Trying selector in iframe: {build} , working...")
                    return _try_click(build(frame))
                except TimeoutError:
                    continue
                except Exception:
                    continue
    raise RuntimeError('Could not locate/click the "or click here" link.')

def parse_arguments(): 
                # Additional wait for network to settlearguments for trade automation"""
    p = argparse.ArgumentParser(description="StonkJournal Trade Automation (Playwright)")
    # Login
    p.add_argument("--username", required=True, help="StonkJournal username")
    p.add_argument("--password", required=True, help="StonkJournal password")
    p.add_argument("--csv-file", dest="csv_file", help="Path to CSV report file with trades")
    p.add_argument("--symbol", help="Stock symbol (default: TSLA)")
    p.add_argument("--quantity", help="Number of shares (default: 10)")
    p.add_argument("--price", help="Price per share (default: 250.50)")
    p.add_argument("--fee", help="Trading fee (default: 2)")
    p.add_argument("--action", choices=["BUY", "SELL"], default="BUY", help="Trade action (default: BUY)")
    p.add_argument("--datetime", dest="dt", help="MM/DD/YYYY,HH:MM (default: today,00:21)")
    # Runtime options
    p.add_argument("--headful", action="store_true", help="Run in headed mode (show browser)",default=False)
    return p.parse_args()

def login_to_stonkjournal(page, username, password):
    """
    Login to StonkJournal with retry mechanism
    Returns True on success, False on failure
    """
    try:
        print("Navigating to StonkJournal dashboard...")
        page.goto("https://app.stonkjournal.com/dashboard/", timeout=30000)

        # Wait for the page to load
        page.wait_for_load_state("domcontentloaded")
        time.sleep(2)
        
        print(f"Current URL: {page.url}")

        # Try to find email field with retry mechanism
        print("\nLooking for email field...")
        email_filled = False
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                print(f"[INFO] Attempt {attempt + 1}/{max_retries} to find email field...")
                email_field = page.locator("#email").or_(page.locator("input[name='email']")).or_(page.locator("input[type='email']")).first
                email_field.wait_for(state="visible", timeout=10000)
                print("[INFO] ✓ Found email field")
                
                email_field.fill(username)
                print(f"[INFO] ✓ Filled email: {username}")
                email_filled = True
                break
            except Exception as e:
                print(f"[WARN] Error with email field (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("[INFO] 🔄 Refreshing page and retrying...")
                    page.reload(wait_until="domcontentloaded", timeout=30000)
                    time.sleep(3)  # Wait for page to stabilize
                else:
                    print("[ERROR] Failed to find email field after all retries")
        
        if not email_filled:
            print("[ERROR] Could not fill email field")
            return False
            
        print("\nLooking for password field...")
        try:
            pwd_field = page.locator("#password").or_(page.locator("input[name='password']")).or_(page.locator("input[type='password']")).first
            pwd_field.wait_for(state="visible", timeout=10000)
            print("[INFO] ✓ Found password field")
            
            pwd_field.fill(password)
            print("[INFO] ✓ Filled password")
            
        except Exception as e:
            print("[ERROR] Could not fill password field")
            return False

        print("\nLooking for submit button...")         
        try:
            submit_btn = page.locator("button:has-text('Login')").first
            submit_btn.wait_for(state="visible", timeout=10000)
            print("[INFO] ✓ Found submit button")
            
            submit_btn.click()
            print("[INFO] ✓ Clicked submit button")
        except Exception as e:
            print("[ERROR] Could not click submit button")
            return False

        # Wait for dashboard
        print("\n[INFO] Waiting for dashboard to load...")
        try:
            page.wait_for_url("**/dashboard**", timeout=30000)
            print("[INFO] ✓ Successfully logged in!")
            print(f"[INFO] Current URL: {page.url}")
        except Exception as e:
            print(f"[WARN] Dashboard URL wait timeout: {e}")


        # Keep open briefly
        time.sleep(10)
        
        return True
       
    except Exception as e:
        print(f"[ERROR] Login failed: {str(e)}")
        page.screenshot(path="stonkjournal_error.png")
        print("[INFO] Error screenshot saved: stonkjournal_error.png")
        return False

def show_open_trades(page):
    """
    Filter trades to show only open trades
    Returns True on success, False on failure
    """
    try:
        print("\n[INFO] Filtering for open trades... - div with text 'OPEN'")
        page.locator('div:has-text("OPEN"):not(:has-text("%"))').first.click()
        print("[INFO] ✓ Successfully filtered for open trades!")
        return True
    except Exception as e:
        try: 
            print("[WARN] Error filtering for open trades, trying again...")
            # Click on the filter button
            page.locator('div:has-text("OPEN"):not(:has-text("%"))').first.click()
            print("[INFO] ✓ Successfully filtered for open trades!")
            return True
        except Exception as e:
            
            print(f"[ERROR] Failed to get open trades: {str(e)}")
            return False

def click_load_more_until_gone(page, max_clicks=20):
    try:
        print(f"\n[INFO] Clicking 'Load More' until button disappears (max {max_clicks} times)")
        clicks = 0
        
        while clicks < max_clicks:
            # Look for "Load More" button
            load_more_btn = page.locator("span:has-text('Load More')")
            
            if load_more_btn.count() == 0:
                print(f"[INFO] ✓ 'Load More' button no longer exists after {clicks} clicks")
                return clicks
            
            # Click "Load More"
            try:
                load_more_btn.click()
                clicks += 1
                print(f"[INFO] Clicked 'Load More' ({clicks}/{max_clicks})")
                time.sleep(3)  # Wait for content to load
            except Exception as e:
                print(f"[WARN] Error clicking 'Load More': {e}")
                print(f"[INFO] ✓ Completed after {clicks} clicks")
                return clicks
        
        print(f"[INFO] ✓ Reached maximum clicks limit ({max_clicks})")
        return clicks
            
    except Exception as e:
        print(f"[ERROR] Error in click_load_more_until_gone: {str(e)}")
        page.screenshot(path="load_more_error.png")
        print("[INFO] Error screenshot saved: load_more_error.png")
        return 0
        
def verify_page_loaded_and_check_trades(page):
    try:
        print("\n[INFO] Verifying page loaded and checking for trades...")
        
        # Wait for page to be in a stable state with retries
        max_retries = 10
        
        for attempt in range(max_retries):
            try:
                # Check if any trades exist
                print("[INFO] Checking for existing trades...")
                # Try multiple selectors to find trade rows
                trade_rows = page.locator("tr.trade-row, tbody tr, .trade-item, [class*='trade']").all()
                trades_count = len(trade_rows)
                print(f"[INFO] Found {trades_count} potential trade row(s)")
                if trades_count > 1:
                    print(f"[INFO] ✓ Found {trades_count} trade(s) on the page")
                    return True, trades_count
                else:
                    if attempt == 0:
                        fix_click = click_or_click_here(page, open_same_tab=True) 
                        page = fix_click or page
                        page.bring_to_front()
                    elif attempt < max_retries - 1:
                        print(f"[WARN] No trades found. Refreshing page and retrying... {trades_count} trades found")
                        page.screenshot(path=f"page_verification_attempt_{attempt + 1}.png")
                        print(f"[INFO] Screenshot saved: page_verification_attempt_{attempt + 1}.png")
                        time.sleep(1)
                        dup = page.context.new_page()
                        dup.goto(page.url, wait_until="domcontentloaded")
                        time.sleep(5)
                        dup.close()
                        page.reload(wait_until="domcontentloaded", timeout=30000)
                        time.sleep(5)
                    else:
                        print("[ERROR] No trades found after all retries")
                        print("[ERROR] Page appears stuck on loading or DB not connected")
                        page.screenshot(path="page_verification_error_final.png")
                        print("[INFO] Final screenshot saved: page_verification_error_final.png")
                        
                        try:
                            body_text = page.locator("body").text_content()
                            print(f"[DEBUG] Page body text (first 500 chars): {body_text[:500]}")
                            if "loading" in body_text.lower():
                                print("[ERROR] Page is still showing loading indicator!")
                        except:
                            pass
                        
                        return False, 0
            except Exception as e:
                print(f"[WARN] Error checking trades (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("[INFO] 🔄 Refreshing page and retrying...")
                    page.reload(wait_until="domcontentloaded", timeout=30000)
                    time.sleep(3)  # Wait for page to stabilize
                else:
                    print("[ERROR] Failed to verify page load after all retries")        
        return False, 0
        
    except Exception as e:
        print(f"[ERROR] Failed to verify page or check trades: {str(e)}")
        page.screenshot(path="page_verification_error.png")
        print("[INFO] Error screenshot saved: page_verification_error.png")
        return False, 0

def check_symbol_in_open_trades(page, symbol):
    """
    Check if symbol exists in the trades-table
    Returns the OPEN status cell element if found, None otherwise
    """
    try:
        print(f"\n[INFO] Checking if symbol '{symbol}' exists in open trades table...")
        
        # Wait for the trades table to be visible
        trades_table = page.locator(".trades-table, #trades-table, table.trades, table").first
        trades_table.wait_for(state="visible", timeout=10000)
        print("[INFO] ✓ Trades table found")
        

        # Find all rows in the table
        rows = page.locator(".trades-table tbody tr, #trades-table tbody tr, table tbody tr").all()
        print(f"[INFO] Found {len(rows)} rows in the trades table")
        
        # Search for the symbol in each row
        for i, row in enumerate(rows):
            try:
                # Get the text content of the row
                row_text = row.text_content()
                
                # Check if the symbol exists in this row
                if symbol.upper() in row_text.upper():
                    print(f"[INFO] ✓ Found symbol '{symbol}' in row {i + 1}")
                    print(f"[INFO] Row content: {row_text.strip()}")
                    
                    # Find the "OPEN" element in this row (status column)
                    # Try different approaches to find the OPEN status
                    open_element = row.locator("td:has-text('OPEN'), span:has-text('OPEN'), a:has-text('OPEN'), .status:has-text('OPEN')").first
                    
                    if open_element.count() > 0:
                        print(f"[INFO] ✓ Found OPEN status element for symbol: {symbol}")
                        return open_element
                    else:
                        print(f"[WARN] Could not find OPEN status element, trying alternative selectors...")
                        # Try to find any clickable element in the row
                        clickable = row.locator("a, button, td[onclick]").first
                        if clickable.count() > 0:
                            print(f"[INFO] Found clickable element in row")
                            return clickable
                        else:
                            print(f"[WARN] No clickable element found, returning row")
                            return row
                        
            except Exception as e:
                print(f"[WARN] Error processing row {i + 1}: {e}")
                continue
        
        print(f"[INFO] No existing open trade found for symbol: {symbol}")
        return None
            
    except Exception as e:
        print(f"[ERROR] Error checking for existing trade: {e}")
        page.screenshot(path="check_symbol_error.png")
        print("[INFO] Error screenshot saved: check_symbol_error.png")
        return None

def add_to_existing_trade(page, symbol_element, args):
    """
    Open an existing trade and add the action to it
    """
    try:
        print(f"\n[INFO] Adding {args.action} action to existing trade...")
        
        # Click on the trade row to open it
        symbol_element.click()
        print("[INFO] ✓ Opened existing trade")
        time.sleep(2)

        # Look for "Add Action" or similar button
        add_action_btn = page.locator("button:has-text('+')").first
        add_action_btn.wait_for(state="visible", timeout=10000)
        add_action_btn.click()
        print("[INFO] ✓ Clicked '+' button")
        time.sleep(2)
        
        # Fill in the action details (Quantity, Price, Fee, Action, DateTime)
        print("[INFO] Filling in action details...")
        
        # Quantity
        print(f"[INFO] Entering quantity: {args.quantity}")
        quantity_field = page.locator("//span[text()='Quantity']/following-sibling::input").last
        quantity_field.wait_for(state="visible", timeout=10000)
        quantity_field.fill(str(args.quantity))
        print(f"[INFO] ✓ Filled quantity: {args.quantity}")
        time.sleep(0.5)
        
        # Price
        print(f"[INFO] Entering price: {args.price}")
        price_field = page.locator("//span[text()='Price']/following-sibling::input").last
        price_field.wait_for(state="visible", timeout=10000)
        price_field.fill(str(args.price))
        print(f"[INFO] ✓ Filled price: {args.price}")
        time.sleep(0.5)
        
        # Fee
        print(f"[INFO] Entering fee: {args.fee}")
        fee_field = page.locator("//span[text()='Fee']/following-sibling::input").last
        fee_field.wait_for(state="visible", timeout=10000)
        fee_field.fill(str(args.fee))
        print(f"[INFO] ✓ Filled fee: {args.fee}")
        time.sleep(0.5)
        
        # Action (BUY/SELL)
        print(f"[INFO] Selecting action: {args.action}")
        action_field = page.locator("//span[text()='Action']/following-sibling::a").last
        action_field.wait_for(state="visible", timeout=10000)
        current_action_span = action_field.locator("span").first
        current_action = current_action_span.text_content().strip().upper()
        
        if current_action != args.action.upper():
            print(f"[INFO] Changing action from {current_action} to {args.action}")
            action_field.click()
            time.sleep(0.5)
            print(f"[INFO] ✓ Selected action: {args.action}")
        else:
            print(f"[INFO] ✓ Action already set to {args.action}")
        time.sleep(0.5)
        
        # DateTime
        if args.dt:
            print(f"[INFO] Entering datetime: {args.dt}")
            dt_field = page.locator("//span[contains(text(),'Date')]/following-sibling::input").last
            dt_field.wait_for(state="visible", timeout=10000)
            
            try:
                date_part, time_part = args.dt.split(',')
                month, day, year = date_part.split('/')
                formatted_dt = f"{year}-{month.zfill(2)}-{day.zfill(2)}T{time_part}"
                print(f"[INFO] Converted datetime format: {formatted_dt}")
                dt_field.fill(formatted_dt)
                print(f"[INFO] ✓ Filled datetime: {formatted_dt}")
            except Exception as e:
                print(f"[WARN] Failed to parse datetime format: {e}")
                dt_field.fill(args.dt)
                print(f"[INFO] ✓ Filled datetime (original format): {args.dt}")
            time.sleep(0.5)
        
        # Submit the action
        print("[INFO] Submitting the action...")
        submit_btn = page.locator("button:has-text('Save')").first
        submit_btn.wait_for(state="visible", timeout=10000)
        submit_btn.click()
        print("[INFO] ✓ Action submitted to existing trade")
        time.sleep(3)
        
        print("[INFO] ✓ Successfully added action to existing trade!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to add action to existing trade: {str(e)}")
        return False

def insert_trade(page, args):
    """
    Insert a new trade based on provided arguments
    """
    try:
        print("\n[INFO] Checking if trade already exists for this symbol...")
        
        # Check if symbol already exists in open trades
        existing_trade = check_symbol_in_open_trades(page, args.symbol)
        
        if existing_trade:
            print(f"[INFO] Symbol {args.symbol} found in open trades. Adding action to existing trade...")
            return add_to_existing_trade(page, existing_trade, args)
        
        # If no existing trade, create a new one
        print(f"[INFO] No existing open trade for {args.symbol}. Creating new trade...")
        
        # Click on the "Add Trade" button
        print("\n[INFO] Clicking on 'New Trade' button...")
        add_trade_btn = page.locator("button:has-text('New Trade')").first
        add_trade_btn.wait_for(state="visible", timeout=10000)
        add_trade_btn.click()
        print("[INFO] ✓ 'New Trade' button clicked")
        time.sleep(2)
        
        # Fill in trade details
        print("[INFO] Filling in trade details...")
        
        # Symbol - using the working XPath from Selenium
        print(f"[INFO] Entering symbol: {args.symbol}")
        symbol_field = page.locator("//span[text()='Symbol']/following-sibling::input").first
        symbol_field.wait_for(state="visible", timeout=10000)
        symbol_field.click()  # Focus on the field first
        symbol_field.fill(args.symbol)
        print(f"[INFO] ✓ Filled symbol: {args.symbol}")
        time.sleep(0.5)
        
        # Quantity
        print(f"[INFO] Entering quantity: {args.quantity}")
        quantity_field = page.locator("//span[text()='Quantity']/following-sibling::input").first
        quantity_field.wait_for(state="visible", timeout=10000)
        quantity_field.fill(str(args.quantity))
        print(f"[INFO] ✓ Filled quantity: {args.quantity}")
        time.sleep(0.5)
        
        # Price
        print(f"[INFO] Entering price: {args.price}")
        price_field = page.locator("//span[text()='Price']/following-sibling::input").first
        price_field.wait_for(state="visible", timeout=10000)
        price_field.fill(str(args.price))
        print(f"[INFO] ✓ Filled price: {args.price}")
        time.sleep(0.5)
        
        # Fee
        print(f"[INFO] Entering fee: {args.fee}")
        fee_field = page.locator("//span[text()='Fee']/following-sibling::input").first
        fee_field.wait_for(state="visible", timeout=10000)
        fee_field.fill(str(args.fee))
        print(f"[INFO] ✓ Filled fee: {args.fee}")
        time.sleep(0.5)
        
        # Action (BUY/SELL) - using XPath approach from Selenium
        print(f"[INFO] Selecting action: {args.action}")
        # Find the action dropdown trigger
        action_field = page.locator("//span[text()='Action']/following-sibling::a").first
        action_field.wait_for(state="visible", timeout=10000)
        
        # Get current action value
        current_action_span = action_field.locator("span").first
        current_action = current_action_span.text_content().strip().upper()
        print(f"[INFO] Current action: {current_action}")
        
        # Only click if we need to change the action
        if current_action != args.action.upper():
            print(f"[INFO] Changing action from {current_action} to {args.action}")
            action_field.click()
            time.sleep(0.5)
            print(f"[INFO] ✓ Selected action: {args.action}")
        else:
            print(f"[INFO] ✓ Action already set to {args.action}")
        
        time.sleep(0.5)
        
        # DateTime
        if args.dt:
            print(f"[INFO] Entering datetime: {args.dt}")
            dt_field = page.locator("//span[contains(text(),'Date')]/following-sibling::input").first
            dt_field.wait_for(state="visible", timeout=10000)
            
            # Convert from MM/DD/YYYY,HH:MM to YYYY-MM-DDTHH:MM format
            try:
                date_part, time_part = args.dt.split(',')
                month, day, year = date_part.split('/')
                formatted_dt = f"{year}-{month.zfill(2)}-{day.zfill(2)}T{time_part}"
                print(f"[INFO] Converted datetime format: {formatted_dt}")
                dt_field.fill(formatted_dt)
                print(f"[INFO] ✓ Filled datetime: {formatted_dt}")
            except Exception as e:
                print(f"[WARN] Failed to parse datetime format: {e}")
                # Try filling as-is in case format is already correct
                dt_field.fill(args.dt)
                print(f"[INFO] ✓ Filled datetime (original format): {args.dt}")
            
            time.sleep(0.5)
        
        # Submit the trade
        print("[INFO] Submitting the trade...")
        submit_trade_btn = page.locator("button:has-text('Save')").first
        submit_trade_btn.wait_for(state="visible", timeout=10000)
        submit_trade_btn.click()
        print("[INFO] ✓ Trade submitted")
        
        time.sleep(3)
        

        print("[INFO] ✓ Trade insertion completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to insert trade: {str(e)}")
        page.screenshot(path="trade_insert_error.png")
        print("[INFO] Error screenshot saved: trade_insert_error.png")
        return False


def parse_csv_file(csv_file_path):
    """
    Parse the CSV file and extract EXECUTION records into trade objects
    
    Filters only rows where LevelOfDetail == 'EXECUTION'
    Extracts: Symbol, DateTime, Quantity, IBCommission, Buy/Sell, TradePrice
    Returns: List of trade dictionaries ready for insert_trade
    """
    
    print(f"\n[INFO] Reading CSV file: {csv_file_path}")
    
    executions = []
    
    try:
        # Try different encodings to handle various CSV formats
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                with open(csv_file_path, 'r', encoding=encoding) as file:
                    csv_reader = csv.DictReader(file)
                    
                    for row in csv_reader:
                        # Filter only EXECUTION level of detail AND USD currency
                        if row.get('LevelOfDetail') == 'EXECUTION' and row.get('CurrencyPrimary') == 'USD' and row.get('SubCategory') == 'COMMON':
                            execution = {
                                'Symbol': row.get('Symbol', ''),
                                'DateTime': row.get('DateTime', ''),
                                'Quantity': row.get('Quantity', ''),
                                'IBCommission': row.get('IBCommission', ''),
                                'BuySell': row.get('Buy/Sell', ''),
                                'TradePrice': row.get('TradePrice', '')
                            }
                            executions.append(execution)
                
                print(f"[INFO] Successfully read file with encoding: {encoding}")
                break
                
            except UnicodeDecodeError:
                continue
    
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_file_path}")
        return []
    except Exception as e:
        print(f"[ERROR] Error reading CSV: {e}")
        return []
    
    if not executions:
        print(f"[WARN] No EXECUTION records found in CSV")
        return []
    
    print(f"[INFO] Found {len(executions)} EXECUTION records")
    
    # Sort executions by DateTime
    print(f"[INFO] Sorting trades by DateTime...")
    try:
        executions.sort(key=lambda x: datetime.strptime(
            x['DateTime'].replace(' EDT', '').replace(' EST', ''), 
            "%m/%d/%Y,%H:%M:%S"
        ))
        print(f"[INFO] ✓ Trades sorted chronologically")
    except Exception as e:
        print(f"[WARN] Could not sort by DateTime: {e}. Processing in original order.")
    
    # Timezone conversion setup
    edt_tz = pytz.timezone('America/New_York')  # EDT/EST
    israel_tz = pytz.timezone('Asia/Jerusalem')
    
    # Process and convert each execution to trade format
    trades = []
    for exec_data in executions:
        # Convert commission to positive value
        commission = exec_data['IBCommission']
        if commission:
            try:
                commission = abs(float(commission))
            except:
                commission = 0
        else:
            commission = 0

        quantity = exec_data['Quantity']
        if quantity:
            try:
                quantity = abs(float(quantity))
            except:
                quantity = 0
        else:
            quantity = 0
        # Convert datetime from EDT to Israel time
        datetime_str = exec_data['DateTime']
        israel_datetime = ""
        if datetime_str:
            try:
                # Parse the datetime string (format: "03/10/2025,09:47:03 EDT")
                dt_parts = datetime_str.replace(' EDT', '').replace(' EST', '')
                dt_obj = datetime.strptime(dt_parts, "%m/%d/%Y,%H:%M:%S")
                
                # Assume the input is in EDT/EST
                dt_edt = edt_tz.localize(dt_obj)
                
                # Convert to Israel time
                dt_israel = dt_edt.astimezone(israel_tz)
                
                # Format as MM/DD/YYYY,HH:MM for main.py
                israel_datetime = dt_israel.strftime("%m/%d/%Y,%H:%M")
            except Exception as e:
                print(f"[WARN] Failed to convert datetime '{datetime_str}': {e}")
                # Try to use original without timezone suffix
                israel_datetime = datetime_str.replace(' EDT', '').replace(' EST', '').rsplit(':', 1)[0]
        
        # Create trade object
        trade = {
            'symbol': exec_data['Symbol'],
            'quantity': str(quantity),
            'price': exec_data['TradePrice'],
            'fee': str(commission),
            'action': exec_data['BuySell'],
            'dt': israel_datetime
        }
        trades.append(trade)
    
    print(f"[INFO] Processed {len(trades)} trades from CSV\n")
    return trades

def create_trade_object(symbol, quantity, price, fee, action, dt):
    """Create a trade args object for insert_trade function"""
    class TradeArgs:
        pass
    
    trade = TradeArgs()
    trade.symbol = symbol
    trade.quantity = quantity
    trade.price = price
    trade.fee = fee
    trade.action = action
    trade.dt = dt
    return trade


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    print("=" * 60)
    print("StonkJournal Trade Automation")
    print("=" * 60)
    print(f"Username: {args.username}")
    print(f"Mode: {'Headed (visible browser)' if args.headful else 'Headless'}")
    
    # Determine processing mode: CSV or Single Trade
    trades_to_process = []
    
    if args.csv_file:
        # CSV Batch Mode
        print(f"Processing Mode: CSV Batch")
        print(f"CSV File: {args.csv_file}")
        print("=" * 60)
        
        # Parse CSV and get list of trades
        trades_list = parse_csv_file(args.csv_file)
        
        if not trades_list:
            print("\n[INFO] No trades matching filters found in CSV file.")
            print("[INFO] Filters applied: LevelOfDetail='EXECUTION' AND CurrencyPrimary='USD'")
            print("[INFO] Nothing to process. Exiting successfully.")
            exit(0)
        
        # Convert each trade dict to trade object
        for trade_dict in trades_list:
            trade_obj = create_trade_object(
                symbol=trade_dict['symbol'],
                quantity=trade_dict['quantity'],
                price=trade_dict['price'],
                fee=trade_dict['fee'],
                action=trade_dict['action'],
                dt=trade_dict['dt']
            )
            trades_to_process.append(trade_obj)
            
    else:
        # Single Trade Mode
        print(f"Processing Mode: Single Trade")
        
        # Validate required arguments for single trade
        if not args.symbol:
            print("\n✗ Error: --symbol is required for single trade mode")
            exit(1)
        
        print(f"Symbol: {args.symbol}")
        print(f"Action: {args.action} {args.quantity} shares @ ${args.price}")
        print(f"Fee: ${args.fee}")
        if args.dt:
            print(f"DateTime: {args.dt}")
        print("=" * 60)
        
        # Create single trade object
        single_trade = create_trade_object(
            symbol=args.symbol,
            quantity=args.quantity,
            price=args.price,
            fee=args.fee,
            action=args.action,
            dt=args.dt
        )
        trades_to_process.append(single_trade)
    
    print(f"Total trades to process: {len(trades_to_process)}")
    print("=" * 60)
    print()

    # Create browser session (one session for all trades)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Step 1: Login (once for all trades)
            login_success = login_to_stonkjournal(page, args.username, args.password)
            
            if not login_success:
                print("\n✗ Login failed!")
                exit(1)
            
            # Step 2: Verify page loaded and check for existing trades
            page_loaded, trades_count = verify_page_loaded_and_check_trades(page)

            if not page_loaded:
                print("\n✗ Failed to verify page loaded!")
                exit(1)
            
            print(f"\n[INFO] Page verified with {trades_count} trade(s) displayed")
            
            # Step 3: Process all trades in sequence
            successful_trades = 0
            failed_trades = 0
            total_trades = len(trades_to_process)
            
            for idx, trade in enumerate(trades_to_process, 1):
                print(f"\n{'=' * 60}")
                print(f"Processing Trade {idx}/{total_trades}")
                print(f"{'=' * 60}")
                print(f"Symbol: {trade.symbol}")
                print(f"Action: {trade.action} {trade.quantity} shares @ ${trade.price}")
                print(f"Fee: ${trade.fee}")
                if trade.dt:
                    print(f"DateTime: {trade.dt}")
                print()
                
                # Reload page and filter open trades before each trade insertion
                print("\n[INFO] Reloading page and filtering open trades before trade insertion...")
                page.reload(wait_until="domcontentloaded", timeout=30000)
                time.sleep(2)
                
                # Reapply open trades filter
                filter_success = show_open_trades(page)
                if filter_success:
                    print("[INFO] ✓ Page reloaded and filtered")
                else:
                    print("[WARN] Failed to reapply filter, but continuing...")
                
                time.sleep(2)
                ## Click Load more until button disappears or 20 times
                click_load_more_until_gone(page, max_clicks=20)

                # Insert trade using standard flow
                insert_success = insert_trade(page, trade)
                
                if insert_success:
                    successful_trades += 1
                    print(f"✓ Trade {idx} completed successfully!")
                else:
                    failed_trades += 1
                    print(f"✗ Trade {idx} failed!")
                time.sleep(5)
            
            # Summary
            print(f"\n{'=' * 60}")
            print("EXECUTION SUMMARY")
            print(f"{'=' * 60}")
            print(f"Total trades processed: {total_trades}")
            print(f"Successful: {successful_trades}")
            print(f"Failed: {failed_trades}")
            print(f"{'=' * 60}")
            
            if failed_trades == 0:
                print("\n✓ All trades completed successfully!")
            else:
                print(f"\n⚠ Completed with {failed_trades} failure(s)")
            
        except Exception as e:
            print(f"\n✗ Script failed with error: {str(e)}")
            page.screenshot(path="script_error.png")
            exit(1)
            
        finally:
            browser.close()

