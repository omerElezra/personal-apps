// ====== CONFIG ======
const FROM_EMAIL = 'Info@inter-il.com';                 // Exact sender email
const SUBJECT_PHRASE = 'Activity Flex for ';           // Subject text to match
const LOOKBACK_HOURS = 24;                              // Search window in hours
const LABEL_AFTER_PROCESS = 'Processed/CSV-Reports';    // Label to apply after processing
const DEDUP_KEY = 'processedMessageIds';                // Storage key for deduplication

// ====== ENTRY POINT ======
function processNewReports() {
  // Prevent overlapping runs (concurrent triggers)
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(5000)) {
    console.warn('Another run is in progress, skipping.');
    return;
  }

  try {
    const username = PropertiesService.getScriptProperties().getProperty('USERNAME');
    const password = PropertiesService.getScriptProperties().getProperty('PASSWORD');
    
    const webhook_url = PropertiesService.getScriptProperties().getProperty('WEBHOOK_URL');
    const token = PropertiesService.getScriptProperties().getProperty('BEARER_TOKEN') || '';

    if (!username || !password) throw new Error('USERNAME/PASSWORD missing in Script Properties');
    if (!webhook_url) throw new Error('WEBHOOK_URL missing in Script Properties');
    
    const query = buildQuery();
    const dedup = new Deduper(DEDUP_KEY);
    const label = getOrCreateLabel(LABEL_AFTER_PROCESS);

    const threads = GmailApp.search(query, 0, 50); // Can be paginated
    if (!threads.length) return;

    for (const thread of threads) {
      const messages = thread.getMessages();
      for (const msg of messages) {
        const id = msg.getId();
        const mid = msg.getHeader('Message-ID') || '';

        // Deduplication - don't process again
        if (dedup.seen(id) || dedup.seen(mid)) continue;

        // Additional validation beyond query
        if (!msg.getFrom().includes(FROM_EMAIL)) continue;
        if (!msg.getSubject().includes(SUBJECT_PHRASE)) continue;

        // Extract only CSV files
        const atts = msg.getAttachments({includeInlineImages: false});
        const csvAtts = atts.filter(a => a.getName().toLowerCase().endsWith('.csv'));
        if (!csvAtts.length) continue;

        for (const att of csvAtts) {
          const filename = att.getName();
          // Default: UTF-8. For other encodings, use att.copyBlob().getDataAsString('windows-1255')
          const csvText = att.getDataAsString();

          // Send file to Webhook
          const headers = {
            'Content-Type': 'text/csv',
            'X-Username': username,
            'X-Password': password,
            'X-Filename': filename
          };
          if (token) headers['Authorization'] = 'Bearer ' + token;

          const resp = UrlFetchApp.fetch(webhook_url, {
            method: 'post',
            headers,
            payload: csvText,
            muteHttpExceptions: true,
          });

          const code = resp.getResponseCode();
          if (code < 200 || code >= 300) {
            const errorMsg = resp.getContentText();
            console.error(`Webhook error ${code}: ${errorMsg}`);
            
            // Try to parse error response for screenshot
            let screenshotData = null;
            let screenshotName = null;
            let errorText = errorMsg;
            try {
              const errorJson = JSON.parse(errorMsg);
              console.log('Parsed error JSON:', JSON.stringify(errorJson).substring(0, 200));
              
              // FastAPI wraps the detail parameter in a "detail" key
              if (errorJson.detail) {
                if (typeof errorJson.detail === 'object') {
                  // detail is an object containing error, screenshot, screenshot_name
                  screenshotData = errorJson.detail.screenshot;
                  screenshotName = errorJson.detail.screenshot_name;
                  errorText = errorJson.detail.error || JSON.stringify(errorJson.detail);
                  console.log(`Screenshot found: ${screenshotName ? 'Yes - ' + screenshotName : 'No'}`);
                } else {
                  // detail is a simple string
                  errorText = errorJson.detail;
                }
              }
            } catch (e) {
              console.log('Could not parse error as JSON:', e.message);
              // Not JSON or no screenshot, that's okay - use raw errorMsg
            }
            
            // Send email alert about the error with screenshot if available
            sendErrorEmail(username, filename, code, errorText, msg.getSubject(), msg.getDate(), screenshotData, screenshotName);
            
            // Don't label - so next run will retry
            continue;
          } else {
            // Mark as processed + apply label
            dedup.mark(id);
            if (mid) dedup.mark(mid);
            thread.addLabel(label);
          }
        }


      }
    }
    dedup.flush();
  } finally {
    lock.releaseLock();
  }
}

// ====== HELPERS ======

function buildQuery() {
  const days = Math.max(1, Math.ceil(LOOKBACK_HOURS / 24));
  return `from:${FROM_EMAIL} subject:"${SUBJECT_PHRASE}" has:attachment filename:csv newer_than:${days}d`;
}

function getOrCreateLabel(name) {
  let label = GmailApp.getUserLabelByName(name);
  if (!label) label = GmailApp.createLabel(name);
  return label;
}

/**
 * Send error notification email when webhook fails
 * @param {string} recipientEmail - Email address to send alert to (StonkJournal username)
 * @param {string} filename - CSV filename that failed
 * @param {number} errorCode - HTTP error code
 * @param {string} errorMessage - Error message from webhook
 * @param {string} emailSubject - Original email subject
 * @param {Date} emailDate - Original email date
 * @param {string} screenshotData - Base64 encoded PNG screenshot (optional)
 * @param {string} screenshotName - Screenshot filename (optional)
 */
function sendErrorEmail(recipientEmail, filename, errorCode, errorMessage, emailSubject, emailDate, screenshotData, screenshotName) {
  try {
    const subject = `⚠️ StonkJournal Automation Error - ${filename}`;
    
    const body = `
StonkJournal Trade Automation Error Alert
==========================================

An error occurred while processing your Interactive Brokers report.

ERROR DETAILS:
--------------
HTTP Status Code: ${errorCode}
Error Message: ${errorMessage}

REPORT DETAILS:
---------------
CSV Filename: ${filename}
Original Email Subject: ${emailSubject}
Email Received: ${Utilities.formatDate(emailDate, Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm:ss z')}

WEBHOOK INFORMATION:
--------------------
Webhook URL: ${PropertiesService.getScriptProperties().getProperty('WEBHOOK_URL')}
Processing Time: ${new Date().toLocaleString()}

TROUBLESHOOTING:
----------------
1. Check if the webhook service is running
2. Verify BEARER_TOKEN is configured correctly
3. Check Cloud Run logs: gcloud run services logs tail gmail-csv-webhook --project=stocks-report-474512 --region=me-west1
4. Verify USERNAME and PASSWORD are correct
5. Check if the CSV format is compatible
${screenshotData ? '\n6. Review the attached error screenshot for visual debugging' : ''}

NEXT STEPS:
-----------
- The system will automatically retry processing this email in the next run
- If the error persists, check the webhook service logs
- You can manually trigger processing by running the debugOnce() function

---
This is an automated alert from StonkJournal Trade Automation
Configured in Google Apps Script: ${ScriptApp.getScriptId()}
`;

    // Prepare email options
    const emailOptions = {};
    
    // Attach screenshot if available
    if (screenshotData && screenshotName) {
      try {
        console.log(`Attempting to attach screenshot: ${screenshotName}, data length: ${screenshotData.length}`);
        const imageBlob = Utilities.newBlob(
          Utilities.base64Decode(screenshotData),
          'image/png',
          screenshotName
        );
        emailOptions.attachments = [imageBlob];
        console.log(`✓ Successfully attached screenshot: ${screenshotName} (${imageBlob.getBytes().length} bytes)`);
      } catch (e) {
        console.error(`Failed to decode screenshot: ${e.message}`);
        // Continue without attachment
      }
    } else {
      console.log(`No screenshot to attach (data: ${screenshotData ? 'present' : 'missing'}, name: ${screenshotName || 'missing'})`);
    }

    // Send email to the user
    GmailApp.sendEmail(recipientEmail, subject, body, emailOptions);
    console.log(`Error notification sent to ${recipientEmail}${screenshotData ? ' with screenshot' : ''}`);
    
  } catch (e) {
    console.error(`Failed to send error email: ${e.message}`);
    // Don't throw - we don't want email failure to break the main flow
  }
}

// Deduplication based on PropertiesService (persistent across runs)
class Deduper {
  constructor(key) {
    this.key = key;
    this.props = PropertiesService.getUserProperties();
    this.set = new Set(JSON.parse(this.props.getProperty(this.key) || '[]'));
    this.max = 2000; // Built-in size limit
  }
  seen(v) { return v && this.set.has(v); }
  mark(v) {
    if (!v) return;
    this.set.add(v);
    if (this.set.size > this.max) {
      const arr = Array.from(this.set);
      this.set = new Set(arr.slice(arr.length - this.max));
    }
  }
  flush() { this.props.setProperty(this.key, JSON.stringify(Array.from(this.set))); }
}

// Manual run for testing
function debugOnce() { processNewReports(); }
