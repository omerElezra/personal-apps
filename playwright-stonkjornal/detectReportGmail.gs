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
            console.error(`Webhook error ${code}: ${resp.getContentText()}`);
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
