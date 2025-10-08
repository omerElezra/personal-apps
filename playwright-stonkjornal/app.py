import os
import tempfile
import subprocess
import requests
import shlex
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RAW_URL = os.getenv(
    "RAW_URL",
    "https://raw.githubusercontent.com/omerElezra/personal-apps/refs/heads/main/playwright-stonkjornal/main.py"
)
BEARER = os.getenv("BEARER_TOKEN", "")  # Authentication for Webhook service

app = FastAPI(title="StonkJournal Trade Automation Service")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "StonkJournal Trade Automation",
        "version": "1.0",
        "endpoints": {
            "health": "/healthz",
            "ingest": "/ingest (POST)"
        }
    }

@app.get("/healthz")
def healthz():
    """Health check endpoint"""
    return {"status": "healthy", "service": "stonkjournal-automation"}

@app.post("/ingest")
async def ingest(request: Request):
    """
    Process CSV trade reports and automate StonkJournal entries.
    
    Required Headers:
    - Authorization: Bearer token for authentication
    - X-Username: StonkJournal username
    - X-Password: StonkJournal password
    - X-Filename: (Optional) CSV filename
    
    Body: CSV file content
    """
    # Authentication
    auth = request.headers.get("authorization", "")
    if not BEARER or auth != f"Bearer {BEARER}":
        logger.warning("Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Extract credentials from headers
    username = request.headers.get("x-username")
    password = request.headers.get("x-password")
    filename = request.headers.get("x-filename", "report.csv")
    
    if not username or not password:
        logger.error("Missing username or password in headers")
        raise HTTPException(status_code=400, detail="Missing username/password headers")

    # Read CSV body
    csv_bytes = await request.body()
    if not csv_bytes:
        logger.error("Empty CSV body received")
        raise HTTPException(status_code=400, detail="Empty CSV body")

    logger.info(f"Processing CSV file: {filename} for user: {username}")

    # Download latest main.py script
    try:
        r = requests.get(RAW_URL, timeout=30)
        r.raise_for_status()
        logger.info("Successfully downloaded main.py script")
    except requests.RequestException as e:
        logger.error(f"Failed to download main.py: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download script: {str(e)}")

    # Create temporary directory and execute script
    with tempfile.TemporaryDirectory() as td:
        script_path = os.path.join(td, "main.py")
        csv_path = os.path.join(td, filename)
        
        # Write script and CSV files
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(r.text)
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)

        # Build and execute command
        cmd = (
            f"python {shlex.quote(script_path)} "
            f"--username {shlex.quote(username)} "
            f"--password {shlex.quote(password)} "
            f"--csv-file {shlex.quote(csv_path)}"
        )
        
        logger.info(f"Executing automation script for {filename}")
        
        try:
            run = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            output = (run.stdout + "\n" + run.stderr).strip()
            
            if run.returncode != 0:
                logger.error(f"Script execution failed with code {run.returncode}: {output}")
                raise HTTPException(status_code=500, detail=output)
            
            logger.info(f"Successfully processed {filename}")
            return PlainTextResponse((run.stdout or "OK").strip(), status_code=200)
            
        except subprocess.TimeoutExpired:
            logger.error(f"Script execution timed out for {filename}")
            raise HTTPException(status_code=500, detail="Script execution timed out after 30 minutes")
        except Exception as e:
            logger.error(f"Unexpected error during script execution: {e}")
            raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

