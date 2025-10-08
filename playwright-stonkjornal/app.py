import os, tempfile, subprocess, requests, shlex
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

RAW_URL = "https://raw.githubusercontent.com/omerElezra/personal-apps/refs/heads/selenium-stonk/playwright-stonkjornal/main.py"
BEARER = os.getenv("BEARER_TOKEN", "")  # Authetication for Webhook service

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ingest")
async def ingest(request: Request):
    auth = request.headers.get("authorization", "")
    if not BEARER or auth != f"Bearer {BEARER}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    username = request.headers.get("x-username")
    password = request.headers.get("x-password")
    filename = request.headers.get("x-filename", "report.csv")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Missing username/password headers")

    csv_bytes = await request.body()
    if not csv_bytes:
        raise HTTPException(status_code=400, detail="Empty CSV body")

    r = requests.get(RAW_URL, timeout=30)
    r.raise_for_status()

    with tempfile.TemporaryDirectory() as td:
        script_path = os.path.join(td, "main.py")
        csv_path = os.path.join(td, filename)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(r.text)
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)

        cmd = f"python {shlex.quote(script_path)} --username {shlex.quote(username)} --password {shlex.quote(password)} --csv-file {shlex.quote(csv_path)}"
        run = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        if run.returncode != 0:
            raise HTTPException(status_code=500, detail=(run.stdout + "\n" + run.stderr).strip())
        return PlainTextResponse((run.stdout or "OK").strip(), status_code=200)

