# backend/app/main.py
import os, re, json, uuid, io, csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

# === ChromaDB ===
import chromadb
from chromadb.config import Settings

# === PDF (reportlab) ===
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from google import genai

# ---------------- Config ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
API_KEY = os.getenv("WEATHER_API_KEY")
if not API_KEY:
    raise RuntimeError("WEATHER_API_KEY not set in .env at project root.")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "backend" / "data" / "chroma"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WEATHERAPI_CURRENT_URL = "http://api.weatherapi.com/v1/current.json"
WEATHERAPI_FORECAST_URL = "http://api.weatherapi.com/v1/forecast.json"

# ---------------- App ----------------
app = FastAPI(title="AI Weather App (Radio Modes + CRUD + Export)")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://0.0.0.0:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    f = FRONTEND_DIR / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(f.read_text(encoding="utf-8"))

@app.get("/ping")
async def ping():
    return {"message": "ok"}

# ---------------- Schemas ----------------
class Coords(BaseModel):
    lat: float
    lon: float

class WeatherQuery(BaseModel):
    # radio-based mode from the UI
    input_mode: str = Field(..., pattern="^(name|coords|ip)$")
    place: Optional[str] = None
    coords: Optional[Coords] = None

    @model_validator(mode="after")
    def check_required(self):
        if self.input_mode == "name" and not (self.place and self.place.strip()):
            raise ValueError("place is required for input_mode=name")
        if self.input_mode == "coords" and not self.coords:
            raise ValueError("coords is required for input_mode=coords")
        return self

# ---------------- Helpers ----------------
_latlon_re = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")

def _normalize_coords(lat: float, lon: float) -> Tuple[float, float]:
    def ok(a, b): return -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0
    if ok(lat, lon): return (lat, lon)
    if ok(lon, lat): return (lon, lat)  # auto-fix swapped inputs
    raise HTTPException(
        status_code=422,
        detail="Invalid coordinates. Use 'lat,lon' in decimal degrees, e.g. 48.8567,2.3508"
    )

def _build_weatherapi_q(payload: WeatherQuery, client_ip: Optional[str]) -> str:
    if payload.input_mode == "coords" and payload.coords:
        lat, lon = _normalize_coords(payload.coords.lat, payload.coords.lon)
        return f"{lat:.6f},{lon:.6f}"
    if payload.input_mode == "name" and payload.place:
        # Allow "lat,lon" typed into place
        m = _latlon_re.match(payload.place.strip())
        if m:
            lat, lon = _normalize_coords(float(m.group(1)), float(m.group(2)))
            return f"{lat:.6f},{lon:.6f}"
        return payload.place.strip()
    # ip autodetect (server-side)
    return "auto:ip"

def _call_weatherapi(url: str, q: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = {"key": API_KEY, "q": q, "lang": "en"}
    if extra: params.update(extra)
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream weather API error: {e}")

def _condition_code_from_text(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["thunder", "storm", "lightning"]): return "thunder"
    if any(k in t for k in ["snow", "sleet", "blizzard", "flurr"]): return "snow"
    if any(k in t for k in ["drizzle", "shower"]): return "drizzle"
    if "rain" in t: return "rain"
    if any(k in t for k in ["fog", "mist", "haze"]): return "fog"
    if any(k in t for k in ["wind", "breeze", "gale"]): return "wind"
    if "cloud" in t: return "cloudy" if "overcast" in t else "partly-cloudy"
    if any(k in t for k in ["clear", "sunny", "bright"]): return "clear"
    return "partly-cloudy"

def _normalize_current(data: Dict[str, Any]) -> Dict[str, Any]:
    loc = data.get("location", {}) or {}
    cur = data.get("current", {}) or {}
    condition_text = (cur.get("condition") or {}).get("text", "")
    return {
        "location": ", ".join([v for v in [loc.get("name"), loc.get("region"), loc.get("country")] if v]),
        "condition_code": _condition_code_from_text(condition_text),
        "condition_text": condition_text,
        "temp_c": cur.get("temp_c"),
        "feels_like_c": cur.get("feelslike_c"),
        "humidity": cur.get("humidity"),
        "wind_kph": cur.get("wind_kph"),
        "observed_at": loc.get("localtime") or cur.get("last_updated"),
    }

def _normalize_forecast(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    days = (data.get("forecast", {}) or {}).get("forecastday", []) or []
    out: List[Dict[str, Any]] = []
    for d in days[:5]:
        day = d.get("day", {}) or {}
        ct = (day.get("condition") or {}).get("text", "")
        out.append({
            "date": d.get("date"),
            "condition_code": _condition_code_from_text(ct),
            "condition_text": ct,
            "max_c": day.get("maxtemp_c"),
            "min_c": day.get("mintemp_c"),
            "precip_mm": day.get("totalprecip_mm"),
            "wind_kph": day.get("maxwind_kph"),
        })
    return out

# ---- record utils (flat metadata + JSON document) ----
def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _record_summary(current: Dict[str, Any]) -> str:
    try:
        return f"{round(current.get('temp_c'))}°C, {current.get('condition_text')}"
    except Exception:
        return ""

def _location_display(current: Dict[str, Any]) -> str:
    return current.get("location") or ""

def _sanitize_metadata(d: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only str/int/float/bool & drop Nones. No nested dicts/lists."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
    return out

def _row_from_record(id_: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": id_,
        "input_mode": record.get("input_mode"),
        "location_display": record.get("location_display"),
        "created_at": record.get("created_at"),
        "weather": record.get("weather"),  # contains .summary used by table
        "place": record.get("place"),
        "coords": record.get("coords"),
    }

def _ai_clothing_suggestion(current: Dict[str, Any]) -> Optional[str]:
    """
    Build a short clothing/gear recommendation using Gemini.
    Returns None if Gemini isn't configured.
    """
    llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    )
    print("Using Gemini API Key:")
    # Extract facts for prompt
    temp = current.get("temp_c")
    feels = current.get("feels_like_c")
    cond = current.get("condition_text")
    wind = current.get("wind_kph")
    humid = current.get("humidity")
    loc = current.get("location")
    aqi = current.get("aqi") or {}
    epa = aqi.get("us_epa_index")
    pm25 = aqi.get("pm2_5")

    prompt = f"""
    You are a concise weather stylist. Based on the data below, give 2–4 bullet suggestions
    about what to wear and carry. Keep it short, practical, and localized to the place.

    Location: {loc}
    Condition: {cond}
    Temperature: {temp}°C (feels like {feels}°C)
    Wind: {wind} kph
    Humidity: {humid}%
    Air Quality (US EPA Index): {epa}
    PM2.5: {pm25}

    Rules:
    - Use emojis sparingly: coat 🧥, umbrella ☂️, sunscreen 🧴, mask 😷, hat 🧢, water 💧, layers 🧣, shoes 👟
    - Mention AQI precautions if EPA index ≥ 3 or PM2.5 ≥ 35.
    - If rain likely (precip ≥ 1), suggest rain gear.
    - If temp < 10°C suggest warm layers; if > 28°C suggest light breathable clothing.
    - Output plain text with bullet lines ("• ...").
    """
    # try:
    r = llm_gemini.invoke(prompt)
    text = (r.content or "").strip()
    print(r,text)
    # Keep it short just in case
    return "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()][:6])
    # except Exception:
        # return None
    
# ---------------- ChromaDB Setup ----------------
def get_collection():
    client = chromadb.PersistentClient(path=str(DATA_DIR), settings=Settings(allow_reset=False))
    try:
        return client.get_collection("weather_records")
    except Exception:
        return client.create_collection("weather_records")

def _load_records(rec_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return list of record dicts from documents JSON (all or one)."""
    col = get_collection()
    if rec_id:
        res = col.get(ids=[rec_id])
        if not res.get("ids"):
            raise HTTPException(status_code=404, detail="Record not found")
        docs = res.get("documents") or []
    else:
        res = col.get()
        docs = res.get("documents") or []
    out = []
    for d in docs:
        try:
            out.append(json.loads(d))
        except Exception:
            continue
    return out

# ---------------- Weather Endpoints ----------------
@app.post("/api/weather/current")
def api_current(payload: WeatherQuery, request: Request):
    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    data = _call_weatherapi(WEATHERAPI_CURRENT_URL, q=q)
    if "location" not in data or "current" not in data:
        raise HTTPException(status_code=500, detail="Unexpected response from WeatherAPI (current).")
    current = _normalize_current(data)
    advice = _ai_clothing_suggestion(current)
    out=current.copy()
    out['ai_advice']=advice
    print(out)
    return out

@app.post("/api/weather/forecast")
def api_forecast(payload: WeatherQuery, request: Request):
    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    data = _call_weatherapi(WEATHERAPI_FORECAST_URL, q=q, extra={"days": 5, "aqi": "no", "alerts": "no"})
    if "forecast" not in data:
        raise HTTPException(status_code=500, detail="Unexpected response from WeatherAPI (forecast).")
    return _normalize_forecast(data)

# ---------------- CRUD Endpoints ----------------
@app.get("/api/records")
def list_records():
    col = get_collection()
    res = col.get()
    ids = res.get("ids", [])
    docs = res.get("documents", []) or []
    out = []
    for i, doc in zip(ids, docs):
        try:
            record = json.loads(doc)
            out.append(_row_from_record(i, record))
        except Exception:
            out.append({"id": i, "input_mode": None, "location_display": "", "created_at": "", "weather": None})
    return out

@app.post("/api/records")
def create_record(payload: WeatherQuery, request: Request):
    # Fetch weather snapshot
    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    cur_raw = _call_weatherapi(WEATHERAPI_CURRENT_URL, q=q)
    f_raw = _call_weatherapi(WEATHERAPI_FORECAST_URL, q=q, extra={"days": 5, "aqi": "no", "alerts": "no"})
    current = _normalize_current(cur_raw)
    forecast = _normalize_forecast(f_raw)

    record = {
        "id": None,  # set after UUID
        "created_at": _now_iso(),
        "input_mode": payload.input_mode,
        "place": payload.place,
        "coords": payload.coords.model_dump() if payload.coords else None,
        "location_display": _location_display(current),
        "weather": {
            "current": current,
            "forecast": forecast,
            "summary": _record_summary(current),
        },
    }

    meta = _sanitize_metadata({
        "created_at": record["created_at"],
        "input_mode": record["input_mode"],
        "place": record["place"] or "",
        "location_display": record["location_display"],
        "summary": record["weather"]["summary"],
        "lat": (record["coords"] or {}).get("lat") if record["coords"] else None,
        "lon": (record["coords"] or {}).get("lon") if record["coords"] else None,
    })

    rec_id = str(uuid.uuid4())
    record["id"] = rec_id

    col = get_collection()
    col.add(
        ids=[rec_id],
        metadatas=[meta],
        documents=[json.dumps(record, ensure_ascii=False)],
    )

    return _row_from_record(rec_id, record)

@app.get("/api/records/{rec_id}")
def read_record(rec_id: str):
    records = _load_records(rec_id)
    record = records[0]
    return _row_from_record(rec_id, record)

@app.put("/api/records/{rec_id}")
def update_record(rec_id: str, payload: WeatherQuery, request: Request):
    col = get_collection()
    res = col.get(ids=[rec_id])
    old_doc = (res.get("documents") or ["{}"])[0] if res.get("ids") else "{}"
    try:
        old_record = json.loads(old_doc)
    except Exception:
        old_record = {}

    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    cur_raw = _call_weatherapi(WEATHERAPI_CURRENT_URL, q=q)
    f_raw = _call_weatherapi(WEATHERAPI_FORECAST_URL, q=q, extra={"days": 5, "aqi": "no", "alerts": "no"})
    current = _normalize_current(cur_raw)
    forecast = _normalize_forecast(f_raw)

    record = {
        "id": rec_id,
        "created_at": old_record.get("created_at") or _now_iso(),
        "input_mode": payload.input_mode,
        "place": payload.place,
        "coords": payload.coords.model_dump() if payload.coords else None,
        "location_display": _location_display(current),
        "weather": {
            "current": current,
            "forecast": forecast,
            "summary": _record_summary(current),
        },
    }

    meta = _sanitize_metadata({
        "created_at": record["created_at"],
        "input_mode": record["input_mode"],
        "place": record["place"] or "",
        "location_display": record["location_display"],
        "summary": record["weather"]["summary"],
        "lat": (record["coords"] or {}).get("lat") if record["coords"] else None,
        "lon": (record["coords"] or {}).get("lon") if record["coords"] else None,
    })

    col.update(
        ids=[rec_id],
        metadatas=[meta],
        documents=[json.dumps(record, ensure_ascii=False)],
    )
    return _row_from_record(rec_id, record)

@app.delete("/api/records/{rec_id}")
def delete_record(rec_id: str):
    col = get_collection()
    try:
        col.delete(ids=[rec_id])
    except Exception:
        pass  # idempotent
    return {"ok": True}

@app.post("/api/records/{rec_id}/refresh")
def refresh_record(rec_id: str, request: Request):
    col = get_collection()
    res = col.get(ids=[rec_id])
    if not res.get("ids"):
        raise HTTPException(status_code=404, detail="Record not found")
    doc = (res.get("documents") or ["{}"])[0]
    record = json.loads(doc)

    # Reuse existing selection
    payload = WeatherQuery(
        input_mode=record.get("input_mode"),
        place=record.get("place"),
        coords=record.get("coords"),
    )

    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    cur_raw = _call_weatherapi(WEATHERAPI_CURRENT_URL, q=q)
    f_raw = _call_weatherapi(WEATHERAPI_FORECAST_URL, q=q, extra={"days": 5, "aqi": "no", "alerts": "no"})
    current = _normalize_current(cur_raw)
    forecast = _normalize_forecast(f_raw)

    record["location_display"] = _location_display(current)
    record["weather"] = {
        "current": current,
        "forecast": forecast,
        "summary": _record_summary(current),
    }

    meta = _sanitize_metadata({
        "created_at": record.get("created_at") or _now_iso(),
        "input_mode": record.get("input_mode"),
        "place": record.get("place") or "",
        "location_display": record.get("location_display"),
        "summary": record["weather"]["summary"],
        "lat": (record.get("coords") or {}).get("lat") if record.get("coords") else None,
        "lon": (record.get("coords") or {}).get("lon") if record.get("coords") else None,
    })

    col.update(
        ids=[rec_id],
        metadatas=[meta],
        documents=[json.dumps(record, ensure_ascii=False)],
    )
    return _row_from_record(rec_id, record)

# ---------------- Export Endpoints ----------------
def _flatten_for_csv(r: Dict[str, Any]) -> Dict[str, Any]:
    c = (r.get("coords") or {})
    w = (r.get("weather") or {})
    cur = (w.get("current") or {})
    return {
        "id": r.get("id"),
        "created_at": r.get("created_at"),
        "input_mode": r.get("input_mode"),
        "place": r.get("place") or "",
        "lat": c.get("lat"),
        "lon": c.get("lon"),
        "location": cur.get("location") or r.get("location_display") or "",
        "temp_c": cur.get("temp_c"),
        "feels_like_c": cur.get("feels_like_c"),
        "humidity": cur.get("humidity"),
        "wind_kph": cur.get("wind_kph"),
        "condition": cur.get("condition_text"),
        "summary": (w.get("summary") or ""),
    }

def _to_csv(records: List[Dict[str, Any]]) -> bytes:
    if not records:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id","created_at","input_mode","place","lat","lon","location","temp_c","feels_like_c","humidity","wind_kph","condition","summary"])
        return output.getvalue().encode("utf-8")
    rows = [_flatten_for_csv(r) for r in records]
    fieldnames = list(rows[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows: writer.writerow(row)
    return output.getvalue().encode("utf-8")

def _to_xml(records: List[Dict[str, Any]]) -> bytes:
    def esc(s: Any) -> str:
        if s is None: return ""
        return (str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<records>"]
    for r in records:
        parts.append("  <record>")
        for k, v in r.items():
            if k == "weather":
                parts.append("    <weather>")
                for wk, wv in (v or {}).items():
                    if wk in ("current","forecast"):
                        parts.append(f"      <{wk}>{esc(json.dumps(wv))}</{wk}>")
                    else:
                        parts.append(f"      <{wk}>{esc(wv)}</{wk}>")
                parts.append("    </weather>")
            else:
                parts.append(f"    <{k}>{esc(v)}</{k}>")
        parts.append("  </record>")
    parts.append("</records>")
    return ("\n".join(parts)).encode("utf-8")

def _to_markdown(records: List[Dict[str, Any]]) -> bytes:
    lines = ["# Weather Records", ""]
    if not records:
        lines += ["_No records._"]
        return "\n".join(lines).encode("utf-8")
    # quick table
    lines += ["| ID | Created | Mode | Location | Summary |",
              "| --- | --- | --- | --- | --- |"]
    for r in records:
        lines.append(f"| {r.get('id','')} | {r.get('created_at','')} | {r.get('input_mode','')} | "
                     f"{r.get('location_display','')} | {(r.get('weather') or {}).get('summary','')} |")
    lines.append("")
    # details
    for r in records:
        lines += [f"## {r.get('location_display','(unknown)')} — {r.get('created_at','')}",
                  "",
                  "**Selection**",
                  "",
                  "```json",
                  json.dumps({"input_mode": r.get("input_mode"),
                              "place": r.get("place"),
                              "coords": r.get("coords")}, indent=2),
                  "```",
                  "",
                  "**Current**",
                  "",
                  "```json",
                  json.dumps((r.get("weather") or {}).get("current", {}), indent=2),
                  "```",
                  "",
                  "**Forecast (truncated)**",
                  "",
                  "```json",
                  json.dumps((r.get("weather") or {}).get("forecast", []), indent=2),
                  "```",
                  ""]
    return "\n".join(lines).encode("utf-8")

def _to_pdf(records: List[Dict[str, Any]]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin
    c.setTitle("Weather Records Export")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Weather Records")
    y -= 24
    c.setFont("Helvetica", 10)

    if not records:
        c.drawString(margin, y, "No records.")
        c.showPage(); c.save()
        return buf.getvalue()

    for r in records:
        if y < 140:
            c.showPage(); y = height - margin; c.setFont("Helvetica", 10)
        title = f"{r.get('location_display','(unknown)')} — {r.get('created_at','')}"
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, title); y -= 14
        c.setFont("Helvetica", 10)
        summary = (r.get("weather") or {}).get("summary","")
        sel = f"Mode: {r.get('input_mode','')}  |  Place: {r.get('place','')}"
        coords = r.get("coords") or {}
        if coords:
            sel += f"  |  Coords: {coords.get('lat')},{coords.get('lon')}"
        for block in [sel, f"Summary: {summary}"]:
            lines = simpleSplit(block, "Helvetica", 10, width - 2*margin)
            for line in lines:
                c.drawString(margin, y, line); y -= 12
        y -= 6

        cur = (r.get("weather") or {}).get("current", {})
        if cur:
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(margin, y, "Current:"); y -= 12
            c.setFont("Helvetica", 10)
            for k in ["location","condition_text","temp_c","feels_like_c","humidity","wind_kph","observed_at"]:
                line = f"- {k}: {cur.get(k)}"
                lines = simpleSplit(line, "Helvetica", 10, width - 2*margin)
                for ln in lines: c.drawString(margin+12, y, ln); y -= 12
        y -= 6

        fc = (r.get("weather") or {}).get("forecast", []) or []
        if fc:
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(margin, y, "Forecast:"); y -= 12
            c.setFont("Helvetica", 10)
            for d in fc[:5]:
                line = f"- {d.get('date')}: {d.get('condition_text')} | {d.get('min_c')}–{d.get('max_c')}°C, 💧{d.get('precip_mm')}mm, 💨{d.get('wind_kph')}kph"
                lines = simpleSplit(line, "Helvetica", 10, width - 2*margin)
                for ln in lines:
                    if y < 70:
                        c.showPage(); y = height - margin; c.setFont("Helvetica", 10)
                    c.drawString(margin+12, y, ln); y -= 12
        y -= 18

    c.showPage()
    c.save()
    return buf.getvalue()

@app.get("/api/export")
def export_data(
    format: str = Query(..., pattern="^(json|csv|xml|markdown|pdf)$"),
    id: Optional[str] = Query(None, description="Export a single record by ID"),
):
    records = _load_records(id)
    filename_base = f"weather_export_{id or 'all'}"

    if format == "json":
        data = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
        return Response(
            content=data,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.json"'}
        )
    if format == "csv":
        data = _to_csv(records)
        return Response(
            content=data,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.csv"'}
        )
    if format == "xml":
        data = _to_xml(records)
        return Response(
            content=data,
            media_type="application/xml",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.xml"'}
        )
    if format == "markdown":
        data = _to_markdown(records)
        return Response(
            content=data,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.md"'}
        )
    if format == "pdf":
        data = _to_pdf(records)
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.pdf"'}
        )

    raise HTTPException(status_code=400, detail="Unsupported export format.")
