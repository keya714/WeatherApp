# backend/app/main.py
import os, re, json, uuid, io, csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, Response
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

# ---------------- Config ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
API_KEY = os.getenv("WEATHER_API_KEY")
if not API_KEY:
    raise RuntimeError("WEATHER_API_KEY not set in environment variables.")

# Use /tmp for Vercel serverless writable directory
DATA_DIR = Path("/tmp/chroma")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

WEATHERAPI_CURRENT_URL = "http://api.weatherapi.com/v1/current.json"
WEATHERAPI_FORECAST_URL = "http://api.weatherapi.com/v1/forecast.json"

# ---------------- App ----------------
app = FastAPI(title="AI Weather App (Vercel Serverless)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend minimally
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
    if ok(lon, lat): return (lon, lat)
    raise HTTPException(
        status_code=422,
        detail="Invalid coordinates. Use 'lat,lon' in decimal degrees, e.g. 48.8567,2.3508"
    )

def _build_weatherapi_q(payload: WeatherQuery, client_ip: Optional[str]) -> str:
    if payload.input_mode == "coords" and payload.coords:
        lat, lon = _normalize_coords(payload.coords.lat, payload.coords.lon)
        return f"{lat:.6f},{lon:.6f}"
    if payload.input_mode == "name" and payload.place:
        m = _latlon_re.match(payload.place.strip())
        if m:
            lat, lon = _normalize_coords(float(m.group(1)), float(m.group(2)))
            return f"{lat:.6f},{lon:.6f}"
        return payload.place.strip()
    return "auto:ip"

def _call_weatherapi(url: str, q: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = {"key": API_KEY, "q": q, "lang": "en"}
    if extra: params.update(extra)
    try:
        r = requests.get(url, params=params, timeout=8)  # reduced timeout for Vercel
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

# ---- record utils ----
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
        "weather": record.get("weather"),
        "place": record.get("place"),
        "coords": record.get("coords"),
    }

# ---------------- ChromaDB Setup ----------------
def get_collection():
    client = chromadb.PersistentClient(path=str(DATA_DIR), settings=Settings(allow_reset=False))
    try:
        return client.get_collection("weather_records")
    except Exception:
        return client.create_collection("weather_records")

def _load_records(rec_id: Optional[str] = None) -> List[Dict[str, Any]]:
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

# ---------------- AI Advice ----------------
def _ai_clothing_suggestion(current: Dict[str, Any]) -> str:
    if not GOOGLE_API_KEY:
        return ""
    try:
        llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
        )
        temp = current.get("temp_c")
        feels = current.get("feels_like_c")
        cond = current.get("condition_text")
        wind = current.get("wind_kph")
        humid = current.get("humidity")
        loc = current.get("location")
        prompt = f"""
        You are a concise weather stylist. Provide 2–4 bullet suggestions for clothing and gear.
        Location: {loc}
        Condition: {cond}
        Temperature: {temp}°C (feels like {feels}°C)
        Wind: {wind} kph
        Humidity: {humid}%
        Output plain text with bullet lines ("• ...").
        """
        r = llm_gemini.invoke(prompt)
        text = (r.content or "").strip()
        return "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()][:6])
    except Exception:
        return ""

# ---------------- Weather Endpoints ----------------
@app.post("/api/weather/current")
def api_current(payload: WeatherQuery, request: Request):
    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    data = _call_weatherapi(WEATHERAPI_CURRENT_URL, q=q)
    if "location" not in data or "current" not in data:
        raise HTTPException(status_code=500, detail="Unexpected response from WeatherAPI.")
    current = _normalize_current(data)
    advice = _ai_clothing_suggestion(current)
    out = current.copy()
    out['ai_advice'] = advice
    return out

@app.post("/api/weather/forecast")
def api_forecast(payload: WeatherQuery, request: Request):
    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    data = _call_weatherapi(WEATHERAPI_FORECAST_URL, q=q, extra={"days": 5, "aqi": "no", "alerts": "no"})
    if "forecast" not in data:
        raise HTTPException(status_code=500, detail="Unexpected response from WeatherAPI (forecast).")
    return _normalize_forecast(data)

# ---------------- CRUD & Export endpoints remain largely unchanged ----------------
# Ensure all returned values are JSON serializable
