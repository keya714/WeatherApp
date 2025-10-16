# backend/app/main.py
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

# ---------- Config ----------
load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")
if not API_KEY:
    raise RuntimeError("WEATHER_API_KEY not set. Put it in a .env file at project root.")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

WEATHERAPI_CURRENT_URL = "http://api.weatherapi.com/v1/current.json"
WEATHERAPI_FORECAST_URL = "http://api.weatherapi.com/v1/forecast.json"

# ---------- App ----------
app = FastAPI(title="AI Weather App (FastAPI + WeatherAPI)")

# CORS for local dev (localhost/127.0.0.1)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve /static for CSS, etc.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    html = index_file.read_text(encoding="utf-8")
    # ensure stylesheet path is correct
    html = html.replace('href="./styles.css"', 'href="/static/styles.css"')
    return HTMLResponse(html)

@app.get("/ping")
async def ping():
    return {"message": "ok"}

# ---------- Schemas (Pydantic v2) ----------
class Coords(BaseModel):
    lat: float
    lon: float

class WeatherQuery(BaseModel):
    locator_type: str = Field(..., pattern="^(auto|name|postal|coords)$")
    place: Optional[str] = None
    postal: Optional[str] = None
    coords: Optional[Coords] = None
    autodetect: Optional[bool] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @model_validator(mode="after")
    def _require_fields_for_locator(self):
        lt = self.locator_type
        if lt == "name" and not (self.place and self.place.strip()):
            raise ValueError("place is required for locator_type=name")
        if lt == "postal" and not (self.postal and self.postal.strip()):
            raise ValueError("postal is required for locator_type=postal")
        if lt == "coords" and not self.coords:
            raise ValueError("coords is required for locator_type=coords")
        return self

# ---------- Helpers ----------
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

def _build_weatherapi_q(payload: WeatherQuery, client_ip: Optional[str]) -> str:
    if payload.locator_type == "coords" and payload.coords:
        return f"{payload.coords.lat},{payload.coords.lon}"
    if payload.locator_type == "postal" and payload.postal:
        return payload.postal
    if payload.locator_type == "name" and payload.place:
        return payload.place
    # autodetect by IP (server-side)
    return "auto:ip"

def _call_weatherapi(url: str, q: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = {"key": API_KEY, "q": q, "lang": "en"}
    if extra:
        params.update(extra)
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream weather API error: {e}")

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
        day_info = d.get("day", {}) or {}
        cond_text = (day_info.get("condition") or {}).get("text", "")
        out.append({
            "date": d.get("date"),
            "condition_code": _condition_code_from_text(cond_text),
            "condition_text": cond_text,
            "max_c": day_info.get("maxtemp_c"),
            "min_c": day_info.get("mintemp_c"),
            "precip_mm": day_info.get("totalprecip_mm"),
            "wind_kph": day_info.get("maxwind_kph"),
        })
    return out

# ---------- Routes ----------
@app.post("/api/weather/current")
def api_current(payload: WeatherQuery, request: Request):
    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    data = _call_weatherapi(WEATHERAPI_CURRENT_URL, q=q)
    if "location" not in data or "current" not in data:
        raise HTTPException(status_code=500, detail="Unexpected response from WeatherAPI (current).")
    return _normalize_current(data)

@app.post("/api/weather/forecast")
def api_forecast(payload: WeatherQuery, request: Request):
    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    data = _call_weatherapi(WEATHERAPI_FORECAST_URL, q=q, extra={"days": 5, "aqi": "no", "alerts": "no"})
    if "forecast" not in data:
        raise HTTPException(status_code=500, detail="Unexpected response from WeatherAPI (forecast).")
    return _normalize_forecast(data)
