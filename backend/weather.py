# backend/app/main.py
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
# ---------- Config ----------
load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")
if not API_KEY:
    raise RuntimeError("WEATHER_API_KEY not set. Put it in a .env file.")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

WEATHERAPI_CURRENT_URL = "http://api.weatherapi.com/v1/current.json"
WEATHERAPI_FORECAST_URL = "http://api.weatherapi.com/v1/forecast.json"


# ---------- FastAPI ----------
app = FastAPI(title="AI Weather App (FastAPI + WeatherAPI)")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://0.0.0.0:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the frontend HTML. Rewrites stylesheet to /static if needed."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    html = index_file.read_text(encoding="utf-8")
    # make sure CSS is served from /static/styles.css
    html = html.replace('href="./styles.css"', 'href="/static/styles.css"')
    return HTMLResponse(html)


@app.get("/ping")
async def ping():
    return {"message": "ok"}


# ---------- Request Schemas (match the frontend) ----------
class Coords(BaseModel):
    lat: float
    lon: float


class WeatherQuery(BaseModel):
    locator_type: str = Field(..., regex="^(auto|name|postal|coords)$")
    place: Optional[str] = None
    postal: Optional[str] = None
    coords: Optional[Coords] = None
    autodetect: Optional[bool] = None
    start_date: Optional[str] = None  # not used by WeatherAPI current/5-day
    end_date: Optional[str] = None

    @validator("place", always=True)
    def _clean_place(cls, v, values):
        if values.get("locator_type") == "name" and not v:
            raise ValueError("place is required for locator_type=name")
        return v

    @validator("postal", always=True)
    def _clean_postal(cls, v, values):
        if values.get("locator_type") == "postal" and not v:
            raise ValueError("postal is required for locator_type=postal")
        return v

    @validator("coords", always=True)
    def _clean_coords(cls, v, values):
        if values.get("locator_type") == "coords" and not v:
            raise ValueError("coords is required for locator_type=coords")
        return v


# ---------- Helpers ----------
def _condition_code_from_text(text: str) -> str:
    """Map WeatherAPI condition text to simple icons codes used by the frontend."""
    t = (text or "").lower()
    if any(k in t for k in ["thunder", "storm", "lightning"]):
        return "thunder"
    if any(k in t for k in ["snow", "sleet", "blizzard", "flurr"]):
        return "snow"
    if any(k in t for k in ["drizzle", "shower"]):
        return "drizzle"
    if any(k in t for k in ["rain", "downpour"]):
        return "rain"
    if any(k in t for k in ["fog", "mist", "haze"]):
        return "fog"
    if any(k in t for k in ["wind", "breeze", "gale"]):
        return "wind"
    if "cloud" in t:
        return "cloudy" if "overcast" in t else "partly-cloudy"
    if any(k in t for k in ["clear", "sunny", "bright"]):
        return "clear"
    return "partly-cloudy"


def _build_weatherapi_q(payload: WeatherQuery, client_ip: Optional[str]) -> str:
    """Convert our payload to WeatherAPI q parameter."""
    if payload.locator_type == "coords" and payload.coords:
        return f"{payload.coords.lat},{payload.coords.lon}"
    if payload.locator_type == "postal" and payload.postal:
        return payload.postal
    if payload.locator_type == "name" and payload.place:
        return payload.place
    # autodetect (server-side IP). WeatherAPI supports "auto:ip"
    # Note: we could pass the actual client IP to a paid tier; here we rely on server IP.
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
    """Shape response to what index.html expects in renderCurrent()."""
    loc = data.get("location", {})
    cur = data.get("current", {})

    condition_text = (cur.get("condition") or {}).get("text", "")
    code = _condition_code_from_text(condition_text)

    out = {
        "location": ", ".join([v for v in [loc.get("name"), loc.get("region"), loc.get("country")] if v]),
        "condition_code": code,
        "condition_text": condition_text,
        "temp_c": cur.get("temp_c"),
        "feels_like_c": cur.get("feelslike_c"),
        "humidity": cur.get("humidity"),
        "wind_kph": cur.get("wind_kph"),
        "observed_at": loc.get("localtime") or cur.get("last_updated"),
    }
    return out


def _normalize_forecast(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Shape response to what index.html expects in renderForecast(). (max 5 days)"""
    days = (data.get("forecast", {}) or {}).get("forecastday", []) or []
    out: List[Dict[str, Any]] = []
    for d in days[:5]:
        day_info = d.get("day", {}) or {}
        cond_text = (day_info.get("condition") or {}).get("text", "")
        out.append(
            {
                "date": d.get("date"),
                "condition_code": _condition_code_from_text(cond_text),
                "condition_text": cond_text,
                "max_c": day_info.get("maxtemp_c"),
                "min_c": day_info.get("mintemp_c"),
                "precip_mm": day_info.get("totalprecip_mm"),
                "wind_kph": day_info.get("maxwind_kph"),
            }
        )
    return out


# ---------- API: Current Weather ----------
@app.post("/api/weather/current")
def api_current(payload: WeatherQuery, request: Request):
    """
    Request body (from frontend):
    {
      "locator_type": "auto" | "name" | "postal" | "coords",
      "place": "...", "postal": "...",
      "coords": {"lat":..,"lon":..},
      "autodetect": true|false,
      "start_date": "...", "end_date": "..."
    }
    """
    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    print(q)
    data = _call_weatherapi(WEATHERAPI_CURRENT_URL, q=q)
    # Basic guard against invalid responses
    if "location" not in data or "current" not in data:
        raise HTTPException(status_code=500, detail="Unexpected response from WeatherAPI (current).")
    return _normalize_current(data)


# ---------- API: 5-Day Forecast ----------
@app.post("/api/weather/forecast")
def api_forecast(payload: WeatherQuery, request: Request):
    """
    Same body as /api/weather/current. Returns an array of up to 5 daily items.
    """
    q = _build_weatherapi_q(payload, client_ip=request.client.host if request.client else None)
    # WeatherAPI free tier supports up to 3 days; if limited, it will return fewer than 5.
    data = _call_weatherapi(WEATHERAPI_FORECAST_URL, q=q, extra={"days": 5, "aqi": "no", "alerts": "no"})
    if "forecast" not in data:
        raise HTTPException(status_code=500, detail="Unexpected response from WeatherAPI (forecast).")
    return _normalize_forecast(data)
