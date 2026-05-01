import os
from pathlib import Path
import time

import gdown
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Import custom classes from our training script
from flight_delay_assignment import (
    FlightFeatureEngineer,
    OOFTargetEncoder,
    FullPreprocessor,
    WeightedAdaBoost,
    haversine
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "output" / "models" / "model.joblib"
AIRLINES_PATH = BASE_DIR / "airlines.csv"
AIRPORTS_PATH = BASE_DIR / "airports.csv"
MODEL_GDRIVE_URL_ENV = "MODEL_GDRIVE_URL"
MODEL_GDRIVE_FILE_ID_ENV = "MODEL_GDRIVE_FILE_ID"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SkyPredict — Flight Delay Intelligence",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# THEME STATE
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

# ─────────────────────────────────────────────
# CSS — PREMIUM DESIGN SYSTEM
# ─────────────────────────────────────────────
DARK_COLORS = {
    "bg":         "#0a0e1a",
    "surface":    "#111827",
    "surface2":   "#1a2235",
    "border":     "#1f2d45",
    "text":       "#e8eaf0",
    "muted":      "#6b7a99",
    "accent":     "#3d8ef8",
    "accent2":    "#7c5cfc",
    "success":    "#22d37f",
    "danger":     "#ff4757",
    "warning":    "#ffa940",
}

LIGHT_COLORS = {
    "bg":         "#f0f4fc",
    "surface":    "#ffffff",
    "surface2":   "#e8edf8",
    "border":     "#d0d8ee",
    "text":       "#0f172a",
    "muted":      "#64748b",
    "accent":     "#2563eb",
    "accent2":    "#7c3aed",
    "success":    "#16a34a",
    "danger":     "#dc2626",
    "warning":    "#d97706",
}

C = DARK_COLORS if dark else LIGHT_COLORS

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── RESET & BASE ── */
  html, body, [class*="css"] {{
      font-family: 'DM Sans', sans-serif;
      background-color: {C['bg']} !important;
      color: {C['text']} !important;
  }}

  .stApp {{ background-color: {C['bg']} !important; }}

  /* ── SIDEBAR ── */
  section[data-testid="stSidebar"] {{
      background: {C['surface']} !important;
      border-right: 1px solid {C['border']} !important;
  }}
  section[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}

  /* ── INPUTS ── */
  div[data-baseweb="select"] > div,
  div[data-baseweb="input"] > div {{
      background: {C['surface2']} !important;
      border: 1px solid {C['border']} !important;
      border-radius: 10px !important;
      color: {C['text']} !important;
  }}
  input, .stNumberInput input {{
      background: {C['surface2']} !important;
      color: {C['text']} !important;
      border-radius: 10px !important;
  }}

  /* ── NUMBER INPUT STEPPER BUTTONS (fix +/− visibility in both themes) ── */
  /* Target the stepper wrapper */
  [data-testid="stNumberInput"] button,
  .stNumberInput button,
  div[data-baseweb="input"] ~ div button,
  [data-testid="stNumberInputField"] + div button {{
      background: {C['accent']} !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: 6px !important;
      font-size: 18px !important;
      font-weight: 700 !important;
      min-width: 32px !important;
      height: 32px !important;
      line-height: 1 !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      transition: background 0.2s ease !important;
      cursor: pointer !important;
  }}
  [data-testid="stNumberInput"] button:hover,
  .stNumberInput button:hover {{
      background: {C['accent2']} !important;
      color: #ffffff !important;
  }}
  /* Ensure stepper container background matches surface */
  [data-testid="stNumberInput"] > div {{
      background: {C['surface2']} !important;
      border: 1px solid {C['border']} !important;
      border-radius: 10px !important;
      overflow: hidden;
  }}
  /* Inner input text color fix */
  [data-testid="stNumberInput"] input {{
      background: transparent !important;
      color: {C['text']} !important;
      font-size: 15px !important;
      font-weight: 500 !important;
  }}

  /* ── BUTTONS ── */
  .stButton > button {{
      border-radius: 10px;
      font-family: 'Syne', sans-serif;
      font-weight: 700;
      font-size: 15px;
      letter-spacing: 0.5px;
      border: none;
      transition: all 0.25s ease;
      cursor: pointer;
  }}
  .stButton > button[kind="primary"] {{
      background: linear-gradient(135deg, {C['accent']}, {C['accent2']}) !important;
      color: #fff !important;
      padding: 14px 28px;
      box-shadow: 0 4px 20px rgba(61,142,248,0.35);
  }}
  .stButton > button[kind="primary"]:hover {{
      transform: translateY(-2px);
      box-shadow: 0 8px 28px rgba(61,142,248,0.55);
  }}
  .stButton > button[kind="secondary"] {{
      background: {C['surface2']} !important;
      color: {C['muted']} !important;
      border: 1px solid {C['border']} !important;
  }}
  .stButton > button[kind="secondary"]:hover {{
      border-color: {C['accent']} !important;
      color: {C['accent']} !important;
  }}

  /* ── EXPANDERS ── */
  details summary {{
      background: {C['surface2']} !important;
      border-radius: 10px !important;
      padding: 10px 14px !important;
      font-weight: 600;
  }}

  /* ── DIVIDER ── */
  hr {{ border-color: {C['border']} !important; }}

  /* ── LABELS ── */
  label, .stSelectbox label, .stNumberInput label {{
      font-size: 12px !important;
      font-weight: 500 !important;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: {C['muted']} !important;
  }}

  /* ── METRIC CARDS ── */
  [data-testid="metric-container"] {{
      background: {C['surface2']} !important;
      border: 1px solid {C['border']} !important;
      border-radius: 12px !important;
      padding: 16px !important;
  }}
  [data-testid="stMetricValue"] {{
      color: {C['accent']} !important;
      font-family: 'Syne', sans-serif !important;
      font-weight: 700 !important;
  }}

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {C['bg']}; }}
  ::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 4px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {C['accent']}; }}

  /* ── CUSTOM COMPONENTS ── */
  .hero-title {{
      font-family: 'Syne', sans-serif;
      font-size: clamp(36px, 5vw, 58px);
      font-weight: 800;
      background: linear-gradient(135deg, {C['accent']}, {C['accent2']}, {C['success']});
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.1;
      margin: 0;
  }}
  .hero-sub {{
      font-size: 16px;
      color: {C['muted']};
      margin-top: 8px;
      font-weight: 300;
      letter-spacing: 0.3px;
  }}
  .gradient-bar {{
      height: 3px;
      background: linear-gradient(90deg, {C['accent']}, {C['accent2']}, {C['success']}, transparent);
      border-radius: 2px;
      margin: 20px 0 28px 0;
  }}

  .summary-card {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-radius: 14px;
      padding: 20px 24px;
      margin-bottom: 24px;
  }}
  .summary-title {{
      font-family: 'Syne', sans-serif;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: {C['muted']};
      margin-bottom: 14px;
  }}
  .route-display {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-family: 'Syne', sans-serif;
      font-size: 26px;
      font-weight: 800;
      color: {C['text']};
  }}
  .route-arrow {{
      color: {C['accent']};
      font-size: 20px;
  }}
  .route-meta {{
      font-size: 13px;
      color: {C['muted']};
      margin-top: 10px;
      display: flex;
      gap: 20px;
  }}

  .result-card {{
      padding: 28px 24px;
      border-radius: 14px;
      text-align: center;
      margin-bottom: 20px;
      position: relative;
      overflow: hidden;
  }}
  .result-card::before {{
      content: '';
      position: absolute;
      inset: 0;
      opacity: 0.06;
      background: radial-gradient(circle at 30% 50%, white, transparent 70%);
  }}
  .delayed-card {{
      background: linear-gradient(135deg, rgba(255,71,87,0.15), rgba(255,71,87,0.05));
      border: 1.5px solid {C['danger']};
      animation: pulseRed 2s ease-in-out infinite;
  }}
  .ontime-card {{
      background: linear-gradient(135deg, rgba(34,211,127,0.15), rgba(34,211,127,0.05));
      border: 1.5px solid {C['success']};
      animation: pulseGreen 2s ease-in-out infinite;
  }}
  @keyframes pulseRed {{
      0%, 100% {{ box-shadow: 0 0 0 0 rgba(255,71,87,0.2); }}
      50% {{ box-shadow: 0 0 20px 6px rgba(255,71,87,0.15); }}
  }}
  @keyframes pulseGreen {{
      0%, 100% {{ box-shadow: 0 0 0 0 rgba(34,211,127,0.2); }}
      50% {{ box-shadow: 0 0 20px 6px rgba(34,211,127,0.15); }}
  }}
  .result-label {{
      font-family: 'Syne', sans-serif;
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 2px;
      margin-bottom: 6px;
  }}
  .result-status {{
      font-family: 'Syne', sans-serif;
      font-size: 36px;
      font-weight: 800;
  }}
  .delayed-text {{ color: {C['danger']}; }}
  .ontime-text {{ color: {C['success']}; }}

  .insight-card {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-radius: 12px;
      padding: 14px 16px;
      text-align: center;
  }}
  .insight-icon {{
      font-size: 22px;
      margin-bottom: 6px;
  }}
  .insight-label {{
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: {C['muted']};
      margin-bottom: 4px;
  }}
  .insight-value {{
      font-family: 'Syne', sans-serif;
      font-size: 15px;
      font-weight: 700;
      color: {C['text']};
  }}

  .why-box {{
      background: linear-gradient(135deg, {C['surface']}, {C['surface2']});
      border: 1px solid {C['border']};
      border-left: 3px solid {C['accent']};
      border-radius: 12px;
      padding: 18px 20px;
      font-size: 14px;
      line-height: 1.7;
      color: {C['text']};
  }}
  .why-title {{
      font-family: 'Syne', sans-serif;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: {C['accent']};
      margin-bottom: 8px;
  }}

  .error-box {{
      background: rgba(255,71,87,0.08);
      border: 1px solid {C['danger']};
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 12px;
      font-size: 14px;
      color: {C['danger']};
  }}

  .sidebar-section-title {{
      font-family: 'Syne', sans-serif;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.8px;
      color: {C['muted']};
      margin: 18px 0 8px 0;
  }}

  .footer {{
      text-align: center;
      margin-top: 60px;
      padding: 28px 0 16px 0;
      border-top: 1px solid {C['border']};
  }}
  .footer-title {{
      font-family: 'Syne', sans-serif;
      font-size: 16px;
      font-weight: 700;
      color: {C['text']};
      margin-bottom: 6px;
  }}
  .footer-sub {{
      font-size: 12px;
      color: {C['muted']};
      margin-bottom: 10px;
  }}
  .footer-stack {{
      display: inline-flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: center;
  }}
  .stack-tag {{
      background: {C['surface2']};
      border: 1px solid {C['border']};
      border-radius: 20px;
      padding: 4px 12px;
      font-size: 11px;
      color: {C['muted']};
      font-weight: 500;
  }}

  /* ── LOADING STEPS ── */
  .loading-step {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 0;
      font-size: 13px;
      color: {C['muted']};
  }}
  .loading-step.active {{ color: {C['accent']}; font-weight: 500; }}
  .loading-step.done {{ color: {C['success']}; }}

  .theme-toggle {{
      position: fixed;
      top: 14px;
      right: 14px;
      z-index: 9999;
  }}

  /* ── PLOTLY CHART BG FIX ── */
  .js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {{
      background: transparent !important;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────
def get_runtime_value(name):
    value = os.getenv(name)
    if value:
        return value

    try:
        value = st.secrets[name]
    except Exception:
        value = None

    return str(value) if value else None


def resolve_model_download_url():
    model_url = get_runtime_value(MODEL_GDRIVE_URL_ENV)
    if model_url:
        return model_url

    model_file_id = get_runtime_value(MODEL_GDRIVE_FILE_ID_ENV)
    if model_file_id:
        return f"https://drive.google.com/uc?id={model_file_id}"

    return None


def download_model():
    download_url = resolve_model_download_url()
    if not download_url:
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Set {MODEL_GDRIVE_URL_ENV} or {MODEL_GDRIVE_FILE_ID_ENV} "
            "to enable automatic download."
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with st.spinner("Downloading model artifact from Google Drive..."):
        downloaded_path = gdown.download(
            url=download_url,
            output=str(MODEL_PATH),
            quiet=False,
            
        )

    if not downloaded_path or not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model download failed for {MODEL_PATH}.")

    return MODEL_PATH


def ensure_model_file():
    if not MODEL_PATH.exists():
        download_model()

    return MODEL_PATH


@st.cache_resource
def load_model():
    return joblib.load(str(ensure_model_file()))

@st.cache_data
def load_aux_data():
    airlines = pd.read_csv(AIRLINES_PATH)
    airports = pd.read_csv(AIRPORTS_PATH)
    return airlines, airports

try:
    pipeline_obj = load_model()
    preprocessor   = pipeline_obj['preprocessor']
    selected_features = pipeline_obj['selected_features']
    model          = pipeline_obj.get('calibrated_model', pipeline_obj.get('base_model'))
    threshold      = pipeline_obj['best_threshold']
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

airlines_df, airports_df = load_aux_data()

airline_dict   = pd.Series(airlines_df.AIRLINE.values, index=airlines_df.IATA_CODE).to_dict()
airline_options = [f"{k} - {v}" for k, v in airline_dict.items()]

airport_dict   = pd.Series(
    airports_df.CITY.values + ", " + airports_df.STATE.values,
    index=airports_df.IATA_CODE
).to_dict()
airport_options = [f"{k} - {v}" for k, v in airport_dict.items()]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_airport_coords(iata):
    row = airports_df[airports_df.IATA_CODE == iata]
    if row.empty:
        return None, None
    return float(row.iloc[0].LATITUDE), float(row.iloc[0].LONGITUDE)

def is_peak_hour(hhmm):
    h = hhmm // 100
    return (7 <= h <= 9) or (17 <= h <= 20)

def make_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "family": "Syne", "color": C["text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": C["muted"], "size": 11}, "tickcolor": C["border"]},
            "bar": {"color": C["danger"] if prob >= 0.6 else (C["warning"] if prob >= 0.3 else C["success"]), "thickness": 0.25},
            "bgcolor": C["surface2"],
            "bordercolor": C["border"],
            "borderwidth": 1,
            "steps": [
                {"range": [0,  30], "color": "rgba(34,211,127,0.12)"},
                {"range": [30, 60], "color": "rgba(255,169,64,0.12)"},
                {"range": [60,100], "color": "rgba(255,71,87,0.12)"},
            ],
            "threshold": {
                "line": {"color": C["accent2"], "width": 2},
                "thickness": 0.8,
                "value": round(threshold * 100, 1),
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=20, r=20),
        height=220,
        font={"family": "DM Sans", "color": C["text"]},
    )
    return fig

def make_route_map(orig_iata, dest_iata):
    orig_lat, orig_lon = get_airport_coords(orig_iata)
    dest_lat, dest_lon = get_airport_coords(dest_iata)
    if orig_lat is None or dest_lat is None:
        return None

    fig = go.Figure()

    # Arc line
    lats = np.linspace(orig_lat, dest_lat, 60)
    lons = np.linspace(orig_lon, dest_lon, 60)
    mid = len(lats) // 2
    lats[mid] += abs(dest_lat - orig_lat) * 0.25

    fig.add_trace(go.Scattergeo(
        lat=lats.tolist(),
        lon=lons.tolist(),
        mode="lines",
        line=dict(width=2.5, color=C["accent"]),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Markers
    for lat, lon, iata, label in [
        (orig_lat, orig_lon, orig_iata, "Origin"),
        (dest_lat, dest_lon, dest_iata, "Destination"),
    ]:
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon],
            mode="markers+text",
            marker=dict(size=12, color=C["accent2"], symbol="circle",
                        line=dict(width=2, color=C["accent"])),
            text=[iata],
            textposition="top center",
            textfont=dict(size=13, family="Syne", color=C["text"]),
            name=label,
            hovertemplate=f"<b>{iata}</b><extra></extra>",
        ))

    fig.update_geos(
        scope="usa",
        bgcolor="rgba(0,0,0,0)",
        showland=True, landcolor=C["surface2"],
        showocean=True, oceancolor=C["surface"],
        showlakes=True, lakecolor=C["bg"],
        showcoastlines=True, coastlinecolor=C["border"],
        showframe=False,
        showcountries=True, countrycolor=C["border"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=280,
        showlegend=False,
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo / brand
    st.markdown(f"""
    <div style="padding:4px 0 18px 0; border-bottom:1px solid {C['border']}; margin-bottom:16px">
      <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;
           background:linear-gradient(135deg,{C['accent']},{C['accent2']});
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        🛫 SkyPredict
      </div>
      <div style="font-size:11px;color:{C['muted']};margin-top:2px;">ML-Powered Delay Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    theme_label = "☀️ Light Mode" if dark else "🌙 Dark Mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown(f'<div class="sidebar-section-title">🗓️ Date</div>', unsafe_allow_html=True)
    month      = st.selectbox("Month", range(1, 13), index=4,
                              format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
    day        = st.selectbox("Day of Month", range(1, 32), index=14)
    day_of_week = st.selectbox("Day of Week", range(1, 8), index=0,
                               format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1])

    st.markdown(f'<div class="sidebar-section-title">✈️ Flight</div>', unsafe_allow_html=True)
    default_airline = next((i for i, o in enumerate(airline_options) if o.startswith("AA")), 0)
    airline_sel = st.selectbox("Airline", options=airline_options, index=default_airline)
    airline     = airline_sel.split(" - ")[0]
    airline_name = airline_sel.split(" - ")[1] if " - " in airline_sel else airline

    st.markdown(f'<div class="sidebar-section-title">📍 Route</div>', unsafe_allow_html=True)
    jfk_idx = next((i for i, o in enumerate(airport_options) if o.startswith("JFK")), 0)
    lax_idx = next((i for i, o in enumerate(airport_options) if o.startswith("LAX")), 1)
    origin_sel = st.selectbox("Origin Airport", options=airport_options, index=jfk_idx)
    dest_sel   = st.selectbox("Destination Airport", options=airport_options, index=lax_idx)
    origin     = origin_sel.split(" - ")[0]
    dest       = dest_sel.split(" - ")[0]

    st.markdown(f'<div class="sidebar-section-title">⏱️ Time & Distance</div>', unsafe_allow_html=True)

    # ── Departure time: split HH / MM ──
    st.markdown(
        f'<div style="font-size:12px;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.8px;color:{C["muted"]};margin-bottom:6px;">'
        f'Scheduled Departure</div>',
        unsafe_allow_html=True,
    )
    dep_col1, sep_col, dep_col2 = st.columns([5, 1, 5])
    with dep_col1:
        dep_hour = st.selectbox(
            "Hour",
            options=list(range(0, 24)),
            index=15,                           # default 15 = 3 PM
            format_func=lambda h: f"{h:02d}",
            label_visibility="collapsed",
            key="dep_hour",
            help="Hour (00–23)",
        )
    with sep_col:
        st.markdown(
            f'<div style="text-align:center;font-size:22px;font-weight:800;'
            f'color:{C["accent"]};padding-top:6px;line-height:1;">:</div>',
            unsafe_allow_html=True,
        )
    with dep_col2:
        dep_minute = st.selectbox(
            "Min",
            options=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            index=6,                            # default 30 → 15:30
            format_func=lambda m: f"{m:02d}",
            label_visibility="collapsed",
            key="dep_minute",
            help="Minute (00–55)",
        )
    # Combine back into HHMM integer for the model
    scheduled_departure = dep_hour * 100 + dep_minute

    # Show formatted time preview
    period = "AM" if dep_hour < 12 else "PM"
    disp_h = dep_hour % 12 or 12
    st.markdown(
        f'<div style="font-size:12px;color:{C["muted"]};margin-top:2px;'
        f'margin-bottom:8px;text-align:center;">'
        f'🕐 {dep_hour:02d}:{dep_minute:02d} · {disp_h}:{dep_minute:02d} {period}</div>',
        unsafe_allow_html=True,
    )

    scheduled_time = st.number_input("Scheduled Duration (mins)", value=300, step=5,
                                      min_value=1, max_value=1440)
    distance       = st.number_input("Distance (miles)", value=2500, step=50,
                                      min_value=1)

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button("🚀 Predict Delay", use_container_width=True, type="primary")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("↩ Reset Inputs", use_container_width=True, type="secondary"):
        for key in list(st.session_state.keys()):
            if key != "dark_mode":
                del st.session_state[key]
        st.rerun()

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# ── HERO ──
col_hero, col_badge = st.columns([3, 1])
with col_hero:
    st.markdown('<h1 class="hero-title">Flight Delay Intelligence</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="hero-sub">Real-time ML predictions — know before you go ✈️</p>', unsafe_allow_html=True)
with col_badge:
    st.markdown(f"""
    <div style="text-align:right;padding-top:14px">
      <span style="background:{C['surface2']};border:1px solid {C['border']};
            border-radius:20px;padding:6px 14px;font-size:11px;
            color:{C['muted']};font-weight:600;letter-spacing:0.5px;">
        ⚡ AdaBoost · Calibrated
      </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="gradient-bar"></div>', unsafe_allow_html=True)

# ── FLIGHT SUMMARY CARD ──
origin_city  = origin_sel.split(" - ")[1] if " - " in origin_sel else origin
dest_city    = dest_sel.split(" - ")[1]   if " - " in dest_sel   else dest
dep_h = scheduled_departure // 100
dep_m = scheduled_departure % 100
dep_fmt = f"{dep_h:02d}:{dep_m:02d}"
arr_h = (dep_h + scheduled_time // 60) % 24
arr_m = (dep_m + scheduled_time % 60) % 60
arr_fmt = f"{arr_h:02d}:{arr_m:02d}"

st.markdown(f"""
<div class="summary-card">
  <div class="summary-title">✦ Flight Summary</div>
  <div class="route-display">
    <span>{origin}</span>
    <span class="route-arrow">→</span>
    <span>{dest}</span>
  </div>
  <div style="font-size:13px;color:{C['muted']};margin-top:4px;">{origin_city} → {dest_city}</div>
  <div class="route-meta">
    <span>🏢 {airline_name}</span>
    <span>🕐 {dep_fmt} → ~{arr_fmt}</span>
    <span>📏 {distance:,} mi</span>
    <span>⏱ {scheduled_time} min</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_clicked:
    # ── VALIDATION ──
    errors = []
    if distance <= 0:
        errors.append("📏 Distance must be greater than 0 miles.")
    if scheduled_time <= 0:
        errors.append("⏱ Scheduled duration must be greater than 0 minutes.")
    if origin == dest:
        errors.append("📍 Origin and destination airports must be different.")

    if errors:
        for err in errors:
            st.markdown(f'<div class="error-box">❌ {err}</div>', unsafe_allow_html=True)
    else:
        # ── ANIMATED LOADING ──
        loading_ph = st.empty()
        steps = [
            ("Processing flight inputs…",       0.5),
            ("Applying feature engineering…",   0.6),
            ("Running delay prediction model…", 0.7),
            ("Calibrating probabilities…",      0.4),
        ]
        with loading_ph.container():
            st.markdown(f"""
            <div style="background:{C['surface']};border:1px solid {C['border']};
                        border-radius:12px;padding:20px 24px;margin-bottom:16px;">
              <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
                          color:{C['accent']};margin-bottom:12px;">
                ✦ Analyzing your flight…
              </div>
            """, unsafe_allow_html=True)
            prog = st.progress(0)
            msg  = st.empty()
            for i, (label, delay) in enumerate(steps):
                msg.markdown(f"<div class='loading-step active'>◈ {label}</div>", unsafe_allow_html=True)
                prog.progress((i + 1) / len(steps))
                time.sleep(delay)
            msg.markdown(f"<div class='loading-step done'>✓ Prediction ready</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        time.sleep(0.3)
        loading_ph.empty()

        try:
            # ── BUILD DATAFRAME ──
            data = {
                'MONTH': [month], 'DAY': [day], 'DAY_OF_WEEK': [day_of_week],
                'AIRLINE': [airline], 'ORIGIN_AIRPORT': [origin],
                'DESTINATION_AIRPORT': [dest], 'SCHEDULED_DEPARTURE': [scheduled_departure],
                'SCHEDULED_TIME': [scheduled_time], 'DISTANCE': [distance]
            }
            df = pd.DataFrame(data)

            df = df.merge(airlines_df, left_on='AIRLINE', right_on='IATA_CODE', how='left')
            df.rename(columns={'AIRLINE_y': 'AIRLINE_NAME', 'AIRLINE_x': 'AIRLINE'}, inplace=True)
            if 'IATA_CODE' in df.columns: df.drop(columns=['IATA_CODE'], inplace=True)

            df = df.merge(airports_df[['IATA_CODE','CITY','STATE','LATITUDE','LONGITUDE']],
                          left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')
            df.rename(columns={'CITY':'ORIGIN_CITY','STATE':'ORIGIN_STATE',
                               'LATITUDE':'ORIGIN_LATITUDE','LONGITUDE':'ORIGIN_LONGITUDE'}, inplace=True)
            if 'IATA_CODE' in df.columns: df.drop(columns=['IATA_CODE'], inplace=True)

            df = df.merge(airports_df[['IATA_CODE','CITY','STATE','LATITUDE','LONGITUDE']],
                          left_on='DESTINATION_AIRPORT', right_on='IATA_CODE', how='left')
            df.rename(columns={'CITY':'DEST_CITY','STATE':'DEST_STATE',
                               'LATITUDE':'DEST_LATITUDE','LONGITUDE':'DEST_LONGITUDE'}, inplace=True)
            if 'IATA_CODE' in df.columns: df.drop(columns=['IATA_CODE'], inplace=True)

            X_processed = preprocessor.transform(df)
            X_selected  = X_processed[selected_features]
            prob        = model.predict_proba(X_selected)[0][1]
            pred        = prob >= threshold

            # ── RESULT CARD ──
            st.divider()
            st.markdown(f"""
            <div style="font-family:'Syne',sans-serif;font-size:11px;font-weight:700;
                        text-transform:uppercase;letter-spacing:2px;
                        color:{C['muted']};margin-bottom:12px;">
              ✦ Prediction Result
            </div>
            """, unsafe_allow_html=True)

            if pred:
                st.markdown(f"""
                <div class="result-card delayed-card">
                  <div class="result-label" style="color:{C['danger']};">Flight Status</div>
                  <div class="result-status delayed-text">🚨 Likely Delayed</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card ontime-card">
                  <div class="result-label" style="color:{C['success']};">Flight Status</div>
                  <div class="result-status ontime-text">✅ Expected On-Time</div>
                </div>
                """, unsafe_allow_html=True)

            # ── GAUGE + MAP ──
            col_g, col_m = st.columns([1, 1.6])

            with col_g:
                st.markdown(f"""
                <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                            letter-spacing:1.5px;color:{C['muted']};margin-bottom:6px;">
                  Delay Probability
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(make_gauge(prob), use_container_width=True, config={"displayModeBar": False})

                # Risk badge
                if prob < 0.3:
                    badge_col, badge_icon, badge_txt = C["success"], "🟢", "Low Risk"
                elif prob < 0.6:
                    badge_col, badge_icon, badge_txt = C["warning"], "🟡", "Moderate Risk"
                else:
                    badge_col, badge_icon, badge_txt = C["danger"], "🔴", "High Risk"

                st.markdown(f"""
                <div style="text-align:center;background:rgba(0,0,0,0.15);
                            border:1px solid {badge_col};border-radius:8px;
                            padding:8px 12px;color:{badge_col};
                            font-family:'Syne',sans-serif;font-weight:700;font-size:14px;">
                  {badge_icon} {badge_txt} — {prob:.1%}
                </div>
                """, unsafe_allow_html=True)

            with col_m:
                st.markdown(f"""
                <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                            letter-spacing:1.5px;color:{C['muted']};margin-bottom:6px;">
                  Route Map
                </div>
                """, unsafe_allow_html=True)
                route_fig = make_route_map(origin, dest)
                if route_fig:
                    st.plotly_chart(route_fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("Map unavailable for selected airports.")

            # ── FEATURE INSIGHTS ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                        letter-spacing:1.5px;color:{C['muted']};margin-bottom:12px;">
              ✦ Key Flight Factors
            </div>
            """, unsafe_allow_html=True)

            peak          = is_peak_hour(scheduled_departure)
            long_haul     = distance > 1500
            orig_row      = airports_df[airports_df.IATA_CODE == origin]
            dest_row      = airports_df[airports_df.IATA_CODE == dest]
            same_state    = (not orig_row.empty and not dest_row.empty and
                             orig_row.iloc[0].STATE == dest_row.iloc[0].STATE)
            red_eye       = scheduled_departure // 100 in range(0, 5)

            insights = [
                ("✈️", "Distance",   f"{distance:,} mi",          "long-haul" if long_haul else "short-haul"),
                ("⏰", "Departure",  f"{dep_fmt}",                 "🔴 Peak" if peak else "🟢 Off-peak"),
                ("📏", "Haul Type",  "Long-Haul" if long_haul else "Short-Haul", ""),
                ("🌙", "Red-Eye",    "Yes" if red_eye else "No",   ""),
                ("🗺️", "Same State", "Yes" if same_state else "No",""),
                ("🏢", "Airline",    airline,                      ""),
            ]

            cols_ins = st.columns(len(insights))
            for col_i, (icon, label, val, badge) in zip(cols_ins, insights):
                with col_i:
                    st.markdown(f"""
                    <div class="insight-card">
                      <div class="insight-icon">{icon}</div>
                      <div class="insight-label">{label}</div>
                      <div class="insight-value">{val}</div>
                      {"<div style='font-size:11px;color:" + (C['danger'] if '🔴' in badge else C['success']) + ";margin-top:4px;'>" + badge + "</div>" if badge else ""}
                    </div>
                    """, unsafe_allow_html=True)

            # ── WHY THIS PREDICTION ──
            st.markdown("<br>", unsafe_allow_html=True)

            reasons = []
            if peak:
                reasons.append("departing during peak travel hours (higher congestion risk)")
            if long_haul:
                reasons.append(f"covering a long-haul distance of {distance:,} miles (more exposure to compounding delays)")
            if not same_state:
                reasons.append("crossing state boundaries (greater air-traffic complexity)")
            if red_eye:
                reasons.append("scheduled as a red-eye departure (historically lower on-time rates)")
            if not reasons:
                reasons.append("operating under favorable scheduling conditions")

            confidence = "highly" if abs(prob - 0.5) > 0.25 else "moderately"
            verdict    = "likely to be delayed" if pred else "expected to arrive on-time"
            reason_str = "; ".join(reasons[:3])

            st.markdown(f"""
            <div class="why-box">
              <div class="why-title">✦ Why this prediction?</div>
              This flight is <strong>{confidence} {verdict}</strong>. Key contributing factors include {reason_str}.
              The model assigned a <strong>{prob:.1%} probability of delay</strong> against a threshold of {threshold:.1%}.
              {"Plan for potential delays and monitor the airline's status app." if pred else "Your flight looks good — minimal delay signals detected."}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
              ❌ <strong>Prediction failed.</strong> Please verify your inputs and try again.<br>
              <span style="font-size:12px;opacity:0.7;">Debug: {str(e)}</span>
            </div>
            """, unsafe_allow_html=True)

else:
    # Idle state — show placeholder
    st.markdown(f"""
    <div style="background:{C['surface']};border:1px dashed {C['border']};
                border-radius:14px;padding:48px 24px;text-align:center;margin-top:8px;">
      <div style="font-size:48px;margin-bottom:16px;">🛫</div>
      <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;
                  color:{C['text']};margin-bottom:8px;">
        Ready for Takeoff
      </div>
      <div style="font-size:14px;color:{C['muted']};max-width:340px;margin:0 auto;line-height:1.7;">
        Fill in your flight details in the sidebar, then hit
        <strong style="color:{C['accent']};">Predict Delay</strong>
        to see the ML analysis.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick-tip cards
    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    tips = [
        ("🎯", "High Accuracy", "Trained on millions of US domestic flights with calibrated probability outputs."),
        ("⚡", "Instant Analysis", "Feature engineering + AdaBoost inference in under a second."),
        ("🗺️", "Route Visualization", "See your route plotted on an interactive US map after prediction."),
    ]
    for col_t, (icon, title, desc) in zip([t1, t2, t3], tips):
        with col_t:
            st.markdown(f"""
            <div style="background:{C['surface']};border:1px solid {C['border']};
                        border-radius:12px;padding:20px;text-align:center;height:100%;">
              <div style="font-size:28px;margin-bottom:10px;">{icon}</div>
              <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
                          color:{C['text']};margin-bottom:6px;">{title}</div>
              <div style="font-size:12px;color:{C['muted']};line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <div class="footer-title">SkyPredict ✈️</div>
  <div class="footer-sub">ML-powered flight delay intelligence for smarter travel decisions</div>
  <div class="footer-stack">
    <span class="stack-tag">Python</span>
    <span class="stack-tag">Streamlit</span>
    <span class="stack-tag">AdaBoost</span>
    <span class="stack-tag">Plotly</span>
    <span class="stack-tag">scikit-learn</span>
    <span class="stack-tag">Pandas</span>
  </div>
  <div style="margin-top:12px;font-size:11px;color:{C['muted']};">
    Predictions are probabilistic estimates · Not a substitute for official airline information
  </div>
</div>
""", unsafe_allow_html=True)
