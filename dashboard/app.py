"""
Dashboard Interactivo — Agua Potable CDMX
SACMEX × IIMAS UNAM CU 2026-2 · Calidad y Preprocesamiento de Datos

Ejecutar:
    cd dashboard
    pip install -r requirements.txt
    python app.py
Abrir: http://localhost:8050
"""

import json
import warnings
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
ANALYSIS = ROOT / "datos" / "datos_analisis"
MASTERS  = ROOT / "datos" / "datos_maestros"
CLEAN    = ROOT / "datos" / "datos_limpios"

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _load():
    d = {}

    # dashboard_master
    try:
        df = pd.read_csv(ANALYSIS / "dashboard_master.csv", encoding="utf-8-sig")
        d["master"] = df
        print(f"  ✓ dashboard_master  {df.shape}")
    except Exception as e:
        print(f"  ⚠ dashboard_master: {e}")
        d["master"] = pd.DataFrame()

    # XGBoost predictions
    pred_p = ANALYSIS / "_resumen_q9_predicciones_xgb.csv"
    try:
        pred = pd.read_csv(pred_p, encoding="utf-8-sig")
        d["predictions"] = pred
        print(f"  ✓ predictions       {pred.shape}")
    except Exception:
        d["predictions"] = None
        print("  ⚠ predictions not found — using fallback")

    # Postal code map
    try:
        d["cp"] = pd.read_csv(MASTERS / "cp_a_colonia.csv", encoding="utf-8-sig")
        print(f"  ✓ cp_a_colonia      {d['cp'].shape}")
    except Exception:
        d["cp"] = pd.DataFrame()

    # GeoJSON (colony boundaries)
    gj_path = CLEAN / "colonias_iecm.geojson"
    try:
        with open(gj_path, encoding="utf-8") as f:
            gj = json.load(f)
        first_p = gj["features"][0].get("properties", {})
        d["geojson"] = gj
        d["gj_key"]  = "id_colonia" if "id_colonia" in first_p else "cveut"
        print(f"  ✓ geojson           key='{d['gj_key']}'  features={len(gj['features'])}")
    except Exception as e:
        d["geojson"] = None
        d["gj_key"]  = "id_colonia"
        print(f"  ⚠ geojson: {e}")

    return d


print("Cargando datos...")
DATA = _load()

# ── Merge predictions into master ────────────────────────────────────────────
df = DATA["master"].copy()

if DATA["predictions"] is not None and not df.empty:
    pred_cols = [c for c in ["id_colonia", "n_fugas_predicho", "riesgo_nivel"]
                 if c in DATA["predictions"].columns]
    df = df.merge(DATA["predictions"][pred_cols], on="id_colonia", how="left")

if not df.empty:
    if "n_fugas_predicho" not in df.columns:
        df["n_fugas_predicho"] = (df.get("fugas_por_10k_hab_total", pd.Series(0)).fillna(0) / 7).round(1)
    if "riesgo_nivel" not in df.columns:
        df["riesgo_nivel"] = pd.cut(
            df["n_fugas_predicho"],
            bins=[-1, 20, 80, 200, 9_999],
            labels=["Bajo", "Moderado", "Alto", "Crítico"],
        )
    df["IVH"] = pd.to_numeric(df.get("IVH", pd.Series(dtype=float)), errors="coerce")

print(f"  ✓ df_master final   {df.shape}")

# ── Computed globals ─────────────────────────────────────────────────────────
ALCALDIAS = sorted(df["nom_alcaldia"].dropna().unique()) if not df.empty else []

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CDMX_CENTER = {"lat": 19.43, "lon": -99.13}
MAP_STYLE   = "carto-darkmatter"

C = {  # color aliases
    "critico": "#d62728", "alto": "#ff7f0e", "medio": "#f0c33c", "bajo": "#2ca02c",
    "bg": "#090918", "card": "#111128", "border": "#23235a",
    "text": "#e0e0f8", "muted": "#7070a0", "accent": "#4878d0",
}

METRIC_META = {
    "n_fugas_predicho":   {"label": "Fugas predichas · XGBoost",    "cscale": "RdYlGn_r", "unit": "fugas"},
    "IVH":                {"label": "Índice de Vulnerabilidad Hídrica", "cscale": "RdYlGn_r", "unit": "0-1"},
    "estres_score":       {"label": "Estrés infraestructural",       "cscale": "YlOrRd",   "unit": "0-1"},
    "score_priorizacion": {"label": "Score de priorización",         "cscale": "RdYlGn_r", "unit": "0-1"},
    "fugas_por_10k_hab_total": {"label": "Fugas acumuladas / 10k hab", "cscale": "RdYlGn_r", "unit": "fugas"},
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ivh_tier(v):
    if pd.isna(v):  return "Sin dato", C["muted"]
    if v >= 0.70:   return "CRÍTICO",  C["critico"]
    if v >= 0.50:   return "ALTO",     C["alto"]
    if v >= 0.30:   return "MEDIO",    C["medio"]
    return "BAJO", C["bajo"]

def _fug_tier(v):
    if pd.isna(v): return "Sin dato", C["muted"]
    if v >= 500:   return "CRÍTICO",  C["critico"]
    if v >= 200:   return "ALTO",     C["alto"]
    if v >= 50:    return "MEDIO",    C["medio"]
    return "BAJO", C["bajo"]

def _cal_tier(v):
    if pd.isna(v): return "Sin dato", C["muted"]
    if v >= 50:    return "CRÍTICO",  C["critico"]
    if v >= 10:    return "ALTO",     C["alto"]
    if v > 0:      return "MEDIO",    C["medio"]
    return "BAJO", C["bajo"]

def _rl_color(level):
    return {"Crítico": C["critico"], "Alto": C["alto"],
            "Moderado": C["medio"], "Bajo": C["bajo"]}.get(str(level), C["muted"])

def _empty_fig(msg="Sin datos"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=.5, y=.5,
                       showarrow=False, font=dict(color=C["muted"], size=14))
    fig.update_layout(template="plotly_dark", paper_bgcolor=C["card"],
                      plot_bgcolor=C["card"], margin=dict(l=0,r=0,t=0,b=0))
    return fig

def _map_layout(fig, colorbar_title=""):
    fig.update_layout(
        margin=dict(r=0, t=0, l=0, b=0),
        paper_bgcolor=C["card"],
        coloraxis_colorbar=dict(
            title=colorbar_title, thickness=12, len=.6,
            bgcolor=C["card"],
            tickfont=dict(color=C["text"], size=9),
            title_font=dict(color=C["text"], size=10),
        ),
        legend=dict(bgcolor=C["card"], font=dict(color=C["text"])),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(title, value, sub="", color=None, icon=""):
    return dbc.Card(dbc.CardBody([
        html.Div(icon,  className="kpi-icon"),
        html.Div(title, className="kpi-title"),
        html.Div(value, className="kpi-value",
                 style={"color": color or C["accent"]}),
        html.Div(sub,   className="kpi-subtitle"),
    ]), className="kpi-card h-100")


def sem_circle(label, level, value_str, color):
    glow = f"0 0 20px {color}99"
    return html.Div([
        html.Div(style={
            "width": "70px", "height": "70px", "borderRadius": "50%",
            "background": f"radial-gradient(circle at 38% 32%, {color}cc, {color}66)",
            "boxShadow": glow, "margin": "0 auto",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
        }),
        html.Div(label,     className="sem-label"),
        html.Div(level,     className="sem-level", style={"color": color}),
        html.Div(value_str, className="sem-value"),
    ], className="sem-item")

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT — HEADER
# ─────────────────────────────────────────────────────────────────────────────
header = dbc.Navbar(dbc.Container([
    html.Div([
        html.Span("💧", style={"fontSize": "1.8rem", "marginRight": "12px"}),
        html.Div([
            html.H4("AGUA POTABLE · CDMX", className="nav-title"),
            html.Small(
                "Sistema de Monitoreo y Predicción de Riesgo Hídrico  ·  SACMEX × IIMAS UNAM",
                className="nav-subtitle"),
        ]),
    ], style={"display": "flex", "alignItems": "center"}),
    html.Div([
        html.Span("● EN VIVO",     className="badge-live"),
        html.Span("1,837 colonias", className="badge-count"),
        html.Span("DAMA-DMBOK",    className="badge-dama"),
    ], className="header-badges"),
], fluid=True), className="main-header", dark=True)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT — VISTA CIUDADANO
# ─────────────────────────────────────────────────────────────────────────────
tab_ciudadano = dbc.Container([

    # Search row
    dbc.Row(dbc.Col(html.Div([
        html.H5("🔍 Consulta el riesgo hídrico de tu colonia", className="search-title"),
        html.P("Ingresa tu código postal para ver el semáforo de calidad, fugas y vulnerabilidad.",
               className="search-sub"),
        dbc.Row([
            dbc.Col(dbc.Input(id="cp-input", type="text",
                              placeholder="Código postal · Ej: 09000, 06600, 14000...",
                              maxLength=5, className="cp-input"), width=8),
            dbc.Col(dbc.Button("CONSULTAR →", id="cp-btn", n_clicks=0,
                               className="cp-btn"), width=4),
        ], className="g-2"),
        html.Div(id="cp-error", className="cp-error"),
    ], className="search-box"), width=12), className="mb-3"),

    # Map + Semaphore
    dbc.Row([
        # ── MAP (left) ──────────────────────────────────────────────────────
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.Span("🗺️ Mapa de Vulnerabilidad Hídrica · CDMX", className="card-header-title"),
                dbc.Badge("IVH", color="danger", className="ms-2"),
                html.Span(" — Haz clic en una colonia para ver su detalle",
                          className="card-header-sub ms-2"),
            ], className="custom-card-header"),
            dbc.CardBody(dcc.Loading(
                dcc.Graph(id="cit-map", style={"height": "520px"},
                          config={"displayModeBar": True, "displaylogo": False,
                                  "modeBarButtonsToRemove": ["select2d", "lasso2d"]}),
                type="circle", color=C["accent"]),
            className="p-1"),
        ], className="dash-card"), width=7),

        # ── SEMAPHORE + KPIs (right) ────────────────────────────────────────
        dbc.Col([
            # Semaphore card
            dbc.Card([
                dbc.CardHeader("🚦 Semáforo de Riesgo Hídrico", className="custom-card-header"),
                dbc.CardBody([
                    html.Div(id="cit-colonia-name", className="sem-colonia-name",
                             children="↑ Busca tu código postal o haz clic en el mapa"),
                    html.Div(id="cit-sem", className="sem-container", children=[
                        sem_circle("Vulnerabilidad\nHídrica",   "—", "—", C["muted"]),
                        sem_circle("Calidad\ndel Agua",          "—", "—", C["muted"]),
                        sem_circle("Frecuencia\nde Fugas",       "—", "—", C["muted"]),
                    ]),
                    # Risk explanation text
                    html.Div(id="cit-risk-text", className="text-muted-sm px-3 pb-2"),
                ]),
            ], className="dash-card mb-3"),

            # KPI cards
            html.Div(id="cit-kpis", children=[
                dbc.Row([
                    dbc.Col(kpi_card("IVH", "—", "Vulnerabilidad hídrica", icon="📊"), width=6),
                    dbc.Col(kpi_card("Fugas / 10k hab", "—", "Tasa anual acumulada", icon="🔧"), width=6),
                ], className="g-2 mb-2"),
                dbc.Row([
                    dbc.Col(kpi_card("% Excede NOM-127", "—", "Calidad del agua", icon="⚗️"), width=6),
                    dbc.Col(kpi_card("Morbilidad / 100k", "—", "GI estimada", icon="🏥"), width=6),
                ], className="g-2 mb-2"),
                dbc.Row([
                    dbc.Col(kpi_card("% Pobreza", "—", "CONEVAL", icon="📉"), width=6),
                    dbc.Col(kpi_card("Ranking CDMX", "—", "de 1,837 colonias", icon="🏆"), width=6),
                ], className="g-2"),
            ]),
        ], width=5),
    ]),
], fluid=True, className="tab-content")

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT — VISTA GOBIERNO
# ─────────────────────────────────────────────────────────────────────────────
tab_gobierno = dbc.Container([

    # ── Filter row ───────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Label("Filtrar alcaldía", className="filter-label"),
            dcc.Dropdown(
                id="gov-alc",
                options=[{"label": "Todas las alcaldías", "value": "ALL"}] +
                        [{"label": a.title(), "value": a} for a in ALCALDIAS],
                value="ALL", clearable=False, className="custom-dropdown",
            ),
        ], width=4),
        dbc.Col([
            html.Label("Métrica del mapa", className="filter-label"),
            dcc.Dropdown(
                id="gov-metric",
                options=[
                    {"label": "🔮 Predicción XGBoost — fugas próx. 6 meses",  "value": "n_fugas_predicho"},
                    {"label": "📊 Índice de Vulnerabilidad Hídrica (IVH)",       "value": "IVH"},
                    {"label": "⚙️  Score de estrés infraestructural",            "value": "estres_score"},
                    {"label": "🎯 Score de priorización de intervención",         "value": "score_priorizacion"},
                    {"label": "🔧 Fugas acumuladas / 10k hab",                   "value": "fugas_por_10k_hab_total"},
                ],
                value="n_fugas_predicho", clearable=False, className="custom-dropdown",
            ),
        ], width=5),
        dbc.Col([
            dbc.Card(dbc.CardBody(html.Div([
                html.Span("🔴", style={"fontSize": "1.3rem"}),
                html.Div([
                    html.Div(id="gov-n-criticas", className="mini-kpi-value", children="—"),
                    html.Div("colonias en riesgo crítico", className="mini-kpi-label"),
                ]),
            ], style={"display": "flex", "alignItems": "center", "gap": "10px"})),
            className="mini-kpi-card"),
        ], width=3),
    ], className="mb-3 align-items-end"),

    # ── Map + Bar ─────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.Span("🗺️ Mapa Predictivo de Riesgo", className="card-header-title"),
                html.Span(" — resultados del modelo XGBoost", className="card-header-sub"),
            ], className="custom-card-header"),
            dbc.CardBody(dcc.Loading(
                dcc.Graph(id="gov-map", style={"height": "460px"},
                          config={"displayModeBar": True, "displaylogo": False}),
                type="circle", color=C["accent"]),
            className="p-1"),
        ], className="dash-card"), width=7),

        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.Span("🎯 Top 20 Colonias — Intervención Prioritaria", className="card-header-title"),
            ], className="custom-card-header"),
            dbc.CardBody(dcc.Loading(
                dcc.Graph(id="gov-bar", style={"height": "460px"},
                          config={"displayModeBar": False, "displaylogo": False}),
                type="circle", color=C["accent"]),
            className="p-1"),
        ], className="dash-card"), width=5),
    ], className="mb-3"),

    # ── Simulation + Distribution ─────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("⚡ Simulador de Impacto en Salud", className="custom-card-header"),
            dbc.CardBody([
                html.P("Mueve el slider para simular cuántas colonias intervenir "
                       "(ordenadas por score predictivo de mayor a menor riesgo):",
                       className="sim-description"),
                dcc.Slider(id="sim-slider", min=10, max=300, step=10, value=100,
                           marks={10: "10", 50: "50", 100: "100",
                                  150: "150", 200: "200", 300: "300"},
                           className="custom-slider",
                           tooltip={"always_visible": True, "placement": "bottom"}),
                html.Div(id="sim-kpis", className="sim-kpis mt-4"),
            ]),
        ], className="dash-card"), width=7),

        dbc.Col(dbc.Card([
            dbc.CardHeader("📊 Distribución de Niveles de Riesgo", className="custom-card-header"),
            dbc.CardBody(dcc.Loading(
                dcc.Graph(id="gov-pie", style={"height": "220px"},
                          config={"displayModeBar": False, "displaylogo": False}),
                type="circle", color=C["accent"]),
            className="p-1"),
        ], className="dash-card"), width=5),
    ]),
], fluid=True, className="tab-content")

# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    title="Agua Potable CDMX · Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server  # for deployment

app.layout = html.Div([
    header,
    dbc.Tabs([
        dbc.Tab(tab_ciudadano, label="👤  Vista Ciudadano",
                tab_id="tab-ciudadano", className="custom-tab"),
        dbc.Tab(tab_gobierno,  label="🏛️  Vista Gobierno",
                tab_id="tab-gobierno",  className="custom-tab"),
    ], id="main-tabs", active_tab="tab-ciudadano", className="main-tabs"),
], className="main-wrapper")

# ─────────────────────────────────────────────────────────────────────────────
# MAP BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def _choropleth(df_map, metric_col, highlight_ids=None, zoom=9.5, center=None):
    """Build a Choroplethmapbox from df_map."""
    if center is None:
        center = CDMX_CENTER
    geojson = DATA.get("geojson")
    gj_key  = DATA.get("gj_key", "id_colonia")
    meta    = METRIC_META.get(metric_col, {"label": metric_col, "cscale": "RdYlGn_r"})
    cscale  = meta["cscale"]

    hover_extra = {c: True for c in
                   ["nom_alcaldia", "IVH", "n_fugas_predicho", "riesgo_nivel",
                    "pct_excede_nom_local", "fugas_por_10k_hab_total"]
                   if c in df_map.columns and c != metric_col}
    hover_extra["id_colonia"] = False

    if geojson:
        fig = px.choropleth_mapbox(
            df_map, geojson=geojson,
            locations="id_colonia", featureidkey=f"properties.{gj_key}",
            color=metric_col, color_continuous_scale=cscale,
            hover_name="nom_colonia", hover_data=hover_extra,
            mapbox_style=MAP_STYLE, zoom=zoom, center=center,
            opacity=0.78,
            labels={metric_col: meta["label"], "nom_alcaldia": "Alcaldía",
                    "IVH": "IVH", "n_fugas_predicho": "Fugas pred.",
                    "riesgo_nivel": "Riesgo", "pct_excede_nom_local": "% excede NOM",
                    "fugas_por_10k_hab_total": "Fugas/10k"},
        )
    else:
        # Fallback scatter when GeoJSON is missing
        fig = px.scatter_mapbox(
            df_map, lat="centroide_lat", lon="centroide_lon",
            color=metric_col, color_continuous_scale=cscale,
            hover_name="nom_colonia", size_max=7, opacity=0.75,
            mapbox_style=MAP_STYLE, zoom=zoom, center=center,
        )

    # Overlay highlighted colonies
    if highlight_ids:
        hl = df_map[df_map["id_colonia"].isin(highlight_ids)].dropna(
            subset=["centroide_lat", "centroide_lon"])
        if not hl.empty:
            fig.add_trace(go.Scattermapbox(
                lat=hl["centroide_lat"], lon=hl["centroide_lon"],
                mode="markers+text",
                marker=dict(size=18, color="#FFD700", opacity=1,
                            symbol="circle"),
                text=hl["nom_colonia"].str[:16], textposition="top right",
                textfont=dict(size=10, color="#FFD700"),
                name="Tu colonia",
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))

    return _map_layout(fig, meta["label"])


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — VISTA CIUDADANO
# ─────────────────────────────────────────────────────────────────────────────

# Initial citizen map (full CDMX IVH on startup)
@app.callback(
    Output("cit-map", "figure"),
    Input("main-tabs", "active_tab"),
    prevent_initial_call=False,
)
def init_cit_map(_):
    if df.empty:
        return _empty_fig("Sin datos de dashboard_master.csv")
    return _choropleth(df, "IVH")


# Search callback
@app.callback(
    Output("cit-map",          "figure",   allow_duplicate=True),
    Output("cit-sem",          "children"),
    Output("cit-colonia-name", "children"),
    Output("cit-kpis",         "children"),
    Output("cit-risk-text",    "children"),
    Output("cp-error",         "children"),
    Input("cp-btn", "n_clicks"),
    Input("cp-input", "n_submit"),
    State("cp-input", "value"),
    prevent_initial_call=True,
)
def search_citizen(n_clicks, n_submit, cp_raw):
    if not cp_raw:
        return (no_update,) * 5 + ("Ingresa un código postal.",)

    cp_str = str(cp_raw).strip().zfill(5)
    cp_df  = DATA.get("cp", pd.DataFrame())

    # Find colony IDs for this CP
    if not cp_df.empty and "codigo_postal" in cp_df.columns:
        matches = cp_df[cp_df["codigo_postal"].astype(str).str.zfill(5) == cp_str]
        if matches.empty:
            return (no_update,) * 5 + (f"Código postal '{cp_str}' no encontrado.",)
        ids = matches["id_colonia"].tolist()
    else:
        ids = []

    # Get colony data
    if ids and not df.empty:
        sub = df[df["id_colonia"].isin(ids)]
        if sub.empty:
            return (no_update,) * 5 + ("No hay datos para ese código postal.",)
        row = sub.nlargest(1, "IVH").iloc[0] if "IVH" in sub else sub.iloc[0]
    else:
        return (no_update,) * 5 + ("No se encontraron datos.",)

    # ── MAP: zoom to colony ────────────────────────────────────────────────
    clat = row.get("centroide_lat", CDMX_CENTER["lat"])
    clon = row.get("centroide_lon", CDMX_CENTER["lon"])
    try:
        clat, clon = float(clat), float(clon)
    except Exception:
        clat, clon = CDMX_CENTER["lat"], CDMX_CENTER["lon"]

    fig_map = _choropleth(df, "IVH", highlight_ids=ids,
                          zoom=13.5, center={"lat": clat, "lon": clon})

    # ── SEMAPHORE ──────────────────────────────────────────────────────────
    ivh = row.get("IVH", np.nan)
    fug = row.get("fugas_por_10k_hab_total", np.nan)
    cal = row.get("pct_excede_nom_local", np.nan)
    try: ivh = float(ivh)
    except Exception: ivh = np.nan
    try: fug = float(fug)
    except Exception: fug = np.nan
    try: cal = float(cal)
    except Exception: cal = np.nan

    ivh_lbl, ivh_clr = _ivh_tier(ivh)
    fug_lbl, fug_clr = _fug_tier(fug)
    cal_lbl, cal_clr = _cal_tier(cal)

    sem = [
        sem_circle("Vulnerabilidad\nHídrica", ivh_lbl,
                   f"IVH {ivh:.3f}" if not np.isnan(ivh) else "—", ivh_clr),
        sem_circle("Calidad\ndel Agua", cal_lbl,
                   f"{cal:.0f}% fuera NOM" if not np.isnan(cal) else "—", cal_clr),
        sem_circle("Frecuencia\nde Fugas", fug_lbl,
                   f"{fug:.0f} /10k hab" if not np.isnan(fug) else "—", fug_clr),
    ]

    nom_col = str(row.get("nom_colonia", "—")).title()
    nom_alc = str(row.get("nom_alcaldia", "—")).title()
    col_name = f"📍  {nom_col}  ·  {nom_alc}"

    # ── KPI CARDS ──────────────────────────────────────────────────────────
    def _fmt(v, fmt=".2f", sfx=""):
        try:
            return f"{float(v):{fmt}}{sfx}" if not pd.isna(float(v)) else "—"
        except Exception:
            return "—"

    rank_v = row.get("rank", np.nan)
    try:
        rank_str = f"#{int(rank_v)}"
    except Exception:
        rank_str = "—"

    kpis = [
        dbc.Row([
            dbc.Col(kpi_card("IVH", _fmt(ivh), "Vulnerabilidad hídrica", ivh_clr, "📊"), width=6),
            dbc.Col(kpi_card("Fugas / 10k", _fmt(fug, ".0f"), "Acumuladas 2018-24", fug_clr, "🔧"), width=6),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col(kpi_card("% Excede NOM", _fmt(cal, ".0f", "%"), "Calidad agua", cal_clr, "⚗️"), width=6),
            dbc.Col(kpi_card("Morbilidad", _fmt(row.get("tasa_morbilidad_estimada_por_100k"), ",.0f"),
                             "/100k hab est.", C["accent"], "🏥"), width=6),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col(kpi_card("% Pobreza", _fmt(row.get("pobreza_pct_promedio"), ".1f", "%"),
                             "CONEVAL", C["muted"], "📉"), width=6),
            dbc.Col(kpi_card("Ranking CDMX", rank_str, "de 1,837 colonias", C["medio"], "🏆"), width=6),
        ], className="g-2"),
    ]

    # ── RISK TEXT ──────────────────────────────────────────────────────────
    rl = str(row.get("riesgo_nivel", "Moderado"))
    interv = str(row.get("intervencion", "Auditoría técnica recomendada"))
    risk_msg = f"Nivel de riesgo predictivo: {rl}  ·  Intervención sugerida: {interv}"

    return fig_map, sem, col_name, kpis, risk_msg, ""


# Click on map → fill postal code or show info
@app.callback(
    Output("cit-sem",          "children",  allow_duplicate=True),
    Output("cit-colonia-name", "children",  allow_duplicate=True),
    Output("cit-kpis",         "children",  allow_duplicate=True),
    Output("cit-risk-text",    "children",  allow_duplicate=True),
    Input("cit-map", "clickData"),
    prevent_initial_call=True,
)
def map_click_citizen(click_data):
    if not click_data or df.empty:
        return no_update, no_update, no_update, no_update
    try:
        id_col = click_data["points"][0].get("location")
    except Exception:
        return no_update, no_update, no_update, no_update
    if not id_col:
        return no_update, no_update, no_update, no_update

    sub = df[df["id_colonia"] == id_col]
    if sub.empty:
        return no_update, no_update, no_update, no_update
    row = sub.iloc[0]

    # reuse same logic as search callback
    ivh = row.get("IVH", np.nan)
    fug = row.get("fugas_por_10k_hab_total", np.nan)
    cal = row.get("pct_excede_nom_local", np.nan)
    for var_name in ["ivh", "fug", "cal"]:
        try:
            v = float(locals()[var_name])
            locals()[var_name]  # just access
        except Exception:
            pass

    try: ivh = float(ivh)
    except Exception: ivh = np.nan
    try: fug = float(fug)
    except Exception: fug = np.nan
    try: cal = float(cal)
    except Exception: cal = np.nan

    ivh_lbl, ivh_clr = _ivh_tier(ivh)
    fug_lbl, fug_clr = _fug_tier(fug)
    cal_lbl, cal_clr = _cal_tier(cal)

    sem = [
        sem_circle("Vulnerabilidad\nHídrica", ivh_lbl,
                   f"IVH {ivh:.3f}" if not np.isnan(ivh) else "—", ivh_clr),
        sem_circle("Calidad\ndel Agua", cal_lbl,
                   f"{cal:.0f}% fuera NOM" if not np.isnan(cal) else "—", cal_clr),
        sem_circle("Frecuencia\nde Fugas", fug_lbl,
                   f"{fug:.0f} /10k hab" if not np.isnan(fug) else "—", fug_clr),
    ]
    col_name = (f"📍  {str(row.get('nom_colonia','—')).title()}  ·  "
                f"{str(row.get('nom_alcaldia','—')).title()}")

    def _fmt(v, fmt=".2f", sfx=""):
        try: return f"{float(v):{fmt}}{sfx}" if not pd.isna(float(v)) else "—"
        except Exception: return "—"

    rank_v = row.get("rank", np.nan)
    try: rank_str = f"#{int(rank_v)}"
    except Exception: rank_str = "—"

    kpis = [
        dbc.Row([
            dbc.Col(kpi_card("IVH", _fmt(ivh), "Vulnerabilidad hídrica", ivh_clr, "📊"), width=6),
            dbc.Col(kpi_card("Fugas / 10k", _fmt(fug, ".0f"), "Acumuladas 2018-24", fug_clr, "🔧"), width=6),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col(kpi_card("% Excede NOM", _fmt(cal, ".0f", "%"), "Calidad agua", cal_clr, "⚗️"), width=6),
            dbc.Col(kpi_card("Morbilidad", _fmt(row.get("tasa_morbilidad_estimada_por_100k"), ",.0f"),
                             "/100k hab est.", C["accent"], "🏥"), width=6),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col(kpi_card("% Pobreza", _fmt(row.get("pobreza_pct_promedio"), ".1f", "%"),
                             "CONEVAL", C["muted"], "📉"), width=6),
            dbc.Col(kpi_card("Ranking CDMX", rank_str, "de 1,837 colonias", C["medio"], "🏆"), width=6),
        ], className="g-2"),
    ]

    rl = str(row.get("riesgo_nivel", "Moderado"))
    interv = str(row.get("intervencion", "Auditoría técnica recomendada"))
    risk_msg = f"Nivel de riesgo predictivo: {rl}  ·  Intervención sugerida: {interv}"
    return sem, col_name, kpis, risk_msg


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — VISTA GOBIERNO
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("gov-map",        "figure"),
    Output("gov-bar",        "figure"),
    Output("gov-pie",        "figure"),
    Output("gov-n-criticas", "children"),
    Input("gov-alc",    "value"),
    Input("gov-metric", "value"),
)
def update_gobierno(alc_filter, metric):
    if df.empty:
        ef = _empty_fig()
        return ef, ef, ef, "—"

    metric_col = metric if metric in df.columns else "IVH"
    df_g = df.copy()

    # Count colonias críticas (full CDMX, not filtered)
    n_crit = int((df_g["riesgo_nivel"] == "Crítico").sum()) \
             if "riesgo_nivel" in df_g.columns else int((df_g["IVH"] >= 0.70).sum())

    # Filter for bar chart only (map always shows full CDMX)
    if alc_filter and alc_filter != "ALL":
        df_g = df_g[df_g["nom_alcaldia"] == alc_filter]

    # ── MAP: full CDMX always ─────────────────────────────────────────────
    fig_map = _choropleth(df, metric_col)

    # If alcaldía selected, add a highlight scatter
    if alc_filter and alc_filter != "ALL":
        hl = df[df["nom_alcaldia"] == alc_filter].dropna(subset=["centroide_lat", "centroide_lon"])
        if not hl.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=hl["centroide_lat"], lon=hl["centroide_lon"], mode="markers",
                marker=dict(size=5, color=C["accent"], opacity=0.5),
                showlegend=False, hoverinfo="skip",
            ))

    # ── BAR CHART: Top 20 ──────────────────────────────────────────────────
    sort_col = (metric_col if metric_col in df_g.columns
                else ("n_fugas_predicho" if "n_fugas_predicho" in df_g.columns else "IVH"))
    top20 = df_g.nlargest(20, sort_col).sort_values(sort_col, ascending=True)
    meta  = METRIC_META.get(metric_col, {"label": metric_col, "unit": ""})

    bar_colors = [_rl_color(r.get("riesgo_nivel", "Moderado"))
                  for _, r in top20.iterrows()]

    y_text = [
        f"<b>{str(r['nom_colonia'])[:21]}</b>"
        f"<br><span style='font-size:9px;color:{C['muted']}'>{str(r['nom_alcaldia'])[:14].title()}</span>"
        for _, r in top20.iterrows()
    ]

    fig_bar = go.Figure(go.Bar(
        x=top20[sort_col].round(2),
        y=list(range(len(top20))),
        orientation="h",
        marker_color=bar_colors, marker_line_width=0,
        text=[f"  {v:.1f}" for v in top20[sort_col]],
        textposition="outside",
        textfont=dict(color=C["text"], size=8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            f"{meta['label']}: %{{x:.2f}}<br>"
            "Alcaldía: %{customdata[1]}<br>"
            "Riesgo: %{customdata[2]}<extra></extra>"
        ),
        customdata=list(zip(
            top20["nom_colonia"],
            top20["nom_alcaldia"],
            top20.get("riesgo_nivel", pd.Series(["—"] * len(top20))),
        )),
    ))
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        margin=dict(r=25, t=6, l=4, b=35),
        yaxis=dict(tickvals=list(range(len(top20))), ticktext=y_text,
                   tickfont=dict(size=8, color=C["text"]), showgrid=False),
        xaxis=dict(title=f"{meta['label']} ({meta.get('unit','')})",
                   titlefont=dict(color=C["text"], size=9),
                   tickfont=dict(color=C["muted"], size=8),
                   showgrid=True, gridcolor=C["border"]),
        showlegend=False,
    )

    # ── PIE: risk distribution ─────────────────────────────────────────────
    if "riesgo_nivel" in df_g.columns:
        dist = df_g["riesgo_nivel"].value_counts()
        labels_pie = dist.index.tolist()
        vals_pie   = dist.values.tolist()
        colors_pie = [_rl_color(l) for l in labels_pie]
    else:
        tiers = pd.cut(df_g["IVH"].dropna(),
                       bins=[-0.01, .30, .50, .70, 1.01],
                       labels=["Bajo", "Moderado", "Alto", "Crítico"])
        dist = tiers.value_counts()
        labels_pie = dist.index.tolist()
        vals_pie   = dist.values.tolist()
        colors_pie = [_rl_color(l) for l in labels_pie]

    total_g = len(df_g)
    fig_pie = go.Figure(go.Pie(
        labels=labels_pie, values=vals_pie, hole=0.55,
        marker_colors=colors_pie,
        textfont=dict(color=C["text"], size=10),
        hovertemplate="<b>%{label}</b><br>%{value} colonias · %{percent}<extra></extra>",
    ))
    fig_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        margin=dict(r=0, t=6, l=0, b=0),
        legend=dict(font=dict(color=C["text"], size=9), bgcolor=C["card"],
                    orientation="v", x=1.0),
        annotations=[dict(
            text=f"<b>{total_g:,}</b><br><span style='font-size:10px'>colonias</span>",
            font=dict(color=C["text"], size=12), showarrow=False,
        )],
    )

    return fig_map, fig_bar, fig_pie, f"{n_crit:,}"


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — SIMULADOR
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("sim-kpis", "children"),
    Input("sim-slider",  "value"),
    Input("gov-alc",     "value"),
)
def update_sim(n_colonias, alc_filter):
    if df.empty:
        return html.P("Sin datos.", style={"color": C["muted"]})

    df_s = df.copy()
    if alc_filter and alc_filter != "ALL":
        df_s = df_s[df_s["nom_alcaldia"] == alc_filter]

    sort_col = ("n_fugas_predicho" if "n_fugas_predicho" in df_s.columns
                else "score_priorizacion" if "score_priorizacion" in df_s.columns
                else "IVH")
    n_use  = min(n_colonias, len(df_s))
    top_n  = df_s.nlargest(n_use, sort_col)

    pop    = int(top_n["pob_colonia"].sum())           if "pob_colonia"              in top_n.columns else 0
    casos  = int(top_n["casos_morbilidad_estimados"].sum() * 0.30) \
             if "casos_morbilidad_estimados" in top_n.columns else 0
    fugas  = int(top_n["n_fugas_predicho"].sum())      if "n_fugas_predicho"         in top_n.columns else 0
    ninos  = int(top_n["pob_menores_estim"].sum())     if "pob_menores_estim"        in top_n.columns else 0

    pct_cdmx = pop / df["pob_colonia"].sum() * 100 if "pob_colonia" in df.columns and df["pob_colonia"].sum() > 0 else 0

    return dbc.Row([
        dbc.Col(kpi_card("Población protegida",
                         f"{pop:,}", f"≈ {pct_cdmx:.1f}% de CDMX", C["bajo"], "👥"), width=3),
        dbc.Col(kpi_card("Casos GI evitables / año",
                         f"{casos:,}", "Reducción 30% estimada", C["accent"], "🏥"), width=3),
        dbc.Col(kpi_card("Fugas prevenibles",
                         f"{fugas:,}", "Predicción XGBoost 6 meses", C["medio"], "🔧"), width=3),
        dbc.Col(kpi_card("Niños protegidos",
                         f"{ninos:,}", "Menores de 14 años", C["critico"], "👶"), width=3),
    ], className="g-2")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  💧 AGUA POTABLE CDMX · Dashboard SACMEX")
    print("     http://localhost:8050")
    print("═" * 60 + "\n")
    app.run(debug=False, port=8050, host="0.0.0.0")
