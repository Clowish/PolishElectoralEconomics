"""
Interactive Dash explorer for the Polish electoral economics panel.

Usage:
    uv run python3 -m src.app.app
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html
from plotly.subplots import make_subplots
from shapely.geometry import shape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "panel_final_v2.parquet"
GEOJSON_PATH = PROJECT_ROOT / "data" / "raw" / "spatial" / "gminy.geojson"

ELECTION_YEARS = [2001, 2005, 2007, 2011, 2015, 2019, 2023]
DEFAULT_YEAR = 2019
DEFAULT_VARIABLE = "pct_populist_right"

SPECIAL_TERYT6 = {
    "001498",
    "001499",
    "002298",
    "002299",
    "003299",
    # Current panel stores the same special election units with full pseudo-gmina codes.
    "149801",
    "149901",
    "149999",
    "229801",
    "229901",
    "229997",
    "329901",
    "329997",
}

TYPE_LABELS = {
    1: "Miejska",
    2: "Wiejska",
    3: "Miejsko-wiejska",
}
TYPE_OPTIONS = ["Miejska", "Wiejska", "Miejsko-wiejska", "Nieokreślony"]

VARIABLES = {
    "pct_populist_right": {
        "label": "Populistyczna prawica (%)",
        "description": "Udział głosów na partie zaklasyfikowane jako populistyczna lub antysystemowa prawica. Wyższa wartość oznacza większe lokalne poparcie dla tego bloku.",
        "range": (0, 65),
        "colorscale": px.colors.sequential.YlOrRd,
        "center": None,
    },
    "pct_mainstream_right": {
        "label": "Główny nurt prawicy (%)",
        "description": "Udział głosów na partie głównego nurtu prawicy i centrum-prawicy. Uwaga: KO, Trzecia Droga i podobne komitety są w tej klasyfikacji w liberalnym centrum, więc ta seria jest po 2015 roku mała.",
        "range": (0, 65),
        "colorscale": px.colors.sequential.Blues,
        "center": None,
    },
    "pct_post_communist_left": {
        "label": "Lewica postkomunistyczna (%)",
        "description": "Udział głosów na SLD/Lewicę i historycznie pokrewne komitety. Pomaga zobaczyć długookresowe przesunięcia poza prawicą.",
        "range": (0, 50),
        "colorscale": px.colors.sequential.Purples,
        "center": None,
    },
    "pct_liberal_center": {
        "label": "Liberalne centrum (%)",
        "description": "Udział głosów na KO/PO, Trzecią Drogę, Nowoczesną i podobne komitety liberalno-centrowe. To brakujący element, który w wielu gminach domyka obraz 2023 roku.",
        "range": (0, 70),
        "colorscale": px.colors.sequential.Teal,
        "center": None,
    },
    "dochody_pc_relative": {
        "label": "Dochody budżetowe rel. (kraj=1)",
        "description": "Dochody budżetu gminy per capita jako relacja do średniej krajowej ważonej ludnością. Wartość 1 oznacza przeciętną pozycję w kraju.",
        "range": (0.3, 2.5),
        "colorscale": px.colors.diverging.RdBu,
        "center": 1.0,
    },
    "pit_pc_relative": {
        "label": "PIT per capita rel. (kraj=1)",
        "description": "Udziały gminy w PIT/CIT per capita względem średniej krajowej. Bliżej lokalnego rynku pracy niż dochody budżetowe ogółem.",
        "range": (0.3, 2.5),
        "colorscale": px.colors.diverging.RdBu,
        "center": 1.0,
    },
    "bezrobocie_powiat_relative": {
        "label": "Bezrobocie powiatowe rel. (kraj=1)",
        "description": "Stopa bezrobocia rejestrowanego w powiecie względem średniej krajowej. Wartości powyżej 1 oznaczają gorszą sytuację niż przeciętnie.",
        "range": (0.3, 3.0),
        "colorscale": list(reversed(px.colors.diverging.RdYlGn)),
        "center": 1.0,
    },
    "wynagrodzenie_powiat_relative": {
        "label": "Wynagrodzenie powiatowe rel. (kraj=1)",
        "description": "Przeciętne miesięczne wynagrodzenie brutto w powiecie względem średniej krajowej. Wartości powyżej 1 oznaczają wyższe płace niż przeciętnie.",
        "range": (0.5, 1.8),
        "colorscale": px.colors.diverging.RdBu,
        "center": 1.0,
    },
    "delta_pct_populist_right": {
        "label": "Zmiana: pop. prawica (pp)",
        "description": "Zmiana udziału głosów na populistyczną prawicę względem poprzednich wyborów, w punktach procentowych.",
        "range": (-40, 40),
        "colorscale": list(reversed(px.colors.diverging.RdBu)),
        "center": 0.0,
    },
    "delta_dochody_pc_relative": {
        "label": "Zmiana: dochody rel.",
        "description": "Zmiana relatywnej pozycji dochodowej gminy względem poprzednich wyborów. Wartości ujemne oznaczają spadek względem średniej krajowej.",
        "range": (-0.4, 0.4),
        "colorscale": px.colors.diverging.RdBu,
        "center": 0.0,
    },
}

ELECTION_SERIES = [
    ("pct_populist_right", "Populistyczna prawica", "#c44e52"),
    ("pct_liberal_center", "Liberalne centrum", "#2a9d8f"),
    ("pct_post_communist_left", "Lewica postkom.", "#8172b2"),
    ("pct_mainstream_right", "Główny nurt prawicy", "#4c72b0"),
    ("pct_other", "Inne", "#94a3b8"),
]


def type_label(value: object) -> str:
    """Return display label for typ_gminy, treating missing values separately."""

    if pd.isna(value):
        return "Nieokreślony"
    return TYPE_LABELS.get(int(value), "Nieokreślony")


def load_panel() -> pd.DataFrame:
    """Load panel once at app startup."""

    panel = pd.read_parquet(PANEL_PATH)
    panel["teryt6"] = panel["teryt6"].astype(str).str.zfill(6)
    panel = panel.loc[~panel["teryt6"].isin(SPECIAL_TERYT6)].copy()
    panel["typ_label"] = panel["typ_gminy"].map(type_label)
    return panel


def load_geojson() -> tuple[dict, set[str], dict[str, dict[str, float]]]:
    """Load and normalize GeoJSON once at app startup."""

    with GEOJSON_PATH.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = []
    geo_teryt: set[str] = set()
    centroids: dict[str, dict[str, float]] = {}
    for feature in geojson.get("features", []):
        props = feature.setdefault("properties", {})
        teryt = str(props.get("teryt6", "")).strip()
        if teryt.isdigit():
            teryt = teryt.zfill(6)
            props["teryt6"] = teryt
            features.append(feature)
            geo_teryt.add(teryt)
            centroid = shape(feature["geometry"]).centroid
            centroids[teryt] = {"lon": centroid.x, "lat": centroid.y}

    geojson["features"] = features
    return geojson, geo_teryt, centroids


PANEL = load_panel()
GEOJSON, GEO_TERYT, GEO_CENTROIDS = load_geojson()
MAP_PANEL = PANEL.loc[PANEL["teryt6"].isin(GEO_TERYT)].copy()
PANEL_TERYT = set(PANEL["teryt6"])
MUNICIPALITY_OPTIONS = (
    MAP_PANEL[["teryt6", "gmina_nazwa", "typ_label"]]
    .drop_duplicates("teryt6")
    .sort_values(["gmina_nazwa", "typ_label"])
    .assign(label=lambda d: d["gmina_nazwa"].astype(str) + " · " + d["typ_label"].astype(str))
)

PANEL_WITHOUT_GEOMETRY = sorted(PANEL_TERYT - GEO_TERYT)
GEOMETRY_WITHOUT_PANEL = sorted(GEO_TERYT - PANEL_TERYT)
DEFAULT_MAP_COUNT = int(
    MAP_PANEL.loc[
        (MAP_PANEL["year"] == DEFAULT_YEAR)
        & (MAP_PANEL["typ_label"].isin(TYPE_OPTIONS))
        & (MAP_PANEL[DEFAULT_VARIABLE].notna())
    ]["teryt6"].nunique()
)

print(
    "GeoJSON merge diagnostic: "
    f"panel_without_geometry={len(PANEL_WITHOUT_GEOMETRY)}, "
    f"geometry_without_panel={len(GEOMETRY_WITHOUT_PANEL)}, "
    f"default_map_count={DEFAULT_MAP_COUNT}"
)

app = Dash(__name__)
app.title = "Polish Electoral Economics — Explorer"


BASE_CARD_STYLE = {
    "background": "#fffaf0",
    "border": "1px solid rgba(23, 32, 42, 0.10)",
    "borderRadius": "22px",
    "boxShadow": "0 18px 50px rgba(23, 32, 42, 0.12)",
}

PAGE_BG = "#f0ede6"
TEXT = "#111827"
MUTED = "#64748b"
ACCENT = "#d97706"
DARK = "#17202a"
PAPER = "#fffaf0"

LABEL_STYLE = {
    "display": "block",
    "fontSize": "12px",
    "fontWeight": "700",
    "letterSpacing": "0.02em",
    "textTransform": "uppercase",
    "color": "#a7b3c2",
    "margin": "14px 0 7px",
}

CONTROL_STYLE = {"fontSize": "14px"}


def empty_map(message: str) -> go.Figure:
    """Return an empty map figure with a centered message."""

    fig = go.Figure()
    fig.update_layout(
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
        margin={"r": 0, "t": 10, "l": 0, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def filtered_year_frame(year: int, selected_types: list[str] | None) -> pd.DataFrame:
    """Filter map/ranking frame by election year and municipality type."""

    selected = selected_types or []
    return MAP_PANEL.loc[
        (MAP_PANEL["year"] == year)
        & (MAP_PANEL["typ_label"].isin(selected))
    ].copy()


def make_map_figure(
    year: int,
    variable: str,
    selected_types: list[str] | None,
    selected_teryt6: str | None,
    map_view: str = "poland",
) -> go.Figure:
    """Build the Plotly choropleth map."""

    config = VARIABLES[variable]
    frame = filtered_year_frame(year, selected_types)
    if frame.empty:
        return empty_map("Brak gmin dla wybranego filtra.")

    frame = frame.sort_values("teryt6")
    all_locations = frame["teryt6"].tolist()
    data_frame = frame.loc[frame[variable].notna()].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            geojson=GEOJSON,
            locations=all_locations,
            z=[0] * len(all_locations),
            featureidkey="properties.teryt6",
            colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
            showscale=False,
            hoverinfo="skip",
            marker_line_width=0,
        )
    )

    customdata = np.stack(
        [
            data_frame["gmina_nazwa"].astype(str),
            data_frame["typ_label"].astype(str),
        ],
        axis=-1,
    ) if not data_frame.empty else None

    fig.add_trace(
        go.Choropleth(
            geojson=GEOJSON,
            locations=data_frame["teryt6"],
            z=data_frame[variable],
            featureidkey="properties.teryt6",
            colorscale=config["colorscale"],
            zmin=config["range"][0],
            zmax=config["range"][1],
            zmid=config["center"],
            colorbar={
                "title": {"text": config["label"], "font": {"size": 12}},
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": -0.025,
                "len": 0.7,
                "thickness": 10,
                "tickfont": {"size": 11},
            },
            customdata=customdata,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
            + config["label"]
            + ": %{z:.2f}<extra></extra>",
            marker_line_width=0,
        )
    )

    if selected_teryt6 and selected_teryt6 in set(all_locations):
        fig.add_trace(
            go.Choropleth(
                geojson=GEOJSON,
                locations=[selected_teryt6],
                z=[1],
                featureidkey="properties.teryt6",
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                hoverinfo="skip",
                marker_line_color="black",
                marker_line_width=3,
            )
        )

    geo_layout = {
        "visible": False,
        "bgcolor": PAPER,
        "projection_type": "mercator",
        "domain": {"x": [0, 1], "y": [0.08, 0.98]},
    }
    if map_view == "focus" and selected_teryt6 in GEO_CENTROIDS:
        geo_layout.update(
            {
                "center": GEO_CENTROIDS[selected_teryt6],
                "projection_scale": 48,
            }
        )
    else:
        geo_layout["fitbounds"] = "locations"

    fig.update_geos(**geo_layout)
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 20},
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        clickmode="event+select",
        dragmode="pan",
        height=820,
    )
    return fig


def make_ranking(year: int, variable: str, selected_types: list[str] | None) -> list[dict]:
    """Return top-20 municipalities for the ranking table."""

    frame = filtered_year_frame(year, selected_types)
    ranking = (
        frame.loc[frame[variable].notna(), ["teryt6", "gmina_nazwa", "typ_label", variable]]
        .sort_values(variable, ascending=False)
        .head(20)
        .copy()
    )
    ranking["value_display"] = ranking[variable].map(lambda x: f"{x:.2f}")
    ranking["id"] = ranking["teryt6"]
    return ranking[["id", "gmina_nazwa", "value_display", "typ_label"]].to_dict("records")


def line_figure(title: str, y_title: str) -> go.Figure:
    """Shared styling for compact line charts in the right panel."""

    fig = go.Figure()
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 14, "color": TEXT}},
        height=205,
        margin={"l": 34, "r": 12, "t": 34, "b": 42},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0, "font": {"size": 10}},
        xaxis_title="Rok",
        yaxis_title=y_title,
        font={"family": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", "size": 11},
    )
    fig.update_xaxes(showgrid=False, tickmode="array", tickvals=ELECTION_YEARS)
    fig.update_yaxes(gridcolor="#edf2f7", zeroline=False)
    return fig


def selected_panel(selected_teryt6: str | None) -> html.Div:
    """Build the municipality detail panel."""

    if not selected_teryt6:
        return html.Div(
            [
                html.Div("Panel gminy", style={"fontSize": "12px", "fontWeight": "700", "color": MUTED, "textTransform": "uppercase"}),
                html.Div("Kliknij gminę na mapie albo w rankingu.", style={"fontSize": "18px", "fontWeight": "650", "marginTop": "14px", "color": TEXT}),
                html.Div("Po zaznaczeniu zobaczysz trajektorie wyborcze, ekonomiczne i tabelę 7 fal wyborczych.", style={"fontSize": "13px", "lineHeight": "1.45", "marginTop": "8px", "color": MUTED}),
            ]
        )

    muni = PANEL.loc[PANEL["teryt6"] == selected_teryt6].sort_values("year")
    if muni.empty:
        return html.Div("Nie znaleziono gminy w panelu.", style={"color": "#b91c1c"})

    name = str(muni["gmina_nazwa"].dropna().iloc[-1])
    typ = str(muni["typ_label"].dropna().iloc[-1])

    election_fig = line_figure("Kompozycja wyborcza", "% głosów")
    for column, label, color in ELECTION_SERIES:
        if column in muni.columns:
            election_fig.add_trace(
                go.Scatter(
                    x=muni["year"],
                    y=muni[column],
                    mode="lines",
                    name=label,
                    stackgroup="votes",
                    line={"color": color, "width": 1.4},
                    hovertemplate=f"{label}<br>Rok: %{{x}}<br>Udział: %{{y:.2f}}%<extra></extra>",
                )
            )
    election_fig.update_yaxes(range=[0, 100], ticksuffix="%")
    election_fig.update_layout(
        height=235,
        margin={"l": 34, "r": 12, "t": 34, "b": 56},
        legend={"orientation": "h", "yanchor": "top", "y": -0.24, "x": 0, "font": {"size": 10}},
    )

    econ_fig = make_subplots(specs=[[{"secondary_y": True}]])
    econ_fig.add_trace(go.Scatter(x=muni["year"], y=muni["dochody_pc_relative"], mode="lines+markers", name="Dochody", line={"color": "#55a868", "width": 2.4}, marker={"size": 6}), secondary_y=False)
    econ_fig.add_trace(go.Scatter(x=muni["year"], y=muni["wynagrodzenie_powiat_relative"], mode="lines+markers", name="Wynagrodz.", line={"color": "#7f7f7f", "width": 2.4}, marker={"size": 6}), secondary_y=False)
    econ_fig.add_trace(go.Scatter(x=muni["year"], y=muni["bezrobocie_powiat_relative"], mode="lines+markers", name="Bezrobocie", line={"color": "#dd8452", "width": 2.4}, marker={"size": 6}), secondary_y=True)
    econ_fig.add_hline(y=1.0, line_dash="dash", line_color="#475569", line_width=1)
    econ_fig.update_layout(
        title={"text": "Trajektoria ekonomiczna", "x": 0, "font": {"size": 14, "color": TEXT}},
        height=220,
        margin={"l": 34, "r": 42, "t": 34, "b": 46},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0, "font": {"size": 10}},
        xaxis_title="Rok",
        font={"family": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", "size": 11},
    )
    econ_fig.update_xaxes(showgrid=False, tickmode="array", tickvals=ELECTION_YEARS)
    econ_fig.update_yaxes(title_text="Indeks", gridcolor="#edf2f7", zeroline=False, secondary_y=False)
    econ_fig.update_yaxes(title_text="Bezrob.", autorange="reversed", gridcolor="#f8fafc", secondary_y=True)

    table_cols = [
        "year",
        "pct_populist_right",
        "pct_liberal_center",
        "dochody_pc_relative",
        "bezrobocie_powiat_relative",
        "wynagrodzenie_powiat_relative",
        "frekwencja",
    ]
    table_data = muni[table_cols].round(2).to_dict("records")

    return html.Div(
        [
            html.Div("Panel gminy", style={"fontSize": "12px", "fontWeight": "700", "color": MUTED, "textTransform": "uppercase"}),
            html.H2(f"{name} ({typ.lower()})", style={"margin": "4px 0 12px 0", "fontSize": "22px", "lineHeight": "1.15", "color": TEXT}),
            dcc.Graph(figure=election_fig, config={"displayModeBar": False}, style={"height": "235px"}),
            html.Div(
                "Wykres jest skumulowany do 100%: liberalne centrum domyka udział KO/Trzeciej Drogi i wyjaśnia pozornie niskie sumy trzech wcześniejszych serii.",
                style={"fontSize": "11px", "lineHeight": "1.35", "color": MUTED, "margin": "-4px 0 6px"},
            ),
            dcc.Graph(figure=econ_fig, config={"displayModeBar": False}, style={"height": "220px", "marginTop": "4px"}),
            html.Div("Tabela wartości", style={"fontSize": "12px", "fontWeight": "700", "color": MUTED, "textTransform": "uppercase", "margin": "10px 0 6px"}),
            dash_table.DataTable(
                data=table_data,
                columns=[
                    {"name": "Rok", "id": "year"},
                    {"name": "Pop. prawica", "id": "pct_populist_right"},
                    {"name": "Lib. centrum", "id": "pct_liberal_center"},
                    {"name": "Dochody", "id": "dochody_pc_relative"},
                    {"name": "Bezrob.", "id": "bezrobocie_powiat_relative"},
                    {"name": "Wynagrodz.", "id": "wynagrodzenie_powiat_relative"},
                    {"name": "Frekw.", "id": "frekwencja"},
                ],
                page_action="none",
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "system-ui", "fontSize": "11px", "padding": "6px 5px", "textAlign": "right", "border": "0"},
                style_cell_conditional=[{"if": {"column_id": "year"}, "textAlign": "left", "fontWeight": "700"}],
                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc", "border": "0", "color": MUTED},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fbfdff"}],
            ),
        ]
    )


app.layout = html.Div(
    [
        dcc.Store(id="selected-teryt6"),
        html.Div(
            [
                html.Div(
                    "Polish Electoral Economics",
                    style={"fontSize": "12px", "fontWeight": "800", "color": "#fbbf24", "textTransform": "uppercase", "letterSpacing": "0.08em"},
                ),
                html.H1(
                    "Atlas gmin",
                    style={"fontSize": "31px", "lineHeight": "0.98", "margin": "5px 0 8px", "color": "white", "fontWeight": "850"},
                ),
                html.Div(
                    "Mapa, ranking i trajektorie dla panelu wyborczo-ekonomicznego 2001–2023.",
                    style={"fontSize": "13px", "lineHeight": "1.45", "color": "#cbd5e1", "marginBottom": "12px"},
                ),
                html.Label("Znajdź gminę", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="municipality-search",
                    options=[
                        {"label": row.label, "value": row.teryt6}
                        for row in MUNICIPALITY_OPTIONS.itertuples(index=False)
                    ],
                    value=None,
                    clearable=True,
                    placeholder="Wpisz nazwę gminy...",
                    style=CONTROL_STYLE,
                ),
                html.Label("Rok wyborów", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[{"label": str(year), "value": year} for year in ELECTION_YEARS],
                    value=DEFAULT_YEAR,
                    clearable=False,
                    style=CONTROL_STYLE,
                ),
                html.Label("Zmienna na mapie", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="variable-dropdown",
                    options=[{"label": spec["label"], "value": col} for col, spec in VARIABLES.items()],
                    value=DEFAULT_VARIABLE,
                    clearable=False,
                    style=CONTROL_STYLE,
                ),
                html.Div(id="variable-help", style={
                    "background": "rgba(255,255,255,0.08)",
                    "border": "1px solid rgba(255,255,255,0.10)",
                    "borderRadius": "14px",
                    "padding": "10px 11px",
                    "fontSize": "12px",
                    "lineHeight": "1.45",
                    "color": "#dbeafe",
                    "marginTop": "9px",
                }),
                html.Label("Typ gminy", style=LABEL_STYLE),
                dcc.Checklist(
                    id="type-checklist",
                    options=[{"label": label, "value": label} for label in TYPE_OPTIONS],
                    value=TYPE_OPTIONS,
                    inputStyle={"marginRight": "7px", "accentColor": ACCENT},
                    labelStyle={"display": "block", "margin": "5px 0", "fontSize": "13px", "color": "#e5e7eb"},
                    style={"marginBottom": "14px"},
                ),
                html.Label("Widok mapy", style=LABEL_STYLE),
                dcc.RadioItems(
                    id="map-view",
                    options=[
                        {"label": "Cała Polska", "value": "poland"},
                        {"label": "Przybliż zaznaczoną", "value": "focus"},
                    ],
                    value="poland",
                    inputStyle={"marginRight": "7px", "accentColor": ACCENT},
                    labelStyle={"display": "block", "margin": "5px 0", "fontSize": "13px", "color": "#e5e7eb"},
                ),
                html.Div(
                    [
                        html.Div("Ranking", style={"fontSize": "16px", "fontWeight": "800", "color": "white"}),
                        html.Div("Top 20", style={"fontSize": "12px", "color": "#cbd5e1"}),
                    ],
                    style={"display": "flex", "alignItems": "baseline", "justifyContent": "space-between", "margin": "8px 0 8px"},
                ),
                dash_table.DataTable(
                    id="ranking-table",
                    data=[],
                    columns=[],
                    page_action="none",
                    cell_selectable=True,
                    style_table={"height": "34vh", "overflowY": "auto", "borderRadius": "14px", "border": "1px solid rgba(255,255,255,0.15)"},
                    style_cell={"fontFamily": "system-ui", "fontSize": "12px", "padding": "8px 9px", "textAlign": "left", "border": "0", "whiteSpace": "normal", "height": "auto", "backgroundColor": "#fffaf0"},
                    style_cell_conditional=[
                        {"if": {"column_id": "value_display"}, "textAlign": "right", "fontWeight": "700", "color": TEXT},
                        {"if": {"column_id": "typ_label"}, "color": MUTED, "fontSize": "11px"},
                    ],
                    style_header={"fontWeight": "800", "backgroundColor": "#f8fafc", "border": "0", "color": MUTED},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#fbfdff"},
                        {"if": {"state": "active"}, "backgroundColor": "#ccfbf1", "border": f"1px solid {ACCENT}"},
                    ],
                ),
            ],
            style={
                **BASE_CARD_STYLE,
                "background": f"linear-gradient(180deg, {DARK} 0%, #243447 100%)",
                "width": "24%",
                "height": "calc(100vh - 24px)",
                "overflow": "hidden",
                "padding": "18px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Mapa choropleth", style={"fontSize": "13px", "fontWeight": "800", "color": MUTED, "textTransform": "uppercase"}),
                        html.Div("Kliknij gminę, żeby zobaczyć pełną trajektorię.", style={"fontSize": "13px", "color": MUTED}),
                    ],
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "padding": "13px 16px 0"},
                ),
                dcc.Graph(
                    id="map-graph",
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "scrollZoom": True,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    },
                    style={"height": "calc(100vh - 74px)", "width": "100%"},
                )
            ],
            style={**BASE_CARD_STYLE, "width": "49%", "height": "calc(100vh - 24px)", "padding": "0", "overflow": "hidden"},
        ),
        html.Div(
            id="municipality-panel",
            style={**BASE_CARD_STYLE, "width": "28%", "height": "calc(100vh - 24px)", "overflowY": "auto", "padding": "18px"},
        ),
    ],
    style={
        "height": "100vh",
        "display": "flex",
        "gap": "12px",
        "padding": "12px",
        "boxSizing": "border-box",
        "background": PAGE_BG,
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "overflow": "hidden",
        "color": TEXT,
    },
)


@app.callback(
    Output("selected-teryt6", "data"),
    Input("municipality-search", "value"),
    Input("map-graph", "clickData"),
    Input("ranking-table", "active_cell"),
    State("ranking-table", "data"),
    prevent_initial_call=True,
)
def update_selection(
    search_value: str | None,
    click_data: dict | None,
    active_cell: dict | None,
    ranking_data: list[dict] | None,
) -> str | None:
    """Update selected municipality from either map click or ranking row click."""

    triggered = callback_context.triggered_id
    if triggered == "municipality-search":
        return str(search_value) if search_value else None

    if triggered == "map-graph" and click_data:
        points = click_data.get("points", [])
        if points:
            return str(points[0].get("location"))

    if triggered == "ranking-table" and active_cell is not None and ranking_data:
        row_idx = active_cell.get("row")
        if row_idx is not None and 0 <= row_idx < len(ranking_data):
            return str(ranking_data[row_idx].get("id"))

    return None


@app.callback(
    Output("map-graph", "figure"),
    Output("ranking-table", "data"),
    Output("ranking-table", "columns"),
    Output("variable-help", "children"),
    Input("year-dropdown", "value"),
    Input("variable-dropdown", "value"),
    Input("type-checklist", "value"),
    Input("map-view", "value"),
    Input("selected-teryt6", "data"),
)
def update_map_and_ranking(
    year: int,
    variable: str,
    selected_types: list[str] | None,
    map_view: str,
    selected_teryt6: str | None,
) -> tuple[go.Figure, list[dict], list[dict], html.Div]:
    """Update choropleth map and top-20 ranking."""

    figure = make_map_figure(year, variable, selected_types, selected_teryt6, map_view)
    ranking = make_ranking(year, variable, selected_types)
    columns = [
        {"name": "Gmina", "id": "gmina_nazwa"},
        {"name": "Wartość", "id": "value_display"},
        {"name": "Typ", "id": "typ_label"},
    ]
    help_box = html.Div(
        [
            html.Div(VARIABLES[variable]["label"], style={"fontWeight": "800", "marginBottom": "4px", "color": "white"}),
            html.Div(VARIABLES[variable]["description"]),
        ]
    )
    return figure, ranking, columns, help_box


@app.callback(
    Output("municipality-panel", "children"),
    Input("selected-teryt6", "data"),
)
def update_municipality_panel(selected_teryt6: str | None) -> html.Div:
    """Update the right-side detail panel."""

    return selected_panel(selected_teryt6)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True, use_reloader=False)
