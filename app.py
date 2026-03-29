"""
app.py — Room Acoustics V2 — Éditeur de plan + visualisation
"""

import json
import os
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
from flask import request as flask_request, jsonify
import plotly.graph_objects as go
import numpy as np

from acoustics import (compute_modes, pressure_field_2d, pressure_field_3d,
                       speaker_coupling, room_mode_frequency)
from analysis import analyse_room
from room_editor import make_grid_figure, polygon_bounding_box, polygon_area, is_complex_shape
from polygon_acoustics import get_fdm_modes_cached, pressure_field_polygon, speaker_coupling_polygon

app = dash.Dash(__name__, title="Room Acoustics")


@app.server.route("/load-rooms", methods=["GET"])
def route_load_rooms():
    return jsonify(_load_library())

@app.server.route("/save-room", methods=["POST"])
def route_save_room():
    data = flask_request.get_json()
    library = _load_library()
    for i, r in enumerate(library):
        if r["name"] == data["name"]:
            library[i] = data
            _save_library(library)
            return jsonify({"status": "updated"})
    library.append(data)
    _save_library(library)
    return jsonify({"status": "saved"})

@app.server.route("/delete-room", methods=["POST"])
def route_delete_room():
    data = flask_request.get_json()
    _save_library([r for r in _load_library() if r["name"] != data["name"]])
    return jsonify({"status": "deleted"})

AMBER  = "#B45309"
CREAM  = "#FAF9F6"
DARK   = "#1A1814"
WHITE  = "#FFFFFF"
BORDER = "#E8E4DC"

# ─────────────────────────────────────────────────────────────────────────────
# Bibliothèque de pièces — JSON local
# ─────────────────────────────────────────────────────────────────────────────

_LIBRARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rooms_library.json")

def _load_library():
    if os.path.exists(_LIBRARY_FILE):
        with open(_LIBRARY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_library(data):
    with open(_LIBRARY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div([

    html.Div([
        html.H1(["Room ", html.Em("Acoustics"), " — Modes de résonance"], className="title"),
    ], style={"position": "relative"}),
    html.Button("?", id="tutorial-btn", className="tutorial-help-btn", title="Lancer le tutoriel", n_clicks=0),

    # ── Onglets V1 / V2 ──────────────────────────────────────────────────
    html.Div([
        html.Button("V1 — Pièce rectangulaire", id="btn-v1", className="tab-btn tab-active"),
        html.Button("V2 — Dessin libre",         id="btn-v2", className="tab-btn"),
    ], className="tab-bar"),

    # ════════════════════════════════════════════════════════════════════════
    # V1 — MODE RECTANGULAIRE
    # ════════════════════════════════════════════════════════════════════════
    html.Div([

        html.Div([

            html.Div([
                html.H3("Dimensions de la pièce"),
                html.Div([
                    html.Div([html.Span("Longueur X"), html.Span(id="val-lx", className="slider-value")], className="slider-label"),
                    dcc.Slider(id="lx", min=2, max=15, step=0.1, value=6,
                               marks={i: f"{i}m" for i in range(2, 16, 2)}, tooltip={"placement": "bottom"}),
                ], className="slider-row"),
                html.Div([
                    html.Div([html.Span("Largeur Y"), html.Span(id="val-ly", className="slider-value")], className="slider-label"),
                    dcc.Slider(id="ly", min=2, max=15, step=0.1, value=4,
                               marks={i: f"{i}m" for i in range(2, 16, 2)}, tooltip={"placement": "bottom"}),
                ], className="slider-row"),
                html.Div([
                    html.Div([html.Span("Hauteur Z"), html.Span(id="val-lz", className="slider-value")], className="slider-label"),
                    dcc.Slider(id="lz", min=2, max=6, step=0.1, value=2.5,
                               marks={i: f"{i}m" for i in range(2, 7)}, tooltip={"placement": "bottom"}),
                ], className="slider-row"),
            ], className="control-section", id="section-dimensions"),

            html.Div([
                html.H3("Mode à visualiser"),
                html.Div([
                    html.Div([html.Span("m — axe X"), html.Span(id="val-m", className="slider-value")], className="slider-label"),
                    dcc.Slider(id="mode-m", min=0, max=4, step=1, value=1, marks={i: str(i) for i in range(5)}),
                ], className="slider-row"),
                html.Div([
                    html.Div([html.Span("n — axe Y"), html.Span(id="val-n", className="slider-value")], className="slider-label"),
                    dcc.Slider(id="mode-n", min=0, max=4, step=1, value=0, marks={i: str(i) for i in range(5)}),
                ], className="slider-row"),
                html.Div([
                    html.Div([html.Span("p — axe Z"), html.Span(id="val-p", className="slider-value")], className="slider-label"),
                    dcc.Slider(id="mode-p", min=0, max=4, step=1, value=0, marks={i: str(i) for i in range(5)}),
                ], className="slider-row"),
                html.Div(id="freq-display", className="freq-badge"),
            ], className="control-section", id="section-mode-selector"),

            html.Div([
                html.H3("Enceintes"),
                html.P("1er clic → A  |  2e clic → B  |  3e clic → replace A", className="hint"),
                html.Div(id="speaker-info"),
                html.Button("↺ Réinitialiser", id="reset-btn", className="btn btn-outline"),
            ], className="control-section"),

        ], className="controls"),

        html.Div([
            html.Div([
                html.H3("Vue 2D — Cliquer pour placer les enceintes"),
                dcc.Graph(id="graph-2d"),
            ], className="graph-box", id="section-2d-view"),
            html.Div([
                html.H3("Vue 3D — Volume"),
                dcc.Graph(id="graph-3d"),
            ], className="graph-box", id="section-3d-view"),
        ], className="graphs"),

        html.Div([
            html.H3("🔍 Analyse acoustique"),
            html.Div([html.Button("Lancer l'analyse", id="analyse-btn", className="btn")], className="analyse-btn-row"),
            dcc.Loading(type="circle", color=AMBER, children=html.Div(id="analyse-output")),
        ], className="analyse-section", id="section-analyse"),

        html.Div([
            html.H3("Premiers modes de résonance"),
            html.Div(id="modes-table"),
        ], className="modes-section", id="section-modes-table"),

    ], id="panel-v1"),

    # ════════════════════════════════════════════════════════════════════════
    # V2 — MODE DESSIN LIBRE
    # ════════════════════════════════════════════════════════════════════════
    html.Div([

        # Barre d'outils dessin
        html.Div([
            html.Div([
                html.H3("Dessiner la pièce"),
                html.P("1. Définis la taille du canevas, 2. Clique pour placer les murs, 3. Ferme la pièce.", className="hint"),
                html.Div([
                    html.Div([html.Span("Largeur canevas"), html.Span(id="val-canvas-x", className="slider-value")], className="slider-label"),
                    dcc.Slider(id="canvas-x", min=2, max=15, step=0.5, value=8,
                               marks={i: f"{i}m" for i in range(2, 16, 2)}, tooltip={"placement": "bottom"}),
                ], className="slider-row"),
                html.Div([
                    html.Div([html.Span("Hauteur canevas"), html.Span(id="val-canvas-y", className="slider-value")], className="slider-label"),
                    dcc.Slider(id="canvas-y", min=2, max=12, step=0.5, value=6,
                               marks={i: f"{i}m" for i in range(2, 13, 2)}, tooltip={"placement": "bottom"}),
                ], className="slider-row"),
                html.Div([
                    html.Button("✓ Fermer la pièce", id="btn-close-room", className="btn"),
                    html.Button("↩ Annuler dernier", id="btn-undo", className="btn btn-outline"),
                    html.Button("✕ Tout effacer", id="btn-clear-room", className="btn btn-outline"),
                ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "10px"}),
            ], className="control-section"),

            html.Div([
                html.H3("État de la pièce"),
                html.Div(id="room-info"),
                html.H3("Enceintes", style={"marginTop": "14px"}),
                html.P("Cliquer sur le plan après avoir fermé la pièce.", className="hint"),
                html.Div(id="speaker-info-v2"),
                html.Button("↺ Réinitialiser enceintes", id="reset-btn-v2", className="btn btn-outline"),
            ], className="control-section"),

            html.Div([
                html.H3("Mode à visualiser"),
                html.P("Calculé par FDM sur la vraie forme de la pièce.", className="hint"),
                html.Div([
                    html.Div([html.Span("Hauteur Z"), html.Span(id="val-lz2", className="slider-value")], className="slider-label"),
                    dcc.Slider(id="lz2", min=2, max=6, step=0.1, value=2.5,
                               marks={i: f"{i}m" for i in range(2, 7)}, tooltip={"placement": "bottom"}),
                ], className="slider-row"),
                html.Label("Sélectionner un mode :", style={"fontSize":"0.82rem","color":"#6B6560"}),
                dcc.Slider(id="mode-idx2", min=0, max=14, step=1, value=0,
                           marks={i: str(i) for i in range(0,15,2)},
                           tooltip={"placement":"bottom"}),
                html.Div(id="freq-display-v2", className="freq-badge"),
            ], className="control-section"),

            html.Div([
                html.H3("Bibliothèque de pièces"),
                html.Div([
                    dcc.Input(id="room-name-input", type="text", placeholder="Nommer cette pièce…",
                              style={"flex": "1", "padding": "6px 10px", "border": f"1px solid {BORDER}",
                                     "borderRadius": "6px", "fontFamily": "Outfit, sans-serif", "fontSize": "0.88rem"}),
                    html.Button("💾 Sauvegarder", id="btn-save-room", className="btn"),
                ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),
                html.Div([
                    dcc.Dropdown(id="room-library-dropdown", placeholder="Charger une pièce sauvegardée…",
                                 options=[], clearable=True,
                                 style={"flex": "1", "fontSize": "0.88rem"}),
                    html.Button("🗑️ Supprimer", id="btn-delete-room", className="btn btn-outline"),
                ], style={"display": "flex", "gap": "8px", "alignItems": "center", "marginTop": "8px"}),
                html.Div(id="room-library-status", style={"marginTop": "6px", "fontSize": "0.82rem"}),
            ], className="control-section"),
        ], className="controls"),

        # Plan de dessin
        html.Div([
            html.Div([
                html.H3("Plan de la pièce — Papier millimétré"),
                dcc.Graph(id="graph-room-editor",
                          config={"scrollZoom": False, "displayModeBar": False, "staticPlot": False, "responsive": True},
                          style={"height": "500px"},
                          clickData=None),
            ], className="graph-box", style={"flex": "2"}),
            html.Div([
                html.H3("Vue 3D"),
                dcc.Graph(id="graph-3d-v2"),
            ], className="graph-box"),
        ], className="graphs"),

        html.Div([
            html.H3("🔍 Analyse acoustique"),
            html.Div([html.Button("Lancer l'analyse", id="analyse-btn-v2", className="btn")], className="analyse-btn-row"),
            dcc.Loading(type="circle", color=AMBER, children=html.Div(id="analyse-output-v2")),
        ], className="analyse-section"),

        html.Div([
            html.H3("Modes de résonance (bounding box)"),
            html.Div(id="modes-table-v2"),
        ], className="modes-section"),

    ], id="panel-v2", style={"display": "none"}),

    # Stores
    dcc.Store(id="speakers-store",    data={"s1": None, "s2": None}),
    dcc.Store(id="suggestions-store", data={}),
    dcc.Store(id="room-points-store", data={"points": [], "closed": False}),
    dcc.Store(id="speakers-store-v2", data={"s1": None, "s2": None}),
    dcc.Store(id="rooms-library-store", data=[]),
    dcc.Store(id="tutorial-seen", storage_type="local", data=False),
    dcc.Store(id="tutorial-step", data=-1),

    # Tutorial overlay
    html.Div(id="tutorial-hl-dummy", style={"display": "none"}),
    html.Div([
        html.Div(className="tutorial-dim"),
        html.Div([
            html.Div(id="tutorial-step-indicator", className="tutorial-step-indicator"),
            html.H4(id="tutorial-title", className="tutorial-title"),
            html.P(id="tutorial-text", className="tutorial-text"),
            html.Div([
                html.Button("← Retour", id="tutorial-prev", className="btn btn-outline", n_clicks=0),
                html.Button("Suivant →", id="tutorial-next", className="btn", n_clicks=0),
                html.Button("Passer", id="tutorial-skip", className="tutorial-skip-btn", n_clicks=0),
            ], className="tutorial-actions"),
        ], className="tutorial-card"),
    ], id="tutorial-overlay", style={"display": "none"}),

], className="container")


# ─────────────────────────────────────────────────────────────────────────────
# Onglets
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("panel-v1", "style"),
    Output("panel-v2", "style"),
    Output("btn-v1", "className"),
    Output("btn-v2", "className"),
    Input("btn-v1", "n_clicks"),
    Input("btn-v2", "n_clicks"),
)
def switch_tab(c1, c2):
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else "btn-v1.n_clicks"
    if "btn-v2" in trigger:
        return {"display": "none"}, {}, "tab-btn", "tab-btn tab-active"
    return {}, {"display": "none"}, "tab-btn tab-active", "tab-btn"


# ─────────────────────────────────────────────────────────────────────────────
# V1 — Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(Output("val-lx","children"), Input("lx","value"))
def v_lx(v): return f"{v:.1f} m"

@app.callback(Output("val-ly","children"), Input("ly","value"))
def v_ly(v): return f"{v:.1f} m"

@app.callback(Output("val-lz","children"), Input("lz","value"))
def v_lz(v): return f"{v:.1f} m"

@app.callback(Output("val-m","children"), Input("mode-m","value"))
def v_m(v): return str(v)

@app.callback(Output("val-n","children"), Input("mode-n","value"))
def v_n(v): return str(v)

@app.callback(Output("val-p","children"), Input("mode-p","value"))
def v_p(v): return str(v)


@app.callback(
    Output("freq-display","children"),
    Input("mode-m","value"), Input("mode-n","value"), Input("mode-p","value"),
    Input("lx","value"), Input("ly","value"), Input("lz","value"),
)
def update_freq(m, n, p, Lx, Ly, Lz):
    if m == n == p == 0: return "Mode (0,0,0) — non pertinent"
    freq = room_mode_frequency(m, n, p, Lx, Ly, Lz)
    nz = sum([m>0, n>0, p>0])
    mt = "Axial ★★★" if nz==1 else ("Tangentiel ★★☆" if nz==2 else "Oblique ★☆☆")
    return f"{freq:.1f} Hz  —  {mt}"


@app.callback(
    Output("speakers-store","data"),
    Input("graph-2d","clickData"), Input("reset-btn","n_clicks"),
    State("speakers-store","data"),
)
def update_speakers(clickData, reset, spk):
    ctx = callback_context
    if not ctx.triggered: return spk
    if "reset-btn" in ctx.triggered[0]["prop_id"]: return {"s1": None, "s2": None}
    if clickData:
        x = round(clickData["points"][0]["x"], 1)
        y = round(clickData["points"][0]["y"], 1)
        if spk["s1"] is None:   spk["s1"] = {"x": x, "y": y}
        elif spk["s2"] is None: spk["s2"] = {"x": x, "y": y}
        else:                   spk = {"s1": {"x": x, "y": y}, "s2": None}
    return spk


@app.callback(
    Output("speaker-info","children"),
    Input("speakers-store","data"),
    Input("mode-m","value"), Input("mode-n","value"),
    Input("lx","value"), Input("ly","value"),
)
def spk_info(spk, m, n, Lx, Ly):
    items = []
    for label, key, icon in [("Enceinte A","s1","🔶"), ("Enceinte B","s2","⚫")]:
        s = spk.get(key)
        if s:
            c = speaker_coupling(s["x"], s["y"], m, n, Lx, Ly)
            q = "Fort ⚠️" if c > 0.7 else ("Moyen" if c > 0.3 else "Faible ✅")
            items.append(html.P(f"{icon} {label}  X={s['x']:.1f}m ({int(s['x']*100)}cm)  Y={s['y']:.1f}m ({int(s['y']*100)}cm)  {q}"))
        else:
            items.append(html.P(f"{icon} {label} — non placée", style={"color":"#9A948E"}))
    return items


@app.callback(
    Output("graph-2d","figure"),
    Input("mode-m","value"), Input("mode-n","value"),
    Input("lx","value"), Input("ly","value"),
    Input("speakers-store","data"), Input("suggestions-store","data"),
)
def graph2d(m, n, Lx, Ly, spk, sugg):
    X, Y, P = pressure_field_2d(m, n, Lx, Ly, resolution=120)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=P, x=X[0], y=Y[:,0],
        colorscale=[[0,"#1D4ED8"],[0.5,CREAM],[1,AMBER]],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="Pression", font=dict(color=AMBER)), tickfont=dict(color="#6B6560"))))

    for key, sug in sugg.items():
        s_orig = spk.get(key)
        fig.add_trace(go.Scatter(x=[sug["x"]], y=[sug["y"]], mode="markers",
            marker=dict(size=28, color="rgba(0,0,0,0)", line=dict(width=2.5, color=AMBER)),
            name=f"Suggéré {sug['label']}",
            hovertemplate=f"Suggestion<br>X={sug['x']:.1f}m Y={sug['y']:.1f}m<extra></extra>"))
        if s_orig:
            fig.add_annotation(x=sug["x"], y=sug["y"], ax=s_orig["x"], ay=s_orig["y"],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=AMBER)

    s1 = spk.get("s1")
    if s1:
        fig.add_trace(go.Scatter(x=[s1["x"]], y=[s1["y"]], mode="markers+text",
            text=["A"], textposition="middle center",
            textfont=dict(size=13, color=WHITE, family="Arial Black"),
            marker=dict(size=30, color=AMBER, line=dict(width=2, color=WHITE)),
            name="Enceinte A",
            hovertemplate=f"Enceinte A<br>X={s1['x']:.1f}m ({int(s1['x']*100)}cm)<br>Y={s1['y']:.1f}m ({int(s1['y']*100)}cm)<extra></extra>"))

    s2 = spk.get("s2")
    if s2:
        fig.add_trace(go.Scatter(x=[s2["x"]], y=[s2["y"]], mode="markers+text",
            text=["B"], textposition="middle center",
            textfont=dict(size=13, color=DARK, family="Arial Black"),
            marker=dict(size=30, color=WHITE, line=dict(width=3, color=AMBER)),
            name="Enceinte B",
            hovertemplate=f"Enceinte B<br>X={s2['x']:.1f}m ({int(s2['x']*100)}cm)<br>Y={s2['y']:.1f}m ({int(s2['y']*100)}cm)<extra></extra>"))

    fig.update_layout(
        xaxis_title="X (m)", yaxis_title="Y (m)",
        xaxis=dict(range=[0,Lx], gridcolor="#F0EDE6", dtick=0.5),
        yaxis=dict(range=[0,Ly], scaleanchor="x", gridcolor="#F0EDE6", dtick=0.5),
        margin=dict(l=40,r=20,t=10,b=40), height=430,
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(color=DARK, family="Outfit, sans-serif"),
        legend=dict(bgcolor=CREAM))
    return fig


@app.callback(
    Output("graph-3d","figure"),
    Input("mode-m","value"), Input("mode-n","value"), Input("mode-p","value"),
    Input("lx","value"), Input("ly","value"), Input("lz","value"),
    Input("speakers-store","data"),
)
def graph3d(m, n, p, Lx, Ly, Lz, spk):
    X, Y, Z, P = pressure_field_3d(m, n, p, Lx, Ly, Lz, resolution=25)
    fig = go.Figure()
    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=P.flatten(),
        isomin=-0.8, isomax=0.8, surface_count=5,
        colorscale=[[0,"#1D4ED8"],[0.5,CREAM],[1,AMBER]],
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.45, showscale=True,
        colorbar=dict(title=dict(text="Pression", font=dict(color=AMBER)), tickfont=dict(color="#6B6560"))))
    for key, color, border, label in [("s1",AMBER,WHITE,"A"), ("s2",WHITE,AMBER,"B")]:
        s = spk.get(key)
        if s:
            fig.add_trace(go.Scatter3d(
                x=[s["x"]], y=[s["y"]], z=[0.05], mode="markers+text",
                text=[label], textposition="middle center",
                textfont=dict(size=12, color=DARK if color==WHITE else WHITE, family="Arial Black"),
                marker=dict(size=10, color=color, symbol="circle", line=dict(color=border, width=2)),
                name=f"Enceinte {label}"))
    ratio = max(Lx, Ly, Lz)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0,Lx], title="X (m)", backgroundcolor=CREAM, gridcolor=BORDER),
            yaxis=dict(range=[0,Ly], title="Y (m)", backgroundcolor=CREAM, gridcolor=BORDER),
            zaxis=dict(range=[0,Lz], title="Z (m)", backgroundcolor=CREAM, gridcolor=BORDER),
            bgcolor=CREAM, aspectmode="manual",
            aspectratio=dict(x=Lx/ratio, y=Ly/ratio, z=Lz/ratio)),
        margin=dict(l=0,r=0,t=10,b=0), height=450,
        paper_bgcolor=WHITE, font=dict(color=DARK, family="Outfit, sans-serif"))
    return fig


@app.callback(
    Output("modes-table","children"),
    Input("lx","value"), Input("ly","value"), Input("lz","value"),
)
def modes_table(Lx, Ly, Lz):
    modes = compute_modes(Lx, Ly, Lz, max_order=3)[:15]
    header = html.Tr([html.Th(c) for c in ["Mode","Type","Fréq. (Hz)","Importance","Impact"]])
    rows = [html.Tr([
        html.Td(f"({m['m']},{m['n']},{m['p']})"),
        html.Td(m["type"], style={"color":m["color"],"fontWeight":"600"}),
        html.Td(f"{m['freq']:.1f}"),
        html.Td(m["stars"], style={"color":m["color"],"letterSpacing":"2px"}),
        html.Td(m["comment"], style={"color":"#6B6560","fontSize":"0.78rem"}),
    ]) for m in modes]
    return html.Table([html.Thead(header), html.Tbody(rows)], className="modes-table")


@app.callback(
    Output("analyse-output","children"),
    Output("suggestions-store","data"),
    Input("analyse-btn","n_clicks"),
    State("lx","value"), State("ly","value"), State("lz","value"),
    State("mode-m","value"), State("mode-n","value"), State("mode-p","value"),
    State("speakers-store","data"),
    prevent_initial_call=True,
)
def run_analysis(_, Lx, Ly, Lz, m, n, p, spk):
    result = analyse_room(Lx, Ly, Lz, m, n, p, spk)
    conclusion = html.Div([
        html.Strong("⚡ Corrections prioritaires"),
        html.Ul([html.Li(pr) for pr in result["priorities"]], style={"marginTop":"8px","paddingLeft":"18px"}),
    ], className="mini-conclusion") if result["priorities"] else html.Div("✅ Aucun problème majeur.", className="mini-conclusion")

    def col(title, items, color):
        return html.Div([
            html.Div(title, className="analyse-col-title", style={"color":color}),
            *([html.P(i) for i in items] if items else [html.P("Aucun élément.", style={"color":"#9A948E","fontStyle":"italic"})]),
        ], className="analyse-col")

    grid = html.Div([
        col("🔴 Problèmes", result["problemes"], AMBER),
        col("👂 Impact", result["impact"], "#92400E"),
        col("✅ Recommandations", result["recommandations"], "#166534"),
    ], className="analyse-grid")

    return html.Div([conclusion, grid]), result.get("suggestions", {})


# ─────────────────────────────────────────────────────────────────────────────
# V2 — Callbacks éditeur de plan
# ─────────────────────────────────────────────────────────────────────────────







@app.callback(Output("val-lz2","children"), Input("lz2","value"))
def v_lz2(v): return f"{v:.1f} m"



@app.callback(Output("val-canvas-x","children"), Input("canvas-x","value"))
def v_cx(v): return f"{v:.1f} m"

@app.callback(Output("val-canvas-y","children"), Input("canvas-y","value"))
def v_cy(v): return f"{v:.1f} m"

@app.callback(
    Output("room-points-store","data"),
    Input("graph-room-editor","clickData"),
    Input("btn-close-room","n_clicks"),
    Input("btn-undo","n_clicks"),
    Input("btn-clear-room","n_clicks"),
    State("room-points-store","data"),
)
def update_room_points(clickData, close, undo, clear, store):
    ctx = callback_context
    if not ctx.triggered: return store
    trigger = ctx.triggered[0]["prop_id"]

    if "btn-clear-room" in trigger:
        return {"points": [], "closed": False}

    if "btn-undo" in trigger:
        pts = store["points"][:-1]
        return {"points": pts, "closed": False}

    if "btn-close-room" in trigger:
        if len(store["points"]) >= 3:
            return {"points": store["points"], "closed": True}
        return store

    if "graph-room-editor" in trigger and clickData and not store["closed"]:
        x = round(clickData["points"][0]["x"], 1)
        y = round(clickData["points"][0]["y"], 1)
        # Évite les doublons
        if store["points"] and store["points"][-1] == {"x": x, "y": y}:
            return store
        pts = store["points"] + [{"x": x, "y": y}]
        return {"points": pts, "closed": False}

    return store


@app.callback(
    Output("graph-room-editor","figure"),
    Input("room-points-store","data"),
    Input("speakers-store-v2","data"),
    Input("mode-idx2","value"),
    Input("lz2","value"),
    Input("canvas-x","value"),
    Input("canvas-y","value"),
)
def update_room_editor(store, spk, mode_idx, Lz, canvas_x, canvas_y):
    points = store["points"]
    closed = store["closed"]
    fig = make_grid_figure(points, closed, Lx_max=canvas_x, Ly_max=canvas_y)

    if closed and len(points) >= 3:
        modes, fields, grid_info = get_fdm_modes_cached(points, Lz)

        # Superposer le champ de pression FDM si mode disponible
        if modes and mode_idx is not None and int(mode_idx) < len(fields):
            fidx = int(mode_idx)
            if fields[fidx] is not None:
                x_grid = np.array(grid_info["x_grid"])
                y_grid = np.array(grid_info["y_grid"])
                P = fields[fidx]
                P_disp = np.where(np.isnan(P), None, P)
                fig.add_trace(go.Heatmap(
                    z=P_disp, x=x_grid, y=y_grid,
                    colorscale=[[0,"#1D4ED8"],[0.5,"#FAF9F6"],[1,"#B45309"]],
                    zmid=0, zmin=-1, zmax=1, opacity=0.75,
                    colorbar=dict(title=dict(text="Pression", font=dict(color=AMBER)),
                                  tickfont=dict(color="#6B6560")),
                    showscale=True,
                ))
            else:
                fig.add_annotation(
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    text="Mode axial Z — variation uniquement en hauteur<br>Voir la vue 3D",
                    showarrow=False, font=dict(size=13, color=AMBER, family="Outfit"),
                    bgcolor="rgba(255,255,255,0.88)", bordercolor=AMBER, borderwidth=1, borderpad=8)

        # Enceintes
        for key, color, txt_color, border, label in [
            ("s1", AMBER, WHITE, WHITE, "A"),
            ("s2", WHITE, DARK, AMBER, "B")
        ]:
            s = spk.get(key)
            if s:
                # Couplage réel FDM
                c = 0.0
                if modes and mode_idx is not None and int(mode_idx) < len(fields):
                    fidx = int(mode_idx)
                    if fields[fidx] is not None:
                        c = speaker_coupling_polygon(s["x"], s["y"], fields[fidx],
                                                     grid_info["x_grid"], grid_info["y_grid"])
                fig.add_trace(go.Scatter(
                    x=[s["x"]], y=[s["y"]], mode="markers+text",
                    text=[label], textposition="middle center",
                    textfont=dict(size=13, color=txt_color, family="Arial Black"),
                    marker=dict(size=30, color=color, line=dict(width=3, color=border)),
                    name=f"Enceinte {label}",
                    hovertemplate=f"Enceinte {label}<br>X={s['x']:.1f}m  Y={s['y']:.1f}m<br>Couplage FDM: {c:.2f}<extra></extra>"))

    return fig


@app.callback(
    Output("room-info","children"),
    Input("room-points-store","data"),
)
def update_room_info(store):
    pts = store["points"]
    closed = store["closed"]
    if not pts:
        return html.P("Aucun point placé.", style={"color":"#9A948E"})
    info = [html.P(f"Points : {len(pts)}", style={"fontWeight":"600"})]
    if closed:
        Lx, Ly = polygon_bounding_box(pts)
        area = polygon_area(pts)
        info += [
            html.P(f"✅ Pièce fermée", style={"color":"#166534","fontWeight":"600"}),
            html.P(f"Bounding box : {Lx:.1f}m × {Ly:.1f}m"),
            html.P(f"Surface réelle : {area:.1f} m²"),
        ]
        from room_editor import is_complex_shape
        if is_complex_shape(pts):
            info.append(html.P("⚠️ Forme complexe : calcul acoustique basé sur la bounding box (approximation).",
                               style={"color":AMBER,"fontSize":"0.78rem"}))
    else:
        info.append(html.P("Cliquer pour ajouter des points, puis 'Fermer la pièce'.", style={"color":"#9A948E"}))
    return html.Div(info)


@app.callback(
    Output("speakers-store-v2","data"),
    Input("graph-room-editor","clickData"),
    Input("reset-btn-v2","n_clicks"),
    State("speakers-store-v2","data"),
    State("room-points-store","data"),
)
def update_spk_v2(clickData, reset, spk, room):
    ctx = callback_context
    if not ctx.triggered: return spk
    if "reset-btn-v2" in ctx.triggered[0]["prop_id"]: return {"s1": None, "s2": None}

    # On place les enceintes seulement si la pièce est fermée
    if "graph-room-editor" in ctx.triggered[0]["prop_id"] and clickData and room["closed"]:
        x = round(clickData["points"][0]["x"], 1)
        y = round(clickData["points"][0]["y"], 1)
        if spk["s1"] is None:   spk["s1"] = {"x": x, "y": y}
        elif spk["s2"] is None: spk["s2"] = {"x": x, "y": y}
        else:                   spk = {"s1": {"x": x, "y": y}, "s2": None}
    return spk


@app.callback(
    Output("speaker-info-v2","children"),
    Input("speakers-store-v2","data"),
    Input("room-points-store","data"),
    Input("mode-idx2","value"),
    Input("lz2","value"),
)
def spk_info_v2(spk, room, mode_idx, Lz):
    if not room["closed"]:
        return html.P("Fermez d'abord la pièce.", style={"color":"#9A948E"})
    modes, fields, grid_info = get_fdm_modes_cached(room["points"], Lz)
    items = []
    for label, key, icon in [("Enceinte A","s1","🔶"), ("Enceinte B","s2","⚫")]:
        s = spk.get(key)
        if s:
            c = 0.0
            if modes and mode_idx is not None and int(mode_idx) < len(fields) and fields[int(mode_idx)] is not None:
                c = speaker_coupling_polygon(s["x"], s["y"], fields[int(mode_idx)],
                                             grid_info["x_grid"], grid_info["y_grid"])
            q = "Fort ⚠️" if c > 0.7 else ("Moyen" if c > 0.3 else "Faible ✅")
            items.append(html.P(f"{icon} {label}  X={s['x']:.1f}m ({int(s['x']*100)}cm)  Y={s['y']:.1f}m ({int(s['y']*100)}cm)  Couplage FDM: {q}"))
        else:
            items.append(html.P(f"{icon} {label} — non placée", style={"color":"#9A948E"}))
    return items


@app.callback(
    Output("freq-display-v2","children"),
    Input("mode-idx2","value"),
    Input("room-points-store","data"), Input("lz2","value"),
)
def freq_v2(mode_idx, room, Lz):
    if not room["closed"]: return "Fermer la pièce pour calculer"
    modes, _, _ = get_fdm_modes_cached(room["points"], Lz)
    if not modes or mode_idx is None or mode_idx >= len(modes):
        return "—"
    m = modes[int(mode_idx)]
    return f"{m['freq']:.1f} Hz  —  {m['type']}  {m['stars']}"


@app.callback(
    Output("graph-3d-v2","figure"),
    Input("mode-idx2","value"),
    Input("room-points-store","data"), Input("lz2","value"),
    Input("speakers-store-v2","data"),
)
def graph3d_v2(mode_idx, room, Lz, spk):
    import re as _re
    empty = go.Figure()
    empty.update_layout(paper_bgcolor=WHITE, plot_bgcolor=CREAM, height=450,
                        annotations=[dict(text="Dessiner et fermer la pièce", x=0.5, y=0.5,
                                         xref="paper", yref="paper", showarrow=False,
                                         font=dict(size=16, color="#9A948E"))])
    if not room["closed"] or len(room["points"]) < 3:
        return empty

    pts = room["points"]
    modes, fields, grid_info = get_fdm_modes_cached(pts, Lz)
    if not modes or not grid_info:
        return empty

    fidx = min(int(mode_idx) if mode_idx is not None else 0, len(modes) - 1)
    field = fields[fidx] if fidx < len(fields) else None

    x_grid = np.array(grid_info["x_grid"])
    y_grid = np.array(grid_info["y_grid"])
    mask = grid_info["mask"]  # (ny, nx)

    # Sous-échantillonnage pour performance 3D
    step = max(1, len(x_grid) // 40)
    x_3d = x_grid[::step]
    y_3d = y_grid[::step]
    mask_ds = mask[::step, ::step]   # (ny_ds, nx_ds)

    nz = 8
    z_vals = np.linspace(0, Lz, nz)
    X3d, Y3d, Z3d = np.meshgrid(x_3d, y_3d, z_vals, indexing="ij")  # (nx_ds, ny_ds, nz)

    if field is not None:
        # Mode 2D FDM extrudé en Z (variation uniforme sur la hauteur)
        field_ds = field[::step, ::step]          # (ny_ds, nx_ds)
        P3d = np.tile(field_ds.T[:, :, np.newaxis], (1, 1, nz))  # (nx_ds, ny_ds, nz)
        mask_3d = np.tile(mask_ds.T[:, :, np.newaxis], (1, 1, nz))
        P3d[~mask_3d] = np.nan
    else:
        # Mode axial Z : P(x,y,z) = cos(p_z·π·z/Lz) dans le polygone
        p_z_val = 1
        m_pz = _re.search(r"p=(\d+)", modes[fidx].get("comment", ""))
        if m_pz:
            p_z_val = int(m_pz.group(1))
        inside = np.where(mask_ds.T, 1.0, np.nan)  # (nx_ds, ny_ds)
        cos_z = np.cos(p_z_val * np.pi * z_vals / Lz)
        P3d = inside[:, :, np.newaxis] * cos_z[np.newaxis, np.newaxis, :]

    fig = go.Figure()
    fig.add_trace(go.Isosurface(
        x=X3d.flatten(), y=Y3d.flatten(), z=Z3d.flatten(),
        value=P3d.flatten(),
        isomin=-0.7, isomax=0.7, surface_count=5,
        colorscale=[[0,"#1D4ED8"],[0.5,CREAM],[1,AMBER]],
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.5, showscale=True,
        colorbar=dict(title=dict(text="Pression", font=dict(color=AMBER)), tickfont=dict(color="#6B6560"))))

    for key, color, border, label in [("s1",AMBER,WHITE,"A"),("s2",WHITE,AMBER,"B")]:
        s = spk.get(key)
        if s:
            fig.add_trace(go.Scatter3d(
                x=[s["x"]], y=[s["y"]], z=[0.05], mode="markers+text",
                text=[label], textposition="middle center",
                textfont=dict(size=12, color=DARK if color==WHITE else WHITE, family="Arial Black"),
                marker=dict(size=10, color=color, symbol="circle", line=dict(color=border, width=2)),
                name=f"Enceinte {label}"))

    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    Lx = max(xs) - min(xs)
    Ly = max(ys) - min(ys)
    ratio = max(Lx, Ly, Lz)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[min(xs), max(xs)], title="X (m)", backgroundcolor=CREAM, gridcolor=BORDER),
            yaxis=dict(range=[min(ys), max(ys)], title="Y (m)", backgroundcolor=CREAM, gridcolor=BORDER),
            zaxis=dict(range=[0, Lz], title="Z (m)", backgroundcolor=CREAM, gridcolor=BORDER),
            bgcolor=CREAM, aspectmode="manual",
            aspectratio=dict(x=Lx/ratio, y=Ly/ratio, z=Lz/ratio)),
        margin=dict(l=0, r=0, t=10, b=0), height=450,
        paper_bgcolor=WHITE, font=dict(color=DARK, family="Outfit, sans-serif"))
    return fig


@app.callback(
    Output("modes-table-v2","children"),
    Input("room-points-store","data"), Input("lz2","value"),
)
def modes_table_v2(room, Lz):
    if not room["closed"]:
        return html.P("Fermer la pièce pour voir les modes.", style={"color":"#9A948E"})
    pts = room["points"]
    modes, _, _ = get_fdm_modes_cached(pts, Lz)
    if not modes:
        return html.P("Erreur de calcul.", style={"color":"#B45309"})
    header = html.Tr([html.Th(c) for c in ["Fréq. (Hz)","Type","Importance","Impact","Méthode"]])
    rows = [html.Tr([
        html.Td(f"{m['freq']:.1f}", style={"fontWeight":"600"}),
        html.Td(m["type"], style={"color":m["color"],"fontWeight":"600"}),
        html.Td(m["stars"], style={"color":m["color"],"letterSpacing":"2px"}),
        html.Td(m["comment"], style={"color":"#6B6560","fontSize":"0.78rem"}),
        html.Td(m.get("source",""), style={"color":"#9A948E","fontSize":"0.75rem"}),
    ]) for m in modes[:15]]
    return html.Div([
        html.P("✅ Calcul réel par Différences Finies (FDM) — Equation de Helmholtz 2D avec conditions de Neumann.",
               style={"fontSize":"0.78rem","color":"#166534","marginBottom":"10px","fontStyle":"italic"}),
        html.Table([html.Thead(header), html.Tbody(rows)], className="modes-table"),
    ])


@app.callback(
    Output("analyse-output-v2","children"),
    Input("analyse-btn-v2","n_clicks"),
    State("room-points-store","data"), State("lz2","value"),
    State("mode-idx2","value"),
    State("speakers-store-v2","data"),
    prevent_initial_call=True,
)
def run_analysis_v2(_, room, Lz, mode_idx, spk):
    if not room["closed"]:
        return html.P("Fermer la pièce avant de lancer l'analyse.", style={"color":AMBER})

    pts = room["points"]
    modes, fields, grid_info = get_fdm_modes_cached(pts, Lz)

    if not modes:
        return html.P("Erreur lors du calcul FDM.", style={"color":AMBER})

    fidx = int(mode_idx) if mode_idx is not None else 0
    fidx = min(fidx, len(modes)-1)
    selected_mode = modes[fidx]
    selected_field = fields[fidx] if fidx < len(fields) else None

    freq = selected_mode["freq"]
    mode_type = selected_mode["type"]

    # Couplage réel FDM pour chaque enceinte
    s1 = spk.get("s1")
    s2 = spk.get("s2")
    c1 = speaker_coupling_polygon(s1["x"],s1["y"],selected_field,grid_info["x_grid"],grid_info["y_grid"]) if s1 and selected_field is not None else None
    c2 = speaker_coupling_polygon(s2["x"],s2["y"],selected_field,grid_info["x_grid"],grid_info["y_grid"]) if s2 and selected_field is not None else None

    problems, impacts, recommendations, priorities = [], [], [], []

    # Mode sélectionné
    if "Axial" in mode_type:
        problems.append(f"🔴 Mode à {freq:.1f} Hz ({mode_type}) — très énergétique. Crée des zones de boom ou de creux importants dans la pièce.")
    elif "Tangentiel" in mode_type:
        problems.append(f"🟡 Mode à {freq:.1f} Hz ({mode_type}) — énergie modérée. Colorations sonores perceptibles.")
    else:
        impacts.append(f"✅ Mode à {freq:.1f} Hz ({mode_type}) — faible impact.")

    # 3 premiers modes les plus problématiques
    top_modes = [m for m in modes if "Axial" in m["type"] or "Tangentiel" in m["type"]][:3]
    if top_modes:
        fs = " / ".join([f"{m['freq']:.0f} Hz" for m in top_modes])
        problems.append(f"🔴 Modes les plus énergétiques de cette pièce (calcul FDM réel) : {fs}. Ce sont ces fréquences qui créent les 'booms' graves.")

    # Couplage enceintes
    for label, s, c in [("Enceinte A", s1, c1), ("Enceinte B", s2, c2)]:
        if s is None: continue
        pos = f"X={s['x']:.1f}m Y={s['y']:.1f}m"
        if c is None:
            impacts.append(f"⚠️ {label} : couplage non calculable (mode Z analytique).")
        elif c > 0.7:
            problems.append(f"🔴 {label} en {pos} : couplage fort ({c:.2f}) — excite fortement ce mode à {freq:.0f} Hz.")
            recommendations.append(f"Déplacer {label} vers une zone de nœud (zone bleue sur le plan) pour réduire ce couplage.")
            priorities.append(f"{label} : couplage fort ({c:.2f}) à {freq:.0f} Hz")
        elif c < 0.3:
            impacts.append(f"✅ {label} en {pos} : bon couplage ({c:.2f}) — position favorable pour ce mode.")
        else:
            impacts.append(f"🟡 {label} en {pos} : couplage intermédiaire ({c:.2f}).")

        # Distance mur
        from analysis import wall_distance, MIN_WALL_DIST, MAX_WALL_DIST
        wd = wall_distance(s["x"], s["y"], max([p["x"] for p in pts]), max([p["y"] for p in pts]))
        if wd < MIN_WALL_DIST:
            problems.append(f"⚠️ {label} trop proche d'un mur ({int(wd*100)}cm) → +6 dB non contrôlé dans les graves.")
            priorities.append(f"{label} trop proche d'un mur ({int(wd*100)}cm)")
        elif wd > MAX_WALL_DIST:
            problems.append(f"⚠️ {label} trop loin des murs ({int(wd*100)}cm > 100cm) → perte de soutien des basses.")
            priorities.append(f"{label} trop loin des murs ({int(wd*100)}cm)")

    recommendations.append("Positionnez les enceintes dans les zones blanches/claires du plan (nœuds, pression ≈ 0) — évitez les zones bleues (−1) et ambrées (+1) qui sont des ventres de résonance.")
    recommendations.append("REW (Room EQ Wizard, gratuit) permet de mesurer la réponse réelle et de valider ces calculs.")

    # Forme complexe : info
    if is_complex_shape(pts):
        impacts.append(f"ℹ️ Pièce de forme complexe ({len(pts)} sommets) — calcul FDM réel sur le polygone exact ({len(modes)} modes calculés).")

    banner = html.Div(
        f"✅ Calcul acoustique réel par FDM — Equation de Helmholtz 2D — {len(modes)} modes calculés sur le polygone exact.",
        className="mini-conclusion", style={"borderLeftColor":"#166534","color":"#166534"})

    conclusion = html.Div([
        html.Strong("⚡ Corrections prioritaires"),
        html.Ul([html.Li(pr) for pr in priorities], style={"marginTop":"8px","paddingLeft":"18px"}),
    ], className="mini-conclusion") if priorities else html.Div("✅ Aucun problème majeur.", className="mini-conclusion")

    def col(title, items, color):
        return html.Div([
            html.Div(title, className="analyse-col-title", style={"color":color}),
            *([html.P(i) for i in items] if items else [html.P("Aucun.", style={"color":"#9A948E","fontStyle":"italic"})]),
        ], className="analyse-col")

    grid = html.Div([
        col("🔴 Problèmes", problems, AMBER),
        col("👂 Impact", impacts, "#92400E"),
        col("✅ Recommandations", recommendations, "#166534"),
    ], className="analyse-grid")

    return html.Div([banner, conclusion, grid])


# ─────────────────────────────────────────────────────────────────────────────
# V2 — Bibliothèque de pièces
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("rooms-library-store", "data"),
    Output("room-library-status", "children"),
    Input("btn-save-room", "n_clicks"),
    Input("btn-delete-room", "n_clicks"),
    State("room-name-input", "value"),
    State("room-points-store", "data"),
    State("canvas-x", "value"),
    State("canvas-y", "value"),
    State("lz2", "value"),
    State("room-library-dropdown", "value"),
    prevent_initial_call=True,
)
def library_actions(save_clicks, delete_clicks, name, room_store, canvas_x, canvas_y, lz, selected_name):
    ctx = callback_context
    if not ctx.triggered:
        return _load_library(), no_update
    trigger = ctx.triggered[0]["prop_id"]

    if "btn-save-room" in trigger:
        if not name or not name.strip():
            return no_update, html.Span("⚠️ Entrez un nom.", style={"color": AMBER})
        if not room_store.get("points"):
            return no_update, html.Span("⚠️ Aucun point à sauvegarder.", style={"color": AMBER})
        room_data = {
            "name": name.strip(),
            "points": room_store["points"],
            "closed": room_store.get("closed", False),
            "canvas_x": canvas_x,
            "canvas_y": canvas_y,
            "lz": lz,
        }
        library = _load_library()
        for i, r in enumerate(library):
            if r["name"] == room_data["name"]:
                library[i] = room_data
                _save_library(library)
                return library, html.Span(f"✅ '{room_data['name']}' mis à jour.", style={"color": "#166534"})
        library.append(room_data)
        _save_library(library)
        return library, html.Span(f"✅ '{room_data['name']}' sauvegardé.", style={"color": "#166534"})

    if "btn-delete-room" in trigger:
        if not selected_name:
            return no_update, html.Span("⚠️ Sélectionnez une pièce à supprimer.", style={"color": AMBER})
        library = [r for r in _load_library() if r["name"] != selected_name]
        _save_library(library)
        return library, html.Span(f"🗑️ '{selected_name}' supprimé.", style={"color": "#6B6560"})

    return no_update, no_update


@app.callback(
    Output("rooms-library-store", "data", allow_duplicate=True),
    Input("btn-v2", "n_clicks"),
    prevent_initial_call=True,
)
def load_library_on_tab(_):
    return _load_library()


@app.callback(
    Output("room-library-dropdown", "options"),
    Input("rooms-library-store", "data"),
)
def update_dropdown_options(rooms):
    return [{"label": r["name"], "value": r["name"]} for r in rooms]


@app.callback(
    Output("room-points-store", "data", allow_duplicate=True),
    Output("canvas-x", "value", allow_duplicate=True),
    Output("canvas-y", "value", allow_duplicate=True),
    Output("lz2", "value", allow_duplicate=True),
    Input("room-library-dropdown", "value"),
    State("rooms-library-store", "data"),
    prevent_initial_call=True,
)
def load_room_from_library(name, rooms):
    if not name:
        return no_update, no_update, no_update, no_update
    for r in rooms:
        if r["name"] == name:
            return (
                {"points": r["points"], "closed": r.get("closed", True)},
                r.get("canvas_x", 8),
                r.get("canvas_y", 6),
                r.get("lz", 2.5),
            )
    return no_update, no_update, no_update, no_update


# ─────────────────────────────────────────────────────────────────────────────
# Tutorial
# ─────────────────────────────────────────────────────────────────────────────

TUTORIAL_STEPS = [
    {"step": "1 / 6", "title": "Dimensions de la pièce",
     "text": "Réglez les dimensions de votre pièce avec ces curseurs : Longueur X, Largeur Y, Hauteur Z."},
    {"step": "2 / 6", "title": "Sélecteur de mode",
     "text": "Choisissez le mode de résonance à visualiser (m, n, p). Les modes axiaux ★★★ sont les plus problématiques."},
    {"step": "3 / 6", "title": "Vue 2D",
     "text": "Cliquez ici pour placer vos enceintes A et B. La carte montre les zones de haute (+1, ambre) et basse (−1, bleue) pression."},
    {"step": "4 / 6", "title": "Vue 3D",
     "text": "Visualisation 3D du champ de pression dans tout le volume de la pièce."},
    {"step": "5 / 6", "title": "Bouton Analyse",
     "text": "Cliquez ici pour obtenir des recommandations acoustiques personnalisées basées sur la position de vos enceintes et le mode sélectionné."},
    {"step": "6 / 6", "title": "Table des modes",
     "text": "Tous les modes de résonance de la pièce classés par importance. Les modes axiaux ★★★ à basse fréquence sont les plus problématiques."},
]


@app.callback(
    Output("tutorial-step", "data"),
    Input("tutorial-seen", "data"),
    prevent_initial_call=False,
)
def tutorial_init(seen):
    if not seen:
        return 0
    return -1


@app.callback(
    Output("tutorial-step", "data", allow_duplicate=True),
    Output("tutorial-seen", "data"),
    Input("tutorial-btn", "n_clicks"),
    Input("tutorial-skip", "n_clicks"),
    Input("tutorial-next", "n_clicks"),
    Input("tutorial-prev", "n_clicks"),
    State("tutorial-step", "data"),
    prevent_initial_call=True,
)
def tutorial_navigate(btn_n, skip_n, next_n, prev_n, step):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update
    trigger = ctx.triggered[0]["prop_id"]
    if "tutorial-btn" in trigger:
        return 0, False
    if "tutorial-skip" in trigger:
        return -1, True
    if "tutorial-next" in trigger:
        step = step if step is not None else 0
        if step >= len(TUTORIAL_STEPS) - 1:
            return -1, True
        return step + 1, no_update
    if "tutorial-prev" in trigger:
        step = step if step is not None else 0
        return max(0, step - 1), no_update
    return no_update, no_update


@app.callback(
    Output("tutorial-overlay", "style"),
    Output("tutorial-step-indicator", "children"),
    Output("tutorial-title", "children"),
    Output("tutorial-text", "children"),
    Output("tutorial-prev", "style"),
    Output("tutorial-next", "children"),
    Input("tutorial-step", "data"),
)
def tutorial_display(step):
    if step is None or step < 0:
        return {"display": "none"}, no_update, no_update, no_update, no_update, no_update
    s = TUTORIAL_STEPS[min(step, len(TUTORIAL_STEPS) - 1)]
    prev_style = {"visibility": "hidden", "marginTop": "10px"} if step == 0 else {"marginTop": "10px"}
    next_label = "Terminer ✓" if step >= len(TUTORIAL_STEPS) - 1 else "Suivant →"
    return (
        {"display": "block"},
        s["step"],
        s["title"],
        s["text"],
        prev_style,
        next_label,
    )


app.clientside_callback(
    """
    function(step) {
        var targets = [
            'section-dimensions',
            'section-mode-selector',
            'section-2d-view',
            'section-3d-view',
            'section-analyse',
            'section-modes-table'
        ];
        targets.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.classList.remove('tutorial-highlight');
        });
        if (step !== null && step !== undefined && step >= 0 && step < targets.length) {
            var el = document.getElementById(targets[step]);
            if (el) {
                el.classList.add('tutorial-highlight');
                el.scrollIntoView({behavior: 'smooth', block: 'nearest'});
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("tutorial-hl-dummy", "children"),
    Input("tutorial-step", "data"),
)


server = app.server

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
