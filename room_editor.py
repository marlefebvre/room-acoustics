"""
room_editor.py — Éditeur de plan de pièce sur papier millimétré
"""
import numpy as np
import plotly.graph_objects as go


def make_grid_figure(points, closed, Lx_max=15, Ly_max=12, step=0.5):
    """
    Génère la figure du papier millimétré.
    Inclut une grille invisible de points cliquables (0.5m) pour capturer les clics Plotly.
    """
    AMBER  = "#B45309"
    CREAM  = "#FAF9F6"
    DARK   = "#1A1814"
    GRID_M = "#D8D3C8"
    GRID_dm= "#EDE9E1"

    fig = go.Figure()

    # ── Grille invisible cliquable (OBLIGATOIRE pour que clickData fonctionne) ──
    # Sans points de données, Plotly ne déclenche pas clickData sur zone vide
    xs_click = np.arange(0, Lx_max + step, step)
    ys_click = np.arange(0, Ly_max + step, step)
    X_c, Y_c = np.meshgrid(xs_click, ys_click)
    fig.add_trace(go.Scatter(
        x=X_c.flatten(), y=Y_c.flatten(),
        mode="markers",
        marker=dict(size=28, color="rgba(0,0,0,0)", line=dict(width=0)),
        showlegend=False,
        hovertemplate="X=%{x:.1f}m  Y=%{y:.1f}m<extra>Cliquer pour placer un point</extra>",
        name="",
    ))

    # ── Grilles visuelles (shapes) ──────────────────────────────────────
    for x in np.arange(0, Lx_max + 0.01, 0.5):
        fig.add_shape(type="line", x0=round(x,1), y0=0, x1=round(x,1), y1=Ly_max,
                      line=dict(color=GRID_dm, width=0.8))
    for y in np.arange(0, Ly_max + 0.01, 0.5):
        fig.add_shape(type="line", x0=0, y0=round(y,1), x1=Lx_max, y1=round(y,1),
                      line=dict(color=GRID_dm, width=0.8))
    for x in range(0, int(Lx_max) + 1):
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=Ly_max,
                      line=dict(color=GRID_M, width=1.5))
    for y in range(0, int(Ly_max) + 1):
        fig.add_shape(type="line", x0=0, y0=y, x1=Lx_max, y1=y,
                      line=dict(color=GRID_M, width=1.5))

    # ── Murs ────────────────────────────────────────────────────────────
    if len(points) >= 2:
        pts = points + ([points[0]] if closed else [])
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=AMBER, width=3),
            showlegend=False, hoverinfo="skip"))

        for i in range(len(pts) - 1):
            mx = (pts[i]["x"] + pts[i+1]["x"]) / 2
            my = (pts[i]["y"] + pts[i+1]["y"]) / 2
            dx = pts[i+1]["x"] - pts[i]["x"]
            dy = pts[i+1]["y"] - pts[i]["y"]
            length = round((dx**2 + dy**2) ** 0.5, 2)
            fig.add_annotation(x=mx, y=my, text=f"<b>{length:.1f}m</b>",
                showarrow=False, font=dict(size=10, color=AMBER, family="Outfit"),
                bgcolor="rgba(255,255,255,0.9)", bordercolor=AMBER,
                borderwidth=1, borderpad=2)

    # ── Sommets ─────────────────────────────────────────────────────────
    if points:
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in points], y=[p["y"] for p in points],
            mode="markers+text",
            text=[str(i+1) for i in range(len(points))],
            textposition="top right",
            textfont=dict(size=9, color=DARK),
            marker=dict(size=10, color=AMBER, line=dict(width=2, color="white")),
            showlegend=False,
            hovertemplate=[f"Point {i+1}: ({p['x']:.1f}, {p['y']:.1f})<extra></extra>"
                           for i, p in enumerate(points)]))

    # ── Remplissage pièce fermée ────────────────────────────────────────
    if closed and len(points) >= 3:
        xs_f = [p["x"] for p in points] + [points[0]["x"]]
        ys_f = [p["y"] for p in points] + [points[0]["y"]]
        fig.add_trace(go.Scatter(x=xs_f, y=ys_f, fill="toself",
            fillcolor="rgba(180,83,9,0.08)", line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False))

    fig.update_layout(
        xaxis=dict(range=[-0.1, Lx_max + 0.1], dtick=1,
                   showgrid=False, zeroline=False, title="X (m)",
                   fixedrange=True),   # ← désactive zoom axe X
        yaxis=dict(range=[-0.1, Ly_max + 0.1], dtick=1,
                   showgrid=False, zeroline=False, title="Y (m)",
                   scaleanchor="x", fixedrange=True),  # ← désactive zoom axe Y
        plot_bgcolor=CREAM, paper_bgcolor="#FFFFFF",
        font=dict(color=DARK, family="Outfit, sans-serif"),
        margin=dict(l=40, r=20, t=10, b=40),
        autosize=True,
        showlegend=False,
        dragmode=False,
        clickmode="event",
    )
    return fig



def polygon_bounding_box(points):
    """Retourne (Lx, Ly) de la bounding box du polygone."""
    if not points:
        return 6.0, 4.0
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return round(max(xs) - min(xs), 1), round(max(ys) - min(ys), 1)


def polygon_area(points):
    """Aire du polygone (formule de Shoelace)."""
    n = len(points)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points[i]["x"] * points[j]["y"]
        area -= points[j]["x"] * points[i]["y"]
    return abs(area) / 2


def is_complex_shape(points):
    """Retourne True si la forme n'est pas rectangulaire simple."""
    if len(points) != 4:
        return True
    xs = sorted([p["x"] for p in points])
    ys = sorted([p["y"] for p in points])
    return not (xs[0] == xs[1] and xs[2] == xs[3] and ys[0] == ys[1] and ys[2] == ys[3])
