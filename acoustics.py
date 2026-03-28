"""
acoustics.py — Calculs physiques des modes de résonance
"""
import numpy as np

SPEED_OF_SOUND = 343.0

MODE_INFO = {
    "Axial": {
        "stars": "★★★",
        "importance": "Critique",
        "color": "#B45309",
        "comment": "Très énergétique — crée des booms et creux importants",
    },
    "Tangentiel": {
        "stars": "★★☆",
        "importance": "Modéré",
        "color": "#92400E",
        "comment": "Énergie modérée — colorations perceptibles",
    },
    "Oblique": {
        "stars": "★☆☆",
        "importance": "Faible",
        "color": "#9A948E",
        "comment": "Peu d'impact sur l'écoute",
    },
}


def room_mode_frequency(m, n, p, Lx, Ly, Lz):
    return (SPEED_OF_SOUND / 2) * np.sqrt(
        (m / Lx) ** 2 + (n / Ly) ** 2 + (p / Lz) ** 2
    )


def compute_modes(Lx, Ly, Lz, max_order=3):
    modes = []
    for m in range(max_order + 1):
        for n in range(max_order + 1):
            for p in range(max_order + 1):
                if m == n == p == 0:
                    continue
                freq = room_mode_frequency(m, n, p, Lx, Ly, Lz)
                t = mode_type(m, n, p)
                modes.append({
                    "m": m, "n": n, "p": p,
                    "freq": round(freq, 1),
                    "type": t,
                    **MODE_INFO[t],
                })
    return sorted(modes, key=lambda x: x["freq"])


def mode_type(m, n, p):
    non_zero = sum([m > 0, n > 0, p > 0])
    if non_zero == 1: return "Axial"
    elif non_zero == 2: return "Tangentiel"
    else: return "Oblique"


def pressure_field_2d(m, n, Lx, Ly, resolution=100):
    x = np.linspace(0, Lx, resolution)
    y = np.linspace(0, Ly, resolution)
    X, Y = np.meshgrid(x, y)
    Px = np.cos(m * np.pi * X / Lx) if m > 0 else np.ones_like(X)
    Py = np.cos(n * np.pi * Y / Ly) if n > 0 else np.ones_like(Y)
    return X, Y, Px * Py


def pressure_field_3d(m, n, p, Lx, Ly, Lz, resolution=30):
    x = np.linspace(0, Lx, resolution)
    y = np.linspace(0, Ly, resolution)
    z = np.linspace(0, Lz, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    Px = np.cos(m * np.pi * X / Lx) if m > 0 else np.ones_like(X)
    Py = np.cos(n * np.pi * Y / Ly) if n > 0 else np.ones_like(Y)
    Pz = np.cos(p * np.pi * Z / Lz) if p > 0 else np.ones_like(Z)
    return X, Y, Z, Px * Py * Pz


def speaker_coupling(sx, sy, m, n, Lx, Ly):
    Px = np.cos(m * np.pi * sx / Lx) if m > 0 else 1.0
    Py = np.cos(n * np.pi * sy / Ly) if n > 0 else 1.0
    return abs(Px * Py)
