"""
polygon_acoustics.py — Solveur acoustique FDM pour pièces de forme arbitraire.

Méthode : Différences Finies (FDM) sur l'équation de Helmholtz 2D
  ∇²p + k²p = 0  dans le domaine Ω (polygone)
  dp/dn = 0       sur ∂Ω (murs rigides — condition de Neumann)

La discrétisation du Laplacien avec conditions de Neumann donne un problème
aux valeurs propres : L·v = -k²·v
Les fréquences propres sont : f = (c/2π) · sqrt(k²)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from matplotlib.path import Path

SPEED_OF_SOUND = 343.0


def _build_mask(points, x_grid, y_grid):
    """
    Masque booléen : True si le point (x,y) est à l'intérieur du polygone.
    Utilise matplotlib.path.Path.contains_points (ray casting).
    """
    nx, ny = len(x_grid), len(y_grid)
    X, Y = np.meshgrid(x_grid, y_grid)  # shape (ny, nx)
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])

    poly_path = Path([(p["x"], p["y"]) for p in points])
    mask = poly_path.contains_points(grid_pts, radius=1e-6)
    return mask.reshape(ny, nx)


def _build_laplacian(mask, dx, dy):
    """
    Construit la matrice Laplacien creuse avec conditions de Neumann sur les bords.

    Schéma 5-points :
    L[i,j] = 1/dx² (si voisin droit/gauche intérieur)
    L[i,j] = 1/dy² (si voisin haut/bas intérieur)
    L[i,i] = -(somme des voisins intérieurs)/dx² ou dy²

    Condition Neumann : si voisin extérieur → contribution nulle
    (point fantôme = point courant → termes s'annulent)
    """
    ny, nx = mask.shape
    interior_flat = np.where(mask.ravel())[0]
    n = len(interior_flat)

    # Mapping indice global → indice local (intérieur)
    g2l = np.full(nx * ny, -1, dtype=np.int32)
    g2l[interior_flat] = np.arange(n)

    rows, cols, vals = [], [], []

    for gidx in interior_flat:
        iloc = g2l[gidx]
        iy, ix = divmod(int(gidx), nx)
        diag = 0.0

        # 4 voisins : gauche, droite, bas, haut
        for dix, diy, h2 in [(-1,0,dx*dx),(1,0,dx*dx),(0,-1,dy*dy),(0,1,dy*dy)]:
            nix, niy = ix + dix, iy + diy
            if 0 <= nix < nx and 0 <= niy < ny:
                ngidx = niy * nx + nix
                if mask.ravel()[ngidx]:          # voisin intérieur
                    jloc = g2l[ngidx]
                    rows.append(iloc)
                    cols.append(jloc)
                    vals.append(1.0 / h2)
                    diag -= 1.0 / h2
                # else : Neumann → contribution nulle (ghost = self → annulation)
            # else : hors grille → Neumann automatique

        rows.append(iloc)
        cols.append(iloc)
        vals.append(diag)

    L = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return L, interior_flat, g2l


def compute_polygon_modes(points, Lz, resolution=25, n_modes=20):
    """
    Calcule les modes acoustiques 2D pour un polygone arbitraire.

    Paramètres
    ----------
    points     : liste de dict {"x": float, "y": float}
    Lz         : hauteur de la pièce (m) — pour les modes axiaux Z
    resolution : points par mètre (25 = précision ~4cm)
    n_modes    : nombre de modes à calculer

    Retourne
    --------
    modes_2d : liste de dict {freq, type, stars, color, comment, mode_idx}
    pressure_fields : liste de tableaux 2D (un par mode)
    grid_info : dict avec x_grid, y_grid, mask
    """
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    nx = max(20, int((x1 - x0) * resolution) + 1)
    ny = max(20, int((y1 - y0) * resolution) + 1)

    x_grid = np.linspace(x0, x1, nx)
    y_grid = np.linspace(y0, y1, ny)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    mask = _build_mask(points, x_grid, y_grid)
    n_interior = mask.sum()

    if n_interior < 30:
        return [], [], {}

    L, interior_flat, g2l = _build_laplacian(mask, dx, dy)

    # Nombre de modes à demander (limité par la taille du problème)
    k = min(n_modes + 5, n_interior - 2)

    # Résolution du problème aux valeurs propres : L·v = λ·v
    # Le Laplacien discret avec Neumann a des valeurs propres ≤ 0
    # On cherche les valeurs propres les plus proches de 0 (sigma=0)
    try:
        eigenvalues, eigenvectors = eigsh(L, k=k, which='LM', sigma=0.0, tol=1e-6)
    except Exception:
        # Fallback sans shift
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')

    # k² = -eigenvalue (modes propres du Laplacien Neumann)
    k2_vals = -eigenvalues
    # Trier par k² croissant, ignorer le mode 0 (k²≈0, pression uniforme)
    sort_idx = np.argsort(k2_vals)
    k2_sorted = k2_vals[sort_idx]
    evecs_sorted = eigenvectors[:, sort_idx]

    # Fréquences 2D
    valid = k2_sorted > 0.1  # ignore mode constant (k²≈0)
    k2_2d = k2_sorted[valid][:n_modes]
    evecs_2d = evecs_sorted[:, valid][:, :n_modes]

    freq_2d = (SPEED_OF_SOUND / (2 * np.pi)) * np.sqrt(k2_2d)

    # Ajouter les modes axiaux Z (purement verticaux)
    freq_z = []
    for p_z in range(1, 5):
        fz = (SPEED_OF_SOUND / 2) * (p_z / Lz)
        freq_z.append({"freq": round(fz, 1), "p_z": p_z})

    # Classifier les modes 2D
    modes_out = []
    fields_out = []
    ny_g, nx_g = mask.shape

    for i, (f, k2) in enumerate(zip(freq_2d, k2_2d)):
        # Reconstruire le champ de pression sur la grille complète
        P_flat = np.zeros(nx_g * ny_g)
        P_flat[interior_flat] = evecs_2d[:, i]
        P_2d = P_flat.reshape(ny_g, nx_g)
        # Normaliser entre -1 et 1
        pmax = np.abs(P_2d[mask]).max()
        if pmax > 0:
            P_2d /= pmax
        P_2d[~mask] = np.nan  # hors pièce → NaN

        # Type de mode approximatif basé sur la forme spatiale
        mode_type, stars, color, comment = _classify_mode_2d(P_2d, mask, k2, x1-x0, y1-y0)

        modes_out.append({
            "freq": round(float(f), 1),
            "type": mode_type,
            "stars": stars,
            "color": color,
            "comment": comment,
            "k2": float(k2),
            "source": "FDM 2D",
        })
        fields_out.append(P_2d)

    # Ajouter les modes Z dans la liste globale
    for mz in freq_z:
        modes_out.append({
            "freq": mz["freq"],
            "type": "Axial Z",
            "stars": "★★★",
            "color": "#B45309",
            "comment": f"Mode axial vertical (p={mz['p_z']}) — très énergétique",
            "k2": None,
            "source": "Analytique Z",
        })
        fields_out.append(None)

    # Trier par fréquence
    combined = sorted(zip(modes_out, fields_out), key=lambda x: x[0]["freq"])
    modes_out = [c[0] for c in combined]
    fields_out = [c[1] for c in combined]

    grid_info = {
        "x_grid": x_grid.tolist(),
        "y_grid": y_grid.tolist(),
        "mask": mask,
        "interior_flat": interior_flat,
    }

    return modes_out, fields_out, grid_info


def _classify_mode_2d(P, mask, k2, Lx, Ly):
    """
    Classification heuristique du mode 2D basée sur la structure spatiale.
    Compte les zéros (noeuds) dans les directions X et Y.
    """
    # Profil moyen en X et Y
    rows_valid = np.where(mask.any(axis=1))[0]
    cols_valid = np.where(mask.any(axis=0))[0]

    if len(rows_valid) < 2 or len(cols_valid) < 2:
        return "Inconnu", "★☆☆", "#9A948E", "Mode non classifiable"

    prof_x = np.nanmean(np.where(mask, P, np.nan), axis=0)[cols_valid]
    prof_y = np.nanmean(np.where(mask, P, np.nan), axis=1)[rows_valid]

    def count_sign_changes(arr):
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) < 2: return 0
        return np.sum(np.diff(np.sign(arr_clean)) != 0)

    nx_nodes = count_sign_changes(prof_x)
    ny_nodes = count_sign_changes(prof_y)

    if nx_nodes > 0 and ny_nodes == 0:
        return "Axial X", "★★★", "#B45309", "Mode axial horizontal — très énergétique"
    elif ny_nodes > 0 and nx_nodes == 0:
        return "Axial Y", "★★★", "#B45309", "Mode axial latéral — très énergétique"
    elif nx_nodes > 0 and ny_nodes > 0:
        return "Tangentiel", "★★☆", "#92400E", "Mode tangentiel — énergie modérée"
    else:
        return "Oblique", "★☆☆", "#9A948E", "Mode oblique — faible impact"


def pressure_field_polygon(points, mode_idx, modes_list, fields_list, grid_info):
    """
    Retourne le champ de pression 2D pour un mode donné.
    mode_idx : index dans modes_list / fields_list
    """
    if mode_idx >= len(fields_list) or fields_list[mode_idx] is None:
        return None, None, None

    P = fields_list[mode_idx]
    x_grid = np.array(grid_info["x_grid"])
    y_grid = np.array(grid_info["y_grid"])
    return x_grid, y_grid, P


def speaker_coupling_polygon(sx, sy, P, x_grid, y_grid):
    """
    Couplage d'une enceinte avec un mode : valeur de pression normalisée à la position.
    """
    if P is None:
        return 0.0
    x_grid = np.array(x_grid)
    y_grid = np.array(y_grid)
    ix = np.argmin(np.abs(x_grid - sx))
    iy = np.argmin(np.abs(y_grid - sy))
    val = P[iy, ix]
    if np.isnan(val):
        return 0.0
    return abs(float(val))


# ── Cache module-level ─────────────────────────────────────────────────────
_cache = {}

def get_fdm_modes_cached(points, Lz, resolution=25, n_modes=15):
    """
    Calcule et met en cache les modes FDM.
    Recalcule uniquement si la pièce ou ses paramètres changent.
    """
    key = (str(sorted([(p["x"],p["y"]) for p in points])), round(Lz,1), resolution, n_modes)
    if key not in _cache:
        _cache[key] = compute_polygon_modes(points, Lz, resolution, n_modes)
    return _cache[key]
