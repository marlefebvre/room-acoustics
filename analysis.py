"""
analysis.py — Analyse acoustique rule-based
"""
from acoustics import compute_modes, room_mode_frequency, speaker_coupling

SPEED_OF_SOUND = 343.0
MAX_WALL_DIST = 1.0   # règle : enceinte à max 1m d'un mur
MIN_WALL_DIST = 0.3   # règle : enceinte à min 30cm d'un mur


def wall_distance(x, y, Lx, Ly):
    """Distance minimale à n'importe quel mur (utilisée pour la règle min 30cm)."""
    return min(x, Lx - x, y, Ly - y)


def y_wall_distance(y, Ly):
    """Distance au mur de largeur le plus proche (axe Y — front/arrière).
    La règle max 1m s'applique à cet axe : les enceintes doivent être
    proches d'un mur de largeur, pas d'un mur de côté."""
    return min(y, Ly - y)


def is_valid_speaker_pos(x, y, Lx, Ly):
    """
    Position valide si (règle purement sur l'axe Y, peu importe X) :
    - à plus de 30cm d'un mur de largeur Y (MIN_WALL_DIST)
    - à moins de 1m d'un mur de largeur Y (MAX_WALL_DIST)
    """
    d_y = y_wall_distance(y, Ly)
    return MIN_WALL_DIST <= d_y <= MAX_WALL_DIST


def suggest_speaker_position(s, m, n, Lx, Ly, step=0.1, max_move=1.0):
    """
    Cherche la meilleure position dans ±1m avec couplage minimal.
    - Si n == 0 : mode indépendant de Y → recherche sur X seulement, Y fixe
    - Si m == 0 : mode indépendant de X → recherche sur Y seulement, X fixe
    - Sinon     : recherche 2D complète
    Contrainte : position valide (30cm–1m d'un mur de largeur Y).
    """
    import numpy as np
    best_pos = None
    best_coupling = speaker_coupling(s["x"], s["y"], m, n, Lx, Ly)

    sx = round(s["x"], 1)
    sy = round(s["y"], 1)

    x_min = max(step, round(sx - max_move, 1))
    x_max = min(round(Lx - step, 1), round(sx + max_move, 1))
    y_min = max(step, round(sy - max_move, 1))
    y_max = min(round(Ly - step, 1), round(sy + max_move, 1))

    if n == 0:
        # Mode indépendant de Y : ne varier que X, Y reste fixe
        xs = np.arange(x_min, x_max + 0.005, step)
        ys = [sy]
    elif m == 0:
        # Mode indépendant de X : ne varier que Y, X reste fixe
        xs = [sx]
        ys = np.arange(y_min, y_max + 0.005, step)
    else:
        # Mode 2D : recherche complète
        xs = np.arange(x_min, x_max + 0.005, step)
        ys = np.arange(y_min, y_max + 0.005, step)

    for x in xs:
        for y in ys:
            x, y = round(x, 1), round(y, 1)
            if x == sx and y == sy:
                continue
            if not is_valid_speaker_pos(x, y, Lx, Ly):
                continue
            c = speaker_coupling(x, y, m, n, Lx, Ly)
            if c < best_coupling:
                best_coupling = c
                best_pos = (x, y)

    return best_pos, round(best_coupling, 2)


def fmt_cm(val_m):
    return f"{val_m:.1f}m ({int(round(val_m * 100))}cm)"


def analyse_room(Lx, Ly, Lz, m, n, p, speakers):
    problems, impacts, recommendations = [], [], []
    priorities = []
    suggestions = {}

    freq = room_mode_frequency(m, n, p, Lx, Ly, Lz) if not (m == n == p == 0) else 0
    modes = compute_modes(Lx, Ly, Lz, max_order=4)
    s1 = speakers.get("s1")
    s2 = speakers.get("s2")

    # ── Mode sélectionné ───────────────────────────────────────────────
    if freq > 0:
        nz = sum([m > 0, n > 0, p > 0])
        if nz == 1:
            problems.append(
                f"🔴 Le mode ({m},{n},{p}) à {freq:.1f} Hz est un mode AXIAL — le plus problématique. "
                "Très énergétique, il crée des zones de boom ou de creux importants. "
                "En déplaçant la tête de quelques centimètres, le son peut changer radicalement."
            )
        elif nz == 2:
            problems.append(
                f"🟡 Le mode ({m},{n},{p}) à {freq:.1f} Hz est TANGENTIEL — énergie modérée. "
                "Il crée des colorations sonores perceptibles mais moins prononcées qu'un mode axial."
            )
        else:
            impacts.append(
                f"✅ Le mode ({m},{n},{p}) à {freq:.1f} Hz est OBLIQUE — faible impact. "
                "Implique les 6 surfaces, énergie limitée."
            )

    # ── Modes axials critiques ─────────────────────────────────────────
    axial = [mo for mo in modes if mo["type"] == "Axial" and mo["freq"] < 300][:3]
    if axial:
        fs = " / ".join([f"{mo['freq']:.0f} Hz" for mo in axial])
        problems.append(
            f"🔴 Modes les plus problématiques de cette pièce : {fs} (axials). "
            "Ce sont ces fréquences qui créent les 'booms' graves. "
            "Toute note musicale proche sera amplifiée ou annulée selon votre position."
        )
        impacts.append(
            f"👂 À {axial[0]['freq']:.0f} Hz, l'écart entre ventre et nœud peut dépasser 20 dB "
            "— différence de volume très perceptible."
        )

    # ── Ratio dimensions ───────────────────────────────────────────────
    dims = sorted([Lx, Ly, Lz])
    r1 = dims[1] / dims[0]
    if abs(r1 - 1.0) < 0.1:
        problems.append(
            f"⚠️ Deux dimensions quasi-identiques ({dims[0]:.1f}m / {dims[1]:.1f}m) → modes dégénérés, "
            "accumulation de résonances à la même fréquence."
        )
        recommendations.append(
            f"Modifier une dimension d'au moins 15% (ex: {dims[1]:.1f}m → {dims[1]*1.15:.1f}m)."
        )
    if abs(r1 - 2.0) < 0.12:
        problems.append(
            "⚠️ Une dimension est le double d'une autre → modes alignés, boom doublé."
        )

    # ── Fréquence de Schroeder ─────────────────────────────────────────
    volume = Lx * Ly * Lz
    surf = 2 * (Lx*Ly + Lx*Lz + Ly*Lz)
    tr60 = max(0.2, min(0.16 * volume / (0.2 * surf), 1.5))
    schroeder = 2000 * (tr60 / volume) ** 0.5
    problems.append(
        f"📐 Fréquence de Schroeder : ~{schroeder:.0f} Hz. "
        f"En dessous → zone modale (ondes stationnaires dominantes). "
        f"Au-dessus → comportement diffus et prévisible."
    )

    # ── Enceintes ──────────────────────────────────────────────────────
    for label, s, key in [("Enceinte A", s1, "s1"), ("Enceinte B", s2, "s2")]:
        if not s:
            continue

        c = speaker_coupling(s["x"], s["y"], m, n, Lx, Ly)
        wd = wall_distance(s["x"], s["y"], Lx, Ly)       # dist min à tout mur
        wd_y = y_wall_distance(s["y"], Ly)                # dist au mur de largeur (Y)
        wd_cm = int(wd * 100)
        wd_y_cm = int(wd_y * 100)
        pos_str = f"({fmt_cm(s['x'])}, {fmt_cm(s['y'])})"

        # ── Règle distance mur Y ───────────────────────────────────────
        if wd_y < MIN_WALL_DIST:
            problems.append(
                f"⚠️ {label} {pos_str} : trop proche d'un mur de largeur ({wd_y_cm}cm). "
                "Amplification des graves d'environ +6 dB — son boomy et non contrôlé."
            )
            recommendations.append(f"Éloigner {label} à 30–100cm d'un mur de largeur (axe Y).")
            priorities.append(f"{label} trop proche du mur Y ({wd_y_cm}cm)")

        elif wd_y > MAX_WALL_DIST:
            # Trop loin des murs de LARGEUR (axe Y) — règle acoustique principale
            problems.append(
                f"⚠️ {label} {pos_str} : trop loin des murs de largeur ({wd_y_cm}cm). "
                f"La règle impose max {int(MAX_WALL_DIST*100)}cm d'un mur de largeur — "
                "les basses perdent leur soutien acoustique."
            )
            # Suggérer 60cm du mur Y le plus proche
            if s["y"] <= Ly / 2:
                sx, sy = round(s["x"], 1), 0.6
                mur = "avant"
            else:
                sx, sy = round(s["x"], 1), round(Ly - 0.6, 1)
                mur = "arrière"
            recommendations.append(
                f"Rapprocher {label} du mur {mur} (largeur) → Y={sy:.1f}m (60cm du mur)."
            )
            priorities.append(
                f"{label} trop loin des murs de largeur ({wd_y_cm}cm) — doit être à moins de 1m d'un mur de largeur"
            )

        # ── Règle couplage avec le mode ────────────────────────────────
        if c > 0.7 and freq > 0:
            better, bc = suggest_speaker_position(s, m, n, Lx, Ly)
            if better:
                suggestions[key] = {"x": better[0], "y": better[1], "label": label}
                dx_cm = int(round((better[0] - s["x"]) * 100))
                dy_cm = int(round((better[1] - s["y"]) * 100))
                dir_x = f"{abs(dx_cm)}cm {'→' if dx_cm > 0 else '←'}" if dx_cm != 0 else ""
                dir_y = f"{abs(dy_cm)}cm {'↑' if dy_cm > 0 else '↓'}" if dy_cm != 0 else ""
                mouvement = " + ".join(filter(None, [dir_x, dir_y]))
                problems.append(
                    f"🔴 {label} {pos_str} : couplage fort ({c:.2f}) — "
                    f"excite fortement la résonance à {freq:.0f} Hz."
                )
                recommendations.append(
                    f"Déplacer {label} de {mouvement} → X={better[0]:.1f}m, Y={better[1]:.1f}m "
                    f"(couplage {c:.2f} → {bc:.2f}). Marqueur pointillé sur le plan."
                )
                priorities.append(
                    f"{label} : {mouvement} → ({better[0]:.1f}m, {better[1]:.1f}m)"
                )
            else:
                problems.append(
                    f"🔴 {label} {pos_str} : couplage fort ({c:.2f}) — "
                    f"aucune meilleure position trouvée dans ±1m respectant les règles."
                )
        elif c < 0.3 and freq > 0:
            impacts.append(
                f"✅ {label} {pos_str} : bonne position (couplage {c:.2f}) — "
                f"excite peu ce mode à {freq:.0f} Hz."
            )
        elif freq > 0:
            impacts.append(
                f"🟡 {label} {pos_str} : couplage intermédiaire ({c:.2f})."
            )

    # ── Distance inter-enceintes ───────────────────────────────────────
    if s1 and s2:
        dist = ((s1["x"]-s2["x"])**2 + (s1["y"]-s2["y"])**2) ** 0.5
        if dist < 1.0:
            problems.append(
                f"⚠️ Enceintes trop proches ({int(dist*100)}cm) — image stéréo inexistante. "
                "Minimum recommandé : 1.5m entre les deux."
            )
            priorities.append(f"Enceintes trop proches ({int(dist*100)}cm)")

    # ── Recommandations générales ──────────────────────────────────────
    recommendations.append(
        f"Éviter le centre exact ({fmt_cm(Lx/2)}, {fmt_cm(Ly/2)}) — cumul de modes."
    )
    recommendations.append(
        "Mesurer avec REW (Room EQ Wizard, gratuit) pour voir la réponse réelle en fréquence."
    )

    return {
        "freq": freq,
        "priorities": priorities,
        "problemes": problems,
        "impact": impacts,
        "recommandations": recommendations,
        "suggestions": suggestions,
    }
