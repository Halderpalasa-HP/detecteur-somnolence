"""Tests des mesures géométriques : distance, angle, EAR et MAR.

Lancer avec :  pytest tests/        (depuis la racine du projet)
Les valeurs attendues viennent des exemples chiffrés du rapport de projet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesures.mesures_visage import CalculateurEAR, CalculateurMAR, Geometrie
from noyau.modeles import ResultatVisage


def fabriquer_visage(points_utiles: dict) -> ResultatVisage:
    """Crée un visage de 478 points nuls, puis place uniquement les points utiles au test."""
    points = [(0.0, 0.0)] * 478
    for indice, coordonnees in points_utiles.items():
        points[indice] = coordonnees
    return ResultatVisage(coordonnees=points, largeur=640, hauteur=480)


def placer_oeil(indices, p1, p2, p3, p4, p5, p6) -> dict:
    """Associe les six points (p1 à p6) d'un œil à leurs indices MediaPipe."""
    return dict(zip(indices, (p1, p2, p3, p4, p5, p6)))


def test_distance_triangle_3_4_5() -> None:
    assert Geometrie.distance((0, 0), (3, 4)) == 5.0


def test_normaliser_angle() -> None:
    assert Geometrie.normaliser_angle(190.0) == -170.0
    assert Geometrie.normaliser_angle(-190.0) == 170.0
    assert Geometrie.normaliser_angle(30.0) == 30.0


def test_ear_oeil_ouvert_et_ferme() -> None:
    # Exemple du rapport : œil ouvert -> EAR ≈ 0,267 ; œil fermé -> EAR ≈ 0,077.
    ouvert = placer_oeil(CalculateurEAR.INDICES_OEIL_DROIT, (250, 200), (268, 190), (292, 189),
                         (310, 203), (293, 205), (267, 206))
    ferme = placer_oeil(CalculateurEAR.INDICES_OEIL_DROIT, (250, 200), (268, 197), (292, 196),
                        (310, 203), (293, 201), (267, 201))
    ear_ouvert = CalculateurEAR()._ear_un_oeil(fabriquer_visage(ouvert), CalculateurEAR.INDICES_OEIL_DROIT)
    ear_ferme = CalculateurEAR()._ear_un_oeil(fabriquer_visage(ferme), CalculateurEAR.INDICES_OEIL_DROIT)
    assert abs(ear_ouvert - 0.267) < 0.002
    assert abs(ear_ferme - 0.077) < 0.002


def test_ear_moyenne_des_deux_yeux() -> None:
    points = {}
    for indices in (CalculateurEAR.INDICES_OEIL_DROIT, CalculateurEAR.INDICES_OEIL_GAUCHE):
        points.update(placer_oeil(indices, (250, 200), (268, 190), (292, 189), (310, 203),
                                  (293, 205), (267, 206)))
    assert abs(CalculateurEAR().calculer(fabriquer_visage(points)) - 0.267) < 0.002


def test_mar_bouche_fermee_et_baillement() -> None:
    # Largeur de bouche 40 px. Fermée : hauteurs 2, 3, 2 -> 7/120 ≈ 0,058. Bâillement : 22, 26, 22 -> 0,583.
    def bouche(h1: float, h2: float, h3: float) -> ResultatVisage:
        return fabriquer_visage({
            78: (300, 360), 308: (340, 360),
            82: (310, 360 - h1 / 2), 87: (310, 360 + h1 / 2),
            13: (320, 360 - h2 / 2), 14: (320, 360 + h2 / 2),
            312: (330, 360 - h3 / 2), 317: (330, 360 + h3 / 2),
        })
    assert abs(CalculateurMAR().calculer(bouche(2, 3, 2)) - 0.058) < 0.002
    assert abs(CalculateurMAR().calculer(bouche(22, 26, 22)) - 0.583) < 0.002


def test_ear_protege_contre_la_division_par_zero() -> None:
    visage = fabriquer_visage({})   # tous les points confondus : largeur d'œil nulle
    assert CalculateurEAR().calculer(visage) == 0.0
