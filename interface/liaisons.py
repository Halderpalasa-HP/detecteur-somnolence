"""Liaisons du maillage du visage, regroupées par élément (yeux, lèvres, contour…).

================================================================================
CE QUE FAIT CE FICHIER (BLOC 4 : STRUCTURE DU MAILLAGE)
================================================================================
DIT QUELS POINTS RELIER pour dessiner chaque élément du visage.
MediaPipe fournit 478 points mais PAS les connexions → il faut les définir.

PROBLÈME : MediaPipe a CHANGÉ l'emplacement de ces constantes selon les versions.
SOLUTION : Essaie API récente (Tasks) → sinon API legacy (mp.solutions) → sinon masque limité.

GROUPES DÉFINIS (nom interne → constante MediaPipe) :
- maillage     → TESSELATION    (triangles complets, ~1500 connexions)
- ovale        → FACE_OVAL      (contour du visage)
- oeil_gauche  → LEFT_EYE       (contour œil gauche)
- oeil_droit   → RIGHT_EYE      (contour œil droit)
- sourcil_gauche → LEFT_EYEBROW
- sourcil_droit  → RIGHT_EYEBROW
- iris_gauche  → LEFT_IRIS      (dispo seulement modèle 478 pts)
- iris_droit   → RIGHT_IRIS
- levres       → LIPS
- nez          → NOSE
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Liaison = Tuple[int, int]

# Nom de groupe utilisé dans notre code -> suffixe du nom de constante MediaPipe.
CORRESPONDANCES = {
    "maillage": "TESSELATION",
    "ovale": "FACE_OVAL",
    "oeil_gauche": "LEFT_EYE",
    "oeil_droit": "RIGHT_EYE",
    "sourcil_gauche": "LEFT_EYEBROW",
    "sourcil_droit": "RIGHT_EYEBROW",
    "iris_gauche": "LEFT_IRIS",
    "iris_droit": "RIGHT_IRIS",
    "levres": "LIPS",
    "nez": "NOSE",
}


class LiaisonsVisage:
    """Fournit les paires de points à relier pour chaque groupe du visage."""

    def __init__(self) -> None:
        self._groupes: Dict[str, List[Liaison]] = {}
        try:
            self._groupes = self._charger_tasks()
        except Exception:
            try:
                self._groupes = self._charger_legacy()
            except Exception:
                print("Avertissement : liaisons du maillage indisponibles, le masque sera limité.")

    def groupe(self, nom: str) -> List[Liaison]:
        """Renvoie la liste des paires (a, b) d'un groupe (liste vide si le groupe est inconnu).

        POURQUOI : offrir au masque une interface stable, quelle que soit la version de MediaPipe.
        """
        return self._groupes.get(nom, [])

    def _charger_tasks(self) -> Dict[str, List[Liaison]]:
        """Lit les connexions de l'API récente (`FaceLandmarksConnections`).

        POURQUOI : c'est la voie recommandée, cohérente avec le modèle `face_landmarker.task`.
        """
        # APPORT : cette classe de MediaPipe Tasks contient les listes de connexions du maillage
        # (chaque connexion possède un début `start` et une fin `end`).
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

        groupes: Dict[str, List[Liaison]] = {}
        for nom, suffixe in CORRESPONDANCES.items():
            connexions = getattr(FaceLandmarksConnections, f"FACE_LANDMARKS_{suffixe}", [])
            groupes[nom] = [(c.start, c.end) for c in connexions]
        if not groupes["maillage"]:
            raise AttributeError("Connexions Tasks introuvables.")
        return groupes

    def _charger_legacy(self) -> Dict[str, List[Liaison]]:
        """Lit les constantes de l'ancienne API `mp.solutions.face_mesh`.

        POURQUOI : plan B pour les versions de MediaPipe qui ne proposent pas l'API récente.
        """
        import mediapipe as mp

        face_mesh = mp.solutions.face_mesh
        groupes: Dict[str, List[Liaison]] = {}
        for nom, suffixe in CORRESPONDANCES.items():
            ensemble = getattr(face_mesh, f"FACEMESH_{suffixe}", frozenset())
            groupes[nom] = [(a, b) for a, b in ensemble]
        if not groupes["maillage"]:
            raise AttributeError("Connexions historiques introuvables.")
        return groupes
