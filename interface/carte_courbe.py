"""Carte à courbe : l'EAR sur les dernières secondes, avec la ligne du seuil.

================================================================================
CE QUE FAIT CE FICHIER (BLOC 4 : COURBE TEMPORELLE EAR)
================================================================================
MONTRE l'ANALYSE TEMPORELLE en direct : courbe EAR + ligne de seuil.

CE QU'ON VOIT :
- Clignement = creux BREF (remonte vite au-dessus du seuil)
- Endormissement = creux LONG (reste SOUS le seuil > 1.5s)
- Fatigue = creux RÉPÉTITIFS (oscillations autour du seuil)

FONCTIONNEMENT :
- deque = fenêtre glissante sur 10s (config.duree_courbe)
- À chaque image : ajouter_point(instant, EAR) → purge vieux points
- dessiner() : trace polyline + ligne horizontale seuil (orange)
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import cv2
import numpy as np

from interface.cartes import Carte, Rectangle
from interface.dessin import Theme
from noyau.modeles import SortieTraitement

EAR_MAXIMUM_AFFICHE = 0.5   # valeur de l'EAR correspondant au haut de la courbe


class CarteCourbe(Carte):
    """Courbe de l'EAR sur une durée glissante."""

    def __init__(self, titre: str, rectangle: Rectangle, duree: float) -> None:
        super().__init__(titre, rectangle)
        self._duree = duree
        # APPORT : `deque` permet de retirer les plus vieux points rapidement en tête de file.
        self._points: Deque[Tuple[float, float]] = deque()

    def ajouter_point(self, instant: float, ear: float) -> None:
        """Ajoute un point (instant, EAR) et retire ceux qui sont plus vieux que la durée affichée.

        POURQUOI : conserver exactement l'historique de quelques secondes visible à l'écran.
        """
        self._points.append((instant, ear))
        while self._points and instant - self._points[0][0] > self._duree:
            self._points.popleft()

    def dessiner(self, canevas: np.ndarray, donnees: SortieTraitement) -> None:
        """Trace la courbe de l'EAR et la ligne horizontale du seuil.

        POURQUOI : illustrer l'analyse temporelle : le même creux est un clignement s'il est
        bref et un endormissement s'il dure plus d'environ 1,5 seconde.
        """
        self._cadre(canevas)
        x1, y1, x2, y2 = self._rectangle
        zone_x1, zone_x2 = x1 + 18, x2 - 18
        zone_y1, zone_y2 = y1 + 40, y2 - 14
        hauteur_zone = zone_y2 - zone_y1

        def vers_y(ear: float) -> int:
            """Convertit une valeur d'EAR en ordonnée de pixel (haut = grand EAR)."""
            ratio = max(0.0, min(1.0, ear / EAR_MAXIMUM_AFFICHE))
            return int(zone_y2 - ratio * hauteur_zone)

        if donnees.seuil_ear > 0:
            y_seuil = vers_y(donnees.seuil_ear)
            cv2.line(canevas, (zone_x1, y_seuil), (zone_x2, y_seuil), Theme.ORANGE, 1, cv2.LINE_AA)
            self._dessinateur.texte(canevas, "seuil", (zone_x2, y_seuil - 4), 0.38, Theme.ORANGE, 1, "droite")

        if len(self._points) < 2:
            return
        instant_final = self._points[-1][0]
        trace = np.array(
            [
                (zone_x1 + (instant - (instant_final - self._duree)) / self._duree * (zone_x2 - zone_x1),
                 vers_y(ear))
                for instant, ear in self._points
            ],
            dtype=np.int32,
        )
        cv2.polylines(canevas, [trace.reshape(-1, 1, 2)], False, Theme.ACCENT, 2, cv2.LINE_AA)
