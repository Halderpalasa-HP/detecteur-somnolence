"""Masque de visage clinique superposé à la vidéo.

================================================================================
CE QUE FAIT CE FICHIER (BLOC 4 : MASQUE VISAGE)
================================================================================
REND VISIBLE ce que voit le modèle : maillage, contours éléments, repères, guides.
Entrée : image caméra + ResultatVisage + Etat + OptionsAffichage
Sortie : image modifiée SUR PLACE (dessins ajoutés par-dessus)

COUCHES DU MASQUE (dans l'ordre, du fond vers l'avant) :
1. MAILLAGE (turquoise translucide 35%) : triangles complets MediaPipe
2. CONTOURS par élément : ovale, sourcils, iris, lèvres, nez (couleurs dédiées)
3. YEUX dans la COULEUR DE L'ÉTAT : vert/jaune/rouge = retour visuel immédiat
4. REPÈRES D'ANGLE : 4 coins style "viseur" autour du visage
5. GUIDES CROIX : croix + cercle centrés sur le visage
6. BADGE : "VISAGE VERROUILLÉ" (point vert) ou "RECHERCHE..." (point orange)

POURQUOI CE MASQUE EST PRÉCIEUX :
- DÉBOGAGE : on voit INSTANTANÉMENT si le suivi fonctionne/perd le visage
- DÉMO : public voit le traitement interne fonctionner en direct
- CONFIANCE : "visage verrouillé" = système opérationnel
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from interface.dessin import Dessinateur, Theme
from interface.liaisons import LiaisonsVisage
from noyau.modeles import Etat, OptionsAffichage, ResultatVisage

PointEntier = Tuple[int, int]


class RenduMasque:
    """Dessine les couches du masque clinique sur l'image de la caméra."""

    def __init__(self, liaisons: LiaisonsVisage) -> None:
        self._liaisons = liaisons
        self._dessinateur = Dessinateur()

    def dessiner(self, image: np.ndarray, visage: Optional[ResultatVisage], etat: Etat,
                 options: OptionsAffichage) -> None:
        """Applique toutes les couches du masque ; sans visage, seul le badge "RECHERCHE" apparaît.

        POURQUOI : point d'entrée unique, appelé une fois par image.
        """
        if visage is None:
            self._badge(image, verrouille=False)
            return
        points = visage.en_entiers()
        if options.maillage:
            self._maillage(image, points)
        self._contours(image, points)
        self._yeux(image, points, etat)
        if options.guides:
            self._reperes_angle(image, points)
            self._guides_croix(image, points)
        self._badge(image, verrouille=True)

    def _maillage(self, image: np.ndarray, points: List[PointEntier]) -> None:
        """Dessine le maillage complet (triangles) en turquoise translucide.

        POURQUOI : montrer la qualité globale du suivi sans cacher le visage.
        """
        calque = image.copy()
        total = len(points)
        for debut, fin in self._liaisons.groupe("maillage"):
            if debut < total and fin < total:
                cv2.line(calque, points[debut], points[fin], Theme.MAILLAGE, 1, cv2.LINE_AA)
        self._dessinateur.fond_translucide(image, calque, 0.35)

    def _contours(self, image: np.ndarray, points: List[PointEntier]) -> None:
        """Dessine le contour de chaque élément dans sa couleur : ovale, sourcils, iris, lèvres, nez.

        POURQUOI : suivre séparément chaque élément du visage, comme sur la capture clinique.
        """
        total = len(points)
        groupes = (
            ("ovale", Theme.OVALE), ("sourcil_gauche", Theme.SOURCILS),
            ("sourcil_droit", Theme.SOURCILS), ("iris_gauche", Theme.IRIS),
            ("iris_droit", Theme.IRIS), ("levres", Theme.LEVRES), ("nez", Theme.NEZ),
        )
        for nom, couleur in groupes:
            for debut, fin in self._liaisons.groupe(nom):
                if debut < total and fin < total:   # les iris n'existent que dans la version à 478 points
                    cv2.line(image, points[debut], points[fin], couleur, 1, cv2.LINE_AA)

    def _yeux(self, image: np.ndarray, points: List[PointEntier], etat: Etat) -> None:
        """Dessine les yeux dans la couleur de l'état : vert, orange ou rouge.

        POURQUOI : donner un retour visuel immédiat sur ce que le système a décidé.
        """
        couleur = Theme.couleur_etat(etat)
        total = len(points)
        for nom in ("oeil_gauche", "oeil_droit"):
            for debut, fin in self._liaisons.groupe(nom):
                if debut < total and fin < total:
                    cv2.line(image, points[debut], points[fin], couleur, 2, cv2.LINE_AA)

    def _reperes_angle(self, image: np.ndarray, points: List[PointEntier]) -> None:
        """Dessine quatre repères d'angle autour du visage (cadrage clinique).

        POURQUOI : donner l'impression d'un visage "verrouillé" par un instrument de mesure.
        """
        # Boîte englobante des 468 points du visage (les points d'iris sont exclus).
        x_min = min(p[0] for p in points[:468])
        x_max = max(p[0] for p in points[:468])
        y_min = min(p[1] for p in points[:468])
        y_max = max(p[1] for p in points[:468])
        marge = int(0.08 * (x_max - x_min))
        x1, y1, x2, y2 = x_min - marge, y_min - marge, x_max + marge, y_max + marge
        longueur = int(0.18 * min(x2 - x1, y2 - y1))
        for x, y, sens_x, sens_y in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(image, (x, y), (x + sens_x * longueur, y), Theme.ACCENT, 2, cv2.LINE_AA)
            cv2.line(image, (x, y), (x, y + sens_y * longueur), Theme.ACCENT, 2, cv2.LINE_AA)

    def _guides_croix(self, image: np.ndarray, points: List[PointEntier]) -> None:
        """Dessine une croix et un cercle centrés sur le visage (guides de la capture).

        POURQUOI : aider à vérifier le centrage du conducteur devant la caméra.
        """
        # Boîte englobante des 468 points du visage (les points d'iris sont exclus).
        x_min = min(p[0] for p in points[:468])
        x_max = max(p[0] for p in points[:468])
        y_min = min(p[1] for p in points[:468])
        y_max = max(p[1] for p in points[:468])
        centre = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        discret = (120, 150, 30)   # turquoise assombri pour des lignes fines et discrètes
        cv2.line(image, (centre[0], y_min), (centre[0], y_max), discret, 1, cv2.LINE_AA)
        cv2.line(image, (x_min, centre[1]), (x_max, centre[1]), discret, 1, cv2.LINE_AA)
        cv2.circle(image, centre, max(4, int(0.06 * (x_max - x_min))), Theme.ACCENT, 1, cv2.LINE_AA)

    def _badge(self, image: np.ndarray, verrouille: bool) -> None:
        """Affiche "VISAGE VERROUILLÉ" (point vert) ou "RECHERCHE..." (point orange).

        POURQUOI : indiquer en permanence si le suivi du visage est actif.
        """
        texte = "VISAGE VERROUILLE" if verrouille else "RECHERCHE..."
        couleur = Theme.VERT if verrouille else Theme.ORANGE
        largeur_image = image.shape[1]
        coin2 = (largeur_image - 12, 40)
        coin1 = (largeur_image - 12 - 190, 12)
        calque = image.copy()
        self._dessinateur.rectangle_arrondi(calque, coin1, coin2, Theme.FOND, rayon=12, rempli=True)
        self._dessinateur.fond_translucide(image, calque, 0.75)
        self._dessinateur.rectangle_arrondi(image, coin1, coin2, Theme.BORDURE, rayon=12)
        cv2.circle(image, (coin1[0] + 16, 26), 5, couleur, -1, cv2.LINE_AA)
        self._dessinateur.texte(image, texte, (coin1[0] + 30, 31), taille=0.45, couleur=Theme.TEXTE)
