"""Thème visuel clinique et primitives de dessin OpenCV.

================================================================================
CE QUE FAIT CE FICHIER (BLOC 4 : PRIMITIVES VISUELLES)
================================================================================
CENTRALISE tout le style : couleurs, polices, formes de base.
Le reste de l'interface (masque, cartes) n'utilise QUE ces primitives.

STYLE "CLINIQUE" :
- Fond quasi noir bleuté (20,13,7)
- Cartes aux bords fins arrondis
- Étiquettes en MAJUSCULES ESPACÉES (tracking)
- Accent turquoise (191,212,45), vert=normal, orange=attention, rouge=danger
- OpenCV ne supporte QUE l'ASCII → accents retirés à l'affichage (ÉVEILLÉ→EVEILLE)
"""

from __future__ import annotations

import unicodedata
from typing import Tuple

import cv2
import numpy as np

from noyau.modeles import Etat

Couleur = Tuple[int, int, int]   # format B-G-R (ordre OpenCV)
Position = Tuple[int, int]


class Theme:
    """Palette de couleurs du thème sombre clinique (format B-G-R)."""

    FOND: Couleur = (20, 13, 7)
    CARTE: Couleur = (34, 23, 13)
    BORDURE: Couleur = (58, 42, 26)
    ACCENT: Couleur = (191, 212, 45)        # turquoise
    TEXTE: Couleur = (238, 224, 207)
    TEXTE_DISCRET: Couleur = (150, 130, 107)
    VERT: Couleur = (128, 222, 74)
    ORANGE: Couleur = (11, 158, 245)
    ROUGE: Couleur = (68, 68, 239)
    BLEU: Couleur = (250, 165, 96)
    GRIS: Couleur = (160, 160, 160)
    MAILLAGE: Couleur = (191, 212, 45)
    OVALE: Couleur = (200, 200, 200)
    SOURCILS: Couleur = (255, 200, 0)
    IRIS: Couleur = (255, 255, 0)
    LEVRES: Couleur = (80, 80, 255)
    NEZ: Couleur = (255, 255, 255)

    @staticmethod
    def couleur_etat(etat: Etat) -> Couleur:
        """Renvoie la couleur associée à un état.

        POURQUOI : les yeux du masque, le badge et les cartes restent ainsi cohérents.
        """
        couleurs = {
            Etat.CALIBRATION: Theme.BLEU,
            Etat.EVEILLE: Theme.VERT,
            Etat.SOMNOLENT: Theme.ORANGE,
            Etat.ENDORMI: Theme.ROUGE,
            Etat.VISAGE_PERDU: Theme.GRIS,
        }
        return couleurs[etat]


class Dessinateur:
    """Primitives de dessin réutilisées par le masque et par les cartes."""

    POLICE = cv2.FONT_HERSHEY_SIMPLEX

    def texte(self, image: np.ndarray, texte: str, position: Position, taille: float = 0.5,
              couleur: Couleur = Theme.TEXTE, epaisseur: int = 1, alignement: str = "gauche") -> int:
        """Écrit un texte aligné à gauche, au centre ou à droite de `position` ; renvoie sa largeur.

        POURQUOI : base de tous les libellés ; gère l'alignement et le retrait des accents.
        """
        # APPORT : `unicodedata.normalize("NFKD")` sépare chaque lettre de son accent pour
        # que l'on puisse retirer l'accent (OpenCV ne dessine que l'ASCII).
        propre = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode("ascii")
        (largeur, _), _ = cv2.getTextSize(propre, self.POLICE, taille, epaisseur)
        x, y = position
        if alignement == "droite":
            x -= largeur
        elif alignement == "centre":
            x -= largeur // 2
        cv2.putText(image, propre, (x, y), self.POLICE, taille, couleur, epaisseur, cv2.LINE_AA)
        return largeur

    def rectangle_arrondi(self, image: np.ndarray, coin1: Position, coin2: Position,
                          couleur: Couleur, rayon: int = 10, epaisseur: int = 1,
                          rempli: bool = False) -> None:
        """Dessine un rectangle aux coins arrondis, plein ou simplement tracé.

        POURQUOI : les cartes aux coins arrondis donnent l'aspect "clinique" de la capture.
        """
        x1, y1 = coin1
        x2, y2 = coin2
        r = max(1, min(rayon, (x2 - x1) // 2, (y2 - y1) // 2))
        if rempli:
            cv2.rectangle(image, (x1 + r, y1), (x2 - r, y2), couleur, -1)
            cv2.rectangle(image, (x1, y1 + r), (x2, y2 - r), couleur, -1)
            for centre in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
                cv2.circle(image, centre, r, couleur, -1, cv2.LINE_AA)
            return
        # Quatre côtés droits puis quatre quarts de cercle aux angles.
        cv2.line(image, (x1 + r, y1), (x2 - r, y1), couleur, epaisseur, cv2.LINE_AA)
        cv2.line(image, (x1 + r, y2), (x2 - r, y2), couleur, epaisseur, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + r), (x1, y2 - r), couleur, epaisseur, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + r), (x2, y2 - r), couleur, epaisseur, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + r, y1 + r), (r, r), 180, 0, 90, couleur, epaisseur, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - r, y1 + r), (r, r), 270, 0, 90, couleur, epaisseur, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - r, y2 - r), (r, r), 0, 0, 90, couleur, epaisseur, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + r, y2 - r), (r, r), 90, 0, 90, couleur, epaisseur, cv2.LINE_AA)

    def barre(self, image: np.ndarray, coin1: Position, coin2: Position, valeur: float,
              couleur: Couleur = Theme.ACCENT, fond: Couleur = Theme.BORDURE) -> None:
        """Dessine une barre de progression fine remplie à `valeur` (de 0 à 1).

        POURQUOI : visualiser d'un coup d'œil l'EAR, le PERCLOS ou l'avancement de la calibration.
        """
        x1, y1 = coin1
        x2, y2 = coin2
        cv2.rectangle(image, (x1, y1), (x2, y2), fond, -1)
        rempli = int((x2 - x1) * max(0.0, min(1.0, valeur)))
        if rempli > 0:
            cv2.rectangle(image, (x1, y1), (x1 + rempli, y2), couleur, -1)

    def etiquette(self, image: np.ndarray, texte: str, position: Position,
                  couleur: Couleur = Theme.TEXTE_DISCRET) -> None:
        """Écrit un petit libellé en majuscules espacées (style de la capture).

        POURQUOI : reproduire les titres discrets des cartes cliniques ("MESURES"…).
        """
        x, y = position
        for lettre in texte.upper():
            largeur = self.texte(image, lettre, (x, y), taille=0.38, couleur=couleur)
            x += max(largeur, 4) + 2   # espacement entre les lettres

    def fond_translucide(self, image: np.ndarray, calque: np.ndarray, alpha: float) -> None:
        """Mélange un calque dessiné à part avec l'image : sortie = α × calque + (1 − α) × image.

        POURQUOI : rendre le maillage et les badges translucides, pour ne pas masquer le visage.
        """
        # APPORT : `cv2.addWeighted` réalise la fusion alpha pixel par pixel.
        cv2.addWeighted(calque, alpha, image, 1.0 - alpha, 0, image)
