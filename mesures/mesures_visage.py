"""Mesures géométriques du visage : distances, EAR (yeux) et MAR (bouche).

================================================================================
CE QUE FAIT CE FICHIER (BLOC 5 : MESURES PURES)
================================================================================
Transforme des coordonnées brutes (pixels) en MESURES INVARIANTES À LA TAILLE.
Ces mesures sont de la GÉOMÉTRIE PURS, sans apprentissage → ENTIÈREMENT EXPLICABLES.

DEUX MESURES PRINCIPALES :
--------------------------
1. EAR (Eye Aspect Ratio) - Ouverture des YEUX
   Formule : EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
   
   Six points par œil (indices MediaPipe) :
       p2 ●──────● p3      (paupière supérieure)
         │       │
     p1 ●       ● p4      (coins de l'œil)
         │       │
       p6 ●──────● p5      (paupière inférieure)
   
   - NUMÉRATEUR = somme des 2 hauteurs (paupière sup → inf)
   - DÉNOMINATEUR = 2 × largeur (coin → coin)
   - RAPPORT = FORME de l'œil, PAS sa taille
   - Indépendant de la distance caméra-conducteur !
   - Valeurs typiques : ~0.27 yeux ouverts, <0.10 yeux fermés

2. MAR (Mouth Aspect Ratio) - Ouverture de la BOUCHE
   Formule : MAR = (||haut1-bas1|| + ||haut2-bas2|| + ||haut3-bas3||) / (3 × ||coinG-coinD||)
   - 3 paires de points lèvres haute/basse / 3 × largeur bouche
   - ~0.06 bouche fermée, >0.55 bâillement

POURQUOI CES FORMULES FONCTIONNENT :
- Rapport de distances = INVARIANT d'échelle (zoom/dézoom ne change rien)
- Uniquement la FORME compte, pas la taille absolue
- C'est pour ça qu'on peut calibrer un seuil personnel (ratio × médiane EAR ouvert)
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from noyau.modeles import ResultatVisage

Point = Tuple[float, float]


class Geometrie:
    """Outils géométriques de base (méthodes statiques)."""

    @staticmethod
    def distance(a: Point, b: Point) -> float:
        """Distance euclidienne entre deux points : √((xa − xb)² + (ya − yb)²).

        POURQUOI : c'est la brique de base de l'EAR, du MAR et de la pose de la tête.
        """
        # APPORT : `math.hypot` calcule directement la racine de la somme des carrés,
        # de façon plus précise et plus lisible que l'écriture manuelle.
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def normaliser_angle(angle: float) -> float:
        """Ramène un angle (en degrés) dans l'intervalle [−180, 180[.

        POURQUOI : le tangage est cyclique ; sans cette normalisation, un angle de 179°
        puis de −179° ressemblerait à un saut de 358° alors que la tête a à peine bougé.
        """
        return (angle + 180.0) % 360.0 - 180.0


class CalculateurEAR:
    """Mesure l'ouverture des yeux : Eye Aspect Ratio (Soukupová et Čech, 2016).

    Six points entourent chaque œil (p1 et p4 sont les coins ; p2, p3 sont sur la paupière
    supérieure ; p5, p6 sur la paupière inférieure) :

            p2      p3
      p1                  p4
            p6      p5

    EAR = ( ‖p2 − p6‖ + ‖p3 − p5‖ ) / ( 2 × ‖p1 − p4‖ )
    """

    # Indices MediaPipe de (p1, p2, p3, p4, p5, p6) pour chaque œil du sujet.
    INDICES_OEIL_DROIT = (33, 160, 158, 133, 153, 144)
    INDICES_OEIL_GAUCHE = (362, 385, 387, 263, 373, 380)

    def calculer(self, visage: ResultatVisage) -> float:
        """Renvoie l'EAR moyen des deux yeux.

        POURQUOI : moyenner les deux yeux réduit le bruit d'un point mal placé.
        Valeurs typiques : environ 0,27 yeux ouverts, moins de 0,10 yeux fermés.
        """
        ear_droit = self._ear_un_oeil(visage, self.INDICES_OEIL_DROIT)
        ear_gauche = self._ear_un_oeil(visage, self.INDICES_OEIL_GAUCHE)
        return (ear_droit + ear_gauche) / 2.0

    def _ear_un_oeil(self, visage: ResultatVisage, indices: Sequence[int]) -> float:
        """Applique la formule de l'EAR à un seul œil.

        POURQUOI : le rapport hauteur/largeur reste stable quand le conducteur s'approche
        ou s'éloigne de la caméra : l'EAR mesure une FORME, pas une taille.
        """
        p1, p2, p3, p4, p5, p6 = visage.points(indices)
        largeur = Geometrie.distance(p1, p4)
        if largeur == 0.0:
            return 0.0   # protection contre une division par zéro (points confondus)
        hauteurs = Geometrie.distance(p2, p6) + Geometrie.distance(p3, p5)
        return hauteurs / (2.0 * largeur)


class CalculateurMAR:
    """Mesure l'ouverture de la bouche : Mouth Aspect Ratio (même principe que l'EAR).

    MAR = ( ‖p82 − p87‖ + ‖p13 − p14‖ + ‖p312 − p317‖ ) / ( 3 × ‖p78 − p308‖ )
    Les points utilisés sont ceux des lèvres internes.
    """

    PAIRES_HAUTEUR = ((82, 87), (13, 14), (312, 317))   # lèvre haute / lèvre basse
    COINS_BOUCHE = (78, 308)                            # coins internes de la bouche

    def calculer(self, visage: ResultatVisage) -> float:
        """Renvoie le MAR : environ 0,06 bouche fermée, plus de 0,55 lors d'un bâillement.

        POURQUOI : un MAR élevé pendant plus d'une seconde signale un bâillement,
        signe classique de fatigue.
        """
        gauche, droite = visage.points(self.COINS_BOUCHE)
        largeur = Geometrie.distance(gauche, droite)
        if largeur == 0.0:
            return 0.0
        somme_hauteurs = sum(
            Geometrie.distance(visage.point(haut), visage.point(bas))
            for haut, bas in self.PAIRES_HAUTEUR
        )
        return somme_hauteurs / (3.0 * largeur)
