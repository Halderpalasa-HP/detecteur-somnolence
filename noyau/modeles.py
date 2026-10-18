"""Objets de données partagés entre tous les modules.

================================================================================
RÔLE : DÉFINIR LES "PAQUETS DE DONNÉES" QUI CIRCULENT DANS LE PIPELINE
================================================================================
Chaque bloc du pipeline reçoit et renvoie un OBJET NOMMÉ et TYPÉ (dataclass/Enum).
C'est BEAUCOUP plus lisible et sûr qu'un tuple/liste dont il faut deviner l'ordre.

CLASSES DÉFINIES ICI :
1. Etat (Enum)           : 5 états possibles (CALIBRATION, EVEILLE, SOMNOLENT, ENDORMI, VISAGE_PERDU)
2. ProfilUtilisateur     : Résultat calibration (seuil_ear personnel, tangage_reference)
3. Indicateurs           : Sorties analyse temporelle → entrées décision
4. ResultatVisage        : 478 points du visage en pixels + méthodes d'accès
5. OptionsAffichage      : Interrupteurs interface (maillage, guides, valeurs)
6. SortieTraitement      : TOUT ce qui sort du traitement d'une image (paquet unique pour l'interface)

POURQUOI DES DATACLASSES ?
- `dataclass` génère auto : __init__, __repr__, __eq__
- `frozen=True` (sur ProfilUtilisateur) = IMMUABLE = pas de modification accidentelle
- Type hints = l'éditeur aide, mypy/pyright trouvent les erreurs avant exécution
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

# Un point du visage en pixels : (x, y).
Point = Tuple[float, float]


class Etat(Enum):
    """Les cinq états possibles du système.

    APPORT : `Enum` remplace des chaînes de caractères fragiles ("endormi", "Endormi"…)
             par des valeurs uniques ; une faute de frappe devient une erreur immédiate.
    """

    CALIBRATION = "CALIBRATION"
    EVEILLE = "ÉVEILLÉ"
    SOMNOLENT = "SOMNOLENT"
    ENDORMI = "ENDORMI"
    VISAGE_PERDU = "VISAGE PERDU"

    @property
    def libelle(self) -> str:
        """Texte français affiché à l'écran.

        POURQUOI : séparer ce que l'on AFFICHE (le libellé) de ce que l'on COMPARE (l'état).
        """
        return self.value


@dataclass(frozen=True)
class ProfilUtilisateur:
    """Résultat de la calibration : ce qui rend le système propre à la personne.

    IMMUABLE (frozen=True) : une fois calibré, le profil ne change plus.
    """

    seuil_ear: float          # En dessous de ce EAR lissé, l'œil est considéré FERMÉ
    tangage_reference: float  # Tangage (degrés) de la tête au repos (position "normale")


@dataclass
class Indicateurs:
    """Sorties de l'analyse temporelle, entrées de la décision.

    Tous ces champs sont calculés par AnalyseurTemporel.mettre_a_jour()
    """

    ferme: bool                 # L'œil est-il fermé À CET INSTANT (EAR lissé < seuil) ?
    duree_fermeture: float      # Durée (s) de la fermeture CONTINUE en cours
    perclos: float              # Fraction du temps yeux fermés sur fenêtre 60s (0 à 1)
    fermetures_longues: int     # Nb fermetures 0.5-1.5s récentes (fatigue)
    baillements: int            # Nb bâillements récents (bouche grande ouverte > 1.2s)
    ear_lisse: float            # EAR après lissage exponentiel (pour courbe/affichage)


@dataclass
class ResultatVisage:
    """Les points du visage détectés, convertis en pixels.

    `coordonnees` contient 478 points (468 visage + 10 iris) sous la forme (x, y).
    """

    coordonnees: List[Point]
    largeur: int
    hauteur: int

    def point(self, indice: int) -> Point:
        """Renvoie le point (x, y) d'indice donné.

        POURQUOI : point d'accès unique aux indices MediaPipe, plus facile à relire.
        """
        return self.coordonnees[indice]

    def points(self, indices: Sequence[int]) -> List[Point]:
        """Renvoie la liste des points demandés (par exemple les 6 points d'un œil).

        POURQUOI : extraire les 6 points d'un œil en une seule ligne.
        """
        return [self.coordonnees[i] for i in indices]

    def en_entiers(self) -> List[Tuple[int, int]]:
        """Renvoie tous les points arrondis à l'entier le plus proche.

        POURQUOI : les fonctions de dessin d'OpenCV (`cv2.line`) n'acceptent que des entiers.
        """
        return [(int(round(x)), int(round(y))) for x, y in self.coordonnees]


@dataclass
class OptionsAffichage:
    """Interrupteurs de l'interface : MAILLAGE, GUIDES, VALEURS."""

    maillage: bool = True
    guides: bool = True
    valeurs: bool = True

    def basculer(self, nom: str) -> None:
        """Inverse l'option nommée (`maillage`, `guides` ou `valeurs`).

        POURQUOI : utilisée à la fois par la souris ET par le clavier.
        """
        if not hasattr(self, nom):
            raise ValueError(f"Option d'affichage inconnue : {nom}")
        setattr(self, nom, not getattr(self, nom))


@dataclass
class SortieTraitement:
    """Tout ce qui sort du traitement d'une image : un seul paquet pour l'interface.

    L'interface (tableau_de_bord) ne reçoit QUE cet objet → découplage total.
    """

    etat: Etat
    visage: Optional[ResultatVisage] = None
    indicateurs: Optional[Indicateurs] = None
    ear: float = 0.0                 # EAR brut (moyenne des deux yeux)
    mar: float = 0.0                 # Ouverture de la bouche (MAR)
    ecart_tangage: float = 0.0       # Écart (degrés) par rapport au tangage de référence
    seuil_ear: float = 0.0           # Seuil personnel issu de la calibration
    progression: float = 0.0         # Avancement de la calibration (0 à 1)
    instant: float = 0.0             # Horodatage (secondes) de l'image traitée
