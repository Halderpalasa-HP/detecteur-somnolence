"""Cartes de données du panneau droit : état, mesures, alerte.

================================================================================
CE QUE FAIT CE FICHIER (BLOC 4 : CARTES DE DONNÉES)
================================================================================
AFFICHE EN DIRECT les chiffres du traitement interne, style clinique.
Le public VOIT le système "penser" : EAR, seuil, MAR, PERCLOS, durée, tangage.

ARCHITECTURE :
- Carte (classe de base) : cadre arrondi + titre + méthode abstraite dessiner()
- CarteEtat      : état en GRAND + explication + barre progression si calibration
- CarteMesures   : 4 lignes (EAR/seuil, MAR, PERCLOS, durée fermeture) + barres
- CarteAlerte    : 3 pastilles (SONNERIE/PAUSE/VOIX) → phase active allumée en rouge

POURQUOI DES CARTES SÉPARÉES ?
- Chaque carte = responsabilité unique → facile à modifier/tester
- Tableau de bord compose juste : carte_etat.dessiner(), carte_mesures.dessiner()...
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from interface.dessin import Dessinateur, Theme
from noyau.modeles import Etat, SortieTraitement

Rectangle = Tuple[int, int, int, int]   # (x1, y1, x2, y2)

# Phrase explicative affichée sous le nom de chaque état.
EXPLICATIONS = {
    Etat.CALIBRATION: "Regardez la route, yeux ouverts",
    Etat.EVEILLE: "Surveillance normale",
    Etat.SOMNOLENT: "Fatigue detectee : restez vigilant",
    Etat.ENDORMI: "Endormissement : alerte en cours",
    Etat.VISAGE_PERDU: "Visage non detecte",
}


class Carte:
    """Base commune des cartes : un cadre arrondi, un titre et une méthode `dessiner`."""

    def __init__(self, titre: str, rectangle: Rectangle) -> None:
        self._titre = titre
        self._rectangle = rectangle
        self._dessinateur = Dessinateur()

    def dessiner(self, canevas: np.ndarray, donnees) -> None:
        """Dessine la carte. À redéfinir dans chaque carte fille.

        POURQUOI : fixer l'interface commune que le tableau de bord appelle pour chaque carte.
        """
        raise NotImplementedError("Chaque carte doit redéfinir la méthode dessiner.")

    def _cadre(self, canevas: np.ndarray) -> None:
        """Dessine le fond arrondi de la carte, sa bordure fine et son titre en majuscules espacées.

        POURQUOI : toutes les cartes partagent le même cadre, d'où la cohérence visuelle.
        """
        x1, y1, x2, y2 = self._rectangle
        self._dessinateur.rectangle_arrondi(canevas, (x1, y1), (x2, y2), Theme.CARTE, rayon=12, rempli=True)
        self._dessinateur.rectangle_arrondi(canevas, (x1, y1), (x2, y2), Theme.BORDURE, rayon=12)
        self._dessinateur.etiquette(canevas, self._titre, (x1 + 18, y1 + 26))


class CarteEtat(Carte):
    """Carte "ÉTAT DU CONDUCTEUR" : l'information la plus importante, en grand."""

    def dessiner(self, canevas: np.ndarray, donnees: Optional[SortieTraitement]) -> None:
        """Affiche l'état en grand dans sa couleur ; `None` signifie que la caméra est arrêtée.

        POURQUOI : l'état doit être lisible à distance, d'un seul coup d'œil.
        """
        self._cadre(canevas)
        x1, y1, x2, y2 = self._rectangle
        if donnees is None:
            self._dessinateur.texte(canevas, "ARRETE", (x1 + 18, y1 + 72), 1.1, Theme.GRIS, 2)
            self._dessinateur.texte(canevas, "Cliquez sur DEMARRER", (x1 + 18, y1 + 98), 0.5, Theme.TEXTE_DISCRET)
            return
        couleur = Theme.couleur_etat(donnees.etat)
        self._dessinateur.texte(canevas, donnees.etat.libelle, (x1 + 18, y1 + 72), 1.1, couleur, 2)
        self._dessinateur.texte(canevas, EXPLICATIONS[donnees.etat], (x1 + 18, y1 + 98), 0.5, Theme.TEXTE_DISCRET)
        if donnees.etat is Etat.CALIBRATION:
            self._dessinateur.barre(canevas, (x2 - 150, y1 + 60), (x2 - 18, y1 + 66), donnees.progression, couleur)


class CarteMesures(Carte):
    """Carte "MESURES" : EAR et seuil, MAR, PERCLOS, durée de fermeture, tangage."""

    def dessiner(self, canevas: np.ndarray, donnees: SortieTraitement) -> None:
        """Affiche une ligne par mesure, avec sa valeur et une barre.

        POURQUOI : montrer les chiffres du traitement interne, qui sont le sujet de l'exposé.
        """
        self._cadre(canevas)
        x1, y1, _, _ = self._rectangle
        indicateurs = donnees.indicateurs
        perclos = indicateurs.perclos if indicateurs else 0.0
        duree = indicateurs.duree_fermeture if indicateurs else 0.0
        ear_ferme = donnees.seuil_ear > 0 and donnees.ear < donnees.seuil_ear

        self._ligne_mesure(canevas, y1 + 50, "EAR (yeux)", f"{donnees.ear:.2f}  /  seuil {donnees.seuil_ear:.2f}",
                           donnees.ear / 0.5, Theme.ORANGE if ear_ferme else Theme.ACCENT)
        self._ligne_mesure(canevas, y1 + 90, "MAR (bouche)", f"{donnees.mar:.2f}",
                           donnees.mar / 1.0, Theme.ACCENT)
        self._ligne_mesure(canevas, y1 + 130, "PERCLOS (60 s)", f"{perclos * 100:.0f} %",
                           perclos / 0.30, Theme.ORANGE if perclos >= 0.15 else Theme.ACCENT)
        self._ligne_mesure(canevas, y1 + 170, "Fermeture continue", f"{duree:.1f} s",
                           duree / 1.5, Theme.ROUGE if duree >= 1.5 else Theme.ACCENT)

    def _ligne_mesure(self, canevas: np.ndarray, y: int, libelle: str, valeur_texte: str,
                      ratio: float, couleur) -> None:
        """Dessine une ligne : libellé à gauche, valeur à droite, barre fine dessous.

        POURQUOI : éviter de répéter quatre fois le même code de mise en page.
        """
        x1, _, x2, _ = self._rectangle
        self._dessinateur.texte(canevas, libelle, (x1 + 18, y), 0.5, Theme.TEXTE_DISCRET)
        self._dessinateur.texte(canevas, valeur_texte, (x2 - 18, y), 0.55, Theme.TEXTE, 1, "droite")
        self._dessinateur.barre(canevas, (x1 + 18, y + 10), (x2 - 18, y + 14), ratio, couleur)


class CarteAlerte(Carte):
    """Carte "ALERTE" : montre la phase de la séquence sonnerie → pause → voix."""

    PHASES = ("SONNERIE", "PAUSE", "VOIX")

    def dessiner(self, canevas: np.ndarray, donnees: str) -> None:
        """Affiche trois pastilles ; celle de la phase en cours est allumée en rouge.

        POURQUOI : prouver visuellement que la séquence demandée se déroule bien.
        """
        self._cadre(canevas)
        x1, y1, x2, y2 = self._rectangle
        largeur = (x2 - x1 - 36 - 20) // 3
        for numero, phase in enumerate(self.PHASES):
            gauche = x1 + 18 + numero * (largeur + 10)
            allumee = donnees == phase
            couleur = Theme.ROUGE if allumee else Theme.BORDURE
            self._dessinateur.rectangle_arrondi(canevas, (gauche, y1 + 38), (gauche + largeur, y2 - 12),
                                                couleur, rayon=8, epaisseur=2, rempli=allumee)
            self._dessinateur.texte(canevas, phase, (gauche + largeur // 2, y1 + 38 + (y2 - y1 - 50) // 2 + 5),
                                    0.45, Theme.TEXTE if allumee else Theme.TEXTE_DISCRET, 1, "centre")
