"""Petits outils de l'analyse dans le temps.

================================================================================
CE QUE FAIT CE FICHIER (BRIQUES DE BASE DU BLOC 7)
================================================================================
4 OUTILS INDÉPENDANTS, combinés par AnalyseurTemporel.
TOUTES LES DURÉES EN SECONDES (jamais en nb d'images → cadence variable).

POURQUOI CES OUTILS ?
- Une seule image ne conclut RIEN (œil fermé = clignement 99% du temps)
- La somnolence se lit dans la DURÉE (fermeture longue) et la RÉPÉTITION (PERCLOS, compteurs)
"""

from __future__ import annotations

from collections import deque
from itertools import islice
from typing import Deque, Optional, Tuple


class LisseurExponentiel:
    """Lissage par moyenne mobile exponentielle (EMA).

    FORMULE : s(t) = α × x(t) + (1 − α) × s(t − 1)
    Exemple (α = 0,4, s précédent = 0,30, x = 0,10) : s = 0,04 + 0,18 = 0,22

    POURQUOI CE FILTRE ?
    - RÉCURSIF : ne mémorise QU'UNE seule valeur (s précédent) → O(1) mémoire
    - RÉACTIF : α=0.4 = 40% nouvelle valeur, 60% historique = bon compromis
    - Évite qu'un pic d'une image (bruit) ne déclenche/interrompe une fermeture
    """

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._valeur_lissee: Optional[float] = None

    def lisser(self, valeur: float) -> float:
        """Renvoie la valeur lissée."""
        if self._valeur_lissee is None:
            self._valeur_lissee = valeur
        else:
            self._valeur_lissee = self._alpha * valeur + (1.0 - self._alpha) * self._valeur_lissee
        return self._valeur_lissee

    def reinitialiser(self) -> None:
        """Oublie l'historique (utile après une perte du visage)."""
        self._valeur_lissee = None


class FenetrePerclos:
    """Fenêtre glissante sur l'état « œil fermé », pour calculer le PERCLOS.

    PERCLOS = (temps passé yeux fermés dans la fenêtre) / (durée COMPLÈTE de la fenêtre, 60 s).

    POURQUOI DIVISER PAR LA DURÉE COMPLÈTE ?
    - Juste après démarrage : 5s fermées / 60s = 8% (pas 100%)
    - Évite un PERCLOS artificiellement élevé au début
    """

    def __init__(self, duree: float) -> None:
        self._duree = duree
        # APPORT : `deque` = file à deux bouts, retire les plus anciens en O(1) → fenêtre glissante
        self._observations: Deque[Tuple[float, bool]] = deque()

    def ajouter(self, instant: float, ferme: bool) -> None:
        """Ajoute une observation (instant, œil fermé ?) et retire les plus anciennes."""
        self._observations.append((instant, ferme))
        self._purger(instant)

    def valeur(self) -> float:
        """Renvoie la fraction de temps « fermé » sur la fenêtre (0 à 1) : le PERCLOS.

        CHAQUE observation « fermé » compte pour l'INTERVALLE DE TEMPS qui la sépare
        de la suivante → résultat INDÉPENDANT de la cadence caméra (30 FPS ou 15 FPS = même PERCLOS).
        """
        if len(self._observations) < 2:
            return 0.0
        temps_ferme = 0.0
        precedente = self._observations[0]
        for suivante in islice(self._observations, 1, None):
            if precedente[1]:
                temps_ferme += suivante[0] - precedente[0]
            precedente = suivante
        return min(1.0, temps_ferme / self._duree)

    def _purger(self, instant: float) -> None:
        """Retire les observations plus vieilles que la durée de la fenêtre."""
        while self._observations and instant - self._observations[0][0] > self._duree:
            self._observations.popleft()


class CompteurEvenements:
    """Compte les événements survenus récemment (fermetures longues, bâillements)."""

    def __init__(self, duree: float) -> None:
        self._duree = duree
        self._instants: Deque[float] = deque()

    def enregistrer(self, instant: float) -> None:
        """Ajoute un événement en mémorisant son instant."""
        self._instants.append(instant)

    def nombre(self, instant: float) -> int:
        """Renvoie le nombre d'événements dans la fenêtre.

        POURQUOI : un événement isolé ne dit rien, c'est la RÉPÉTITION qui signale la fatigue.
        """
        self._purger(instant)
        return len(self._instants)

    def _purger(self, instant: float) -> None:
        """Retire les événements trop anciens."""
        while self._instants and instant - self._instants[0] > self._duree:
            self._instants.popleft()


class ChronometreEtat:
    """Mesure la durée continue pendant laquelle une condition reste vraie.

    POURQUOI : distinguer un CLIGNEMENT (0.1-0.4s) d'un ENDORMISSEMENT (1.5s+).
    """

    def __init__(self) -> None:
        self._instant_debut: Optional[float] = None

    def mettre_a_jour(self, instant: float, actif: bool) -> Tuple[float, Optional[float]]:
        """Renvoie (durée en cours, durée terminée ou None).

        - Tant que condition vraie  : durée en cours > 0
        - À l'instant où elle devient fausse : retourne la durée de la période qui VIENT DE FINIR
        POURQUOI : reconnaître à la fois le DÉBUT et la FIN d'une fermeture.
        """
        if actif:
            if self._instant_debut is None:
                self._instant_debut = instant
            return instant - self._instant_debut, None
        if self._instant_debut is not None:
            duree_terminee = instant - self._instant_debut
            self._instant_debut = None
            return 0.0, duree_terminee
        return 0.0, None

    def reinitialiser(self) -> None:
        """Remet le chronomètre à zéro (évite une "durée fantôme")."""
        self._instant_debut = None
