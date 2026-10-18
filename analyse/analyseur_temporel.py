"""Analyse temporelle : de la suite de mesures aux indicateurs de somnolence.

================================================================================
CE QUE FAIT CE FICHIER (BLOC 7 : ANALYSE TEMPORELLE)
================================================================================
COMBINE les 4 outils (lisseur, fenêtre PERCLOS, chronomètres, compteurs)
en un SEUL point d'entrée : mettre_a_jour(instant, ear, mar, seuil_ear) → Indicateurs

TOUTES LES DURÉES EN SECONDES (time.monotonic), JAMAIS en nombre d'images.
"""

from __future__ import annotations

from analyse.outils_temporels import (
    ChronometreEtat,
    CompteurEvenements,
    FenetrePerclos,
    LisseurExponentiel,
)
from config import Configuration
from noyau.modeles import Indicateurs


class AnalyseurTemporel:
    """Transforme les mesures image par image en indicateurs sur plusieurs secondes."""

    def __init__(self, config: Configuration) -> None:
        self._config = config
        self._lisseur = LisseurExponentiel(config.alpha_lissage)
        self._fenetre_perclos = FenetrePerclos(config.fenetre_perclos)
        self._chrono_yeux = ChronometreEtat()
        self._fermetures_longues = CompteurEvenements(config.fenetre_perclos)
        self._chrono_bouche = ChronometreEtat()
        self._baillements = CompteurEvenements(config.fenetre_baillements)
        # Dernières valeurs calculées pour les yeux.
        self._ear_lisse = 0.0
        self._ferme = False
        self._duree_fermeture = 0.0

    def mettre_a_jour(self, instant: float, ear: float, mar: float, seuil_ear: float) -> Indicateurs:
        """Intègre les mesures de l'image courante et renvoie les indicateurs de l'instant.

        POURQUOI : interface simple pour le pipeline, qui n'a pas à connaître les outils internes.
        """
        self._traiter_yeux(instant, ear, seuil_ear)
        self._traiter_bouche(instant, mar)
        return Indicateurs(
            ferme=self._ferme,
            duree_fermeture=self._duree_fermeture,
            perclos=self._fenetre_perclos.valeur(),
            fermetures_longues=self._fermetures_longues.nombre(instant),
            baillements=self._baillements.nombre(instant),
            ear_lisse=self._ear_lisse,
        )

    def _traiter_yeux(self, instant: float, ear: float, seuil_ear: float) -> None:
        """Lisse l'EAR, met à jour la fenêtre PERCLOS, le chronomètre et les fermetures longues.

        POURQUOI : isoler toute la logique liée aux yeux.
        """
        self._ear_lisse = self._lisseur.lisser(ear)
        self._ferme = self._ear_lisse < seuil_ear
        self._fenetre_perclos.ajouter(instant, self._ferme)

        duree_en_cours, duree_terminee = self._chrono_yeux.mettre_a_jour(instant, self._ferme)
        self._duree_fermeture = duree_en_cours

        # Une fermeture qui vient de se terminer et qui a duré entre 0,5 s et 1,5 s est une
        # « fermeture longue » : signe de fatigue lorsqu'elle se répète.
        if duree_terminee is not None:
            if self._config.duree_fermeture_longue <= duree_terminee < self._config.duree_endormi:
                self._fermetures_longues.enregistrer(instant)

    def _traiter_bouche(self, instant: float, mar: float) -> None:
        """Détecte les bâillements : bouche très ouverte pendant plus de 1,2 s.

        POURQUOI : isoler la logique de la bouche. Parler ou sourire ne dépasse pas cette durée.
        """
        bouche_ouverte = mar > self._config.seuil_mar
        _, duree_terminee = self._chrono_bouche.mettre_a_jour(instant, bouche_ouverte)
        if duree_terminee is not None and duree_terminee >= self._config.duree_baillement:
            self._baillements.enregistrer(instant)

    def reinitialiser(self) -> None:
        """Réinitialise tous les outils (reprise propre après une perte du visage)."""
        self._lisseur.reinitialiser()
        self._chrono_yeux.reinitialiser()
        self._chrono_bouche.reinitialiser()
        self._duree_fermeture = 0.0
        self._ferme = False
