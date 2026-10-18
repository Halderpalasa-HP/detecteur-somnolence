"""Décision : déduire l'état du conducteur à partir des indicateurs.

================================================================================
CE QUE FAIT CE FICHIER (BLOC 8 : DÉCISION PAR RÈGLES)
================================================================================
APPLIQUE des règles ORDONNÉES PAR GRAVITÉ : ENDORMI > SOMNOLENT > ÉVEILLÉ
→ La PREMIÈRE règle qui s'applique DÉTERMINE l'état (pas de pondération floue).

ENTRÉE : Indicateurs (fermé?, durée_fermeture, PERCLOS, fermetures_longues, bâillements, ear_lisse)
        + ecart_tangage (degrés, écart vs tangage_reference de la calibration)

SORTIE : Etat (ENDORMI, SOMNOLENT, ÉVEILLE) → transmis à l'alerte + interface

POURQUOI DES RÈGLES (et pas un classifieur ML) ?
- ENTIÈREMENT TRANSPARENT : on peut expliquer À L'ORAL pourquoi le système a réagi
- DÉTERMINISTE : même entrée = même sortie, pas de "boîte noire"
- FACILE À AJUSTER : changer un seuil dans config.py = comportement prévisible
"""

from __future__ import annotations

from config import Configuration
from noyau.modeles import Etat, Indicateurs


class MoteurDecision:
    """Machine à états à règles : ENDORMI > SOMNOLENT > EVEILLE."""

    def __init__(self, config: Configuration) -> None:
        self._config = config

    def decider(self, indicateurs: Indicateurs, ecart_tangage: float) -> Etat:
        """Renvoie l'état du conducteur.

        POURQUOI : les règles sont testées de la plus grave à la moins grave ; la première
        qui s'applique détermine l'état.
        """
        if self._est_endormi(indicateurs, ecart_tangage):
            return Etat.ENDORMI
        if self._est_somnolent(indicateurs):
            return Etat.SOMNOLENT
        return Etat.EVEILLE

    def _est_endormi(self, indicateurs: Indicateurs, ecart_tangage: float) -> bool:
        """Vrai si les yeux restent fermés au moins 1,5 s, ou 0,8 s avec la tête penchée de plus de 20°.

        POURQUOI : c'est la condition la plus grave, celle qui déclenche l'alerte complète.
        """
        duree = indicateurs.duree_fermeture
        tete_penchee = abs(ecart_tangage) > self._config.angle_tete_penchee
        if duree >= self._config.duree_endormi:
            return True
        return tete_penchee and duree >= self._config.duree_endormi_tete_penchee

    def _est_somnolent(self, indicateurs: Indicateurs) -> bool:
        """Vrai si le PERCLOS atteint 0,15, ou 3 fermetures longues, ou 2 bâillements récents.

        POURQUOI : repérer la fatigue avant l'endormissement, avec un simple bip d'avertissement.
        """
        return (
            indicateurs.perclos >= self._config.seuil_perclos
            or indicateurs.fermetures_longues >= self._config.nombre_fermetures_longues
            or indicateurs.baillements >= self._config.nombre_baillements
        )
