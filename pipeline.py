"""Pipeline de surveillance : traite une image de bout en bout.

================================================================================
CE QUE FAIT CE FICHIER (LE "CŒUR DU CALCUL")
================================================================================
Transforme UNE image brute en une SortieTraitement complète.
Sépare le CALCUL pur de l'AFFICHAGE → testable sans caméra ni fenêtre.

PIPELINE INTERNE (appelé par Application._boucle) :
---------------------------------------------------
traiter(image, instant) :
  1. suivi.detecter(image, instant) → ResultatVisage (478 points) ou None
  2. SI visage détecté :
       - Si pas encore calibré (_profil is None) → _calibrer()
       - Sinon → _surveiller()
  3. SI aucun visage :
       - _visage_perdu() : gère l'absence selon la phase (calibration ou surveillance)

PHASE 1 : CALIBRATION (_calibrer)
---------------------------------
Pendant ~5 secondes (config.duree_calibration) :
  - Mesure EAR brut + tangage à chaque image
  - calibrateur.ajouter_mesure(instant, ear, tangage)
  - À la fin : profil = calibrateur.construire_profil()
    * seuil_ear = 0.70 × médiane(EAR mesurés)  ← personnel à l'utilisateur
    * tangage_reference = médiane(tangages)  ← position tête "normale"
    * Si médiane EAR < 0.15 → calibration faussée (yeux mi-clos) → repart à zéro

POURQUOI LA MÉDIANE et pas la moyenne ?
- Un clignement fait chuter l'EAR brièvement (ex: 0.25 → 0.05 → 0.25)
- La moyenne serait tirée vers le bas par ces pics
- La médiane (valeur du milieu triée) IGNORE les clignements → seuil stable

PHASE 2 : SURVEILLANCE (_surveiller)
------------------------------------
Une fois calibré, pour CHAQUE image :
  1. EAR = calculateur_ear.calculer(visage)  (moyen des 2 yeux)
  2. MAR = calculateur_mar.calculer(visage)  (ouverture bouche)
  3. pose = estimateur_pose.estimer(visage)  → (tangage, lacet, roulis)
  4. ecart_tangage = normaliser_angle(pose.tangage - profil.tangage_reference)
     ← ÉCART par rapport à la position "normale" mesurée en calibration
     ← normaliser_angle évite le saut ±180° (cyclique)
  5. indicateurs = analyseur.mettre_a_jour(instant, EAR, MAR, profil.seuil_ear)
     ← Lissage EAR, PERCLOS (60s), chronomètre fermeture, compteurs
  6. etat = moteur.decider(indicateurs, ecart_tangage)
     ← Règles ordonnées : ENDORMI > SOMNOLENT > ÉVEILLÉ

GESTION VISAGE PERDU (_visage_perdu) :
--------------------------------------
- Avant calibration : reste en CALIBRATION (chronomètre ne démarre pas sans visage)
- Après calibration :
    * Absence > 3s (config.delai_visage_perdu) → VISAGE_PERDU + réinitialise analyseur
    * Absence brève : GARDE le dernier état
      ← Crucial : si conducteur s'endort et tête sort du champ, l'alerte NE DOIT PAS s'arrêter
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from analyse.analyseur_temporel import AnalyseurTemporel
from analyse.calibration import Calibrateur
from analyse.decision import MoteurDecision
from config import Configuration
from mesures.mesures_visage import CalculateurEAR, CalculateurMAR, Geometrie
from mesures.pose_tete import EstimateurPose
from noyau.modeles import Etat, ProfilUtilisateur, ResultatVisage, SortieTraitement
from noyau.suivi_visage import SuiviVisage


class PipelineSurveillance:
    """Transforme chaque image en une `SortieTraitement`."""

    def __init__(self, config: Configuration, suivi: SuiviVisage) -> None:
        self._config = config
        self._suivi = suivi
        self._calculateur_ear = CalculateurEAR()
        self._calculateur_mar = CalculateurMAR()
        self._estimateur_pose = EstimateurPose()
        self._calibrateur = Calibrateur(config)
        self._analyseur = AnalyseurTemporel(config)
        self._moteur = MoteurDecision(config)
        self._profil: Optional[ProfilUtilisateur] = None   # défini à la fin de la calibration
        self._instant_dernier_visage: Optional[float] = None
        self._dernier_etat = Etat.CALIBRATION

    def traiter(self, image: np.ndarray, instant: float) -> SortieTraitement:
        """Traite une image et renvoie le résultat complet.

        POURQUOI : point d'entrée unique du calcul, appelé une fois par image.
        """
        visage = self._suivi.detecter(image, instant)
        if visage is None:
            return self._visage_perdu(instant)
        self._instant_dernier_visage = instant
        if self._profil is None:
            return self._calibrer(visage, instant)
        return self._surveiller(visage, instant)

    def _calibrer(self, visage: ResultatVisage, instant: float) -> SortieTraitement:
        """Phase de calibration : mesure l'utilisateur yeux ouverts pendant quelques secondes.

        POURQUOI : fixer le seuil et le tangage de référence propres à cette personne.
        """
        ear = self._calculateur_ear.calculer(visage)
        mar = self._calculateur_mar.calculer(visage)
        pose = self._estimateur_pose.estimer(visage)
        tangage = pose[0] if pose is not None else 0.0

        self._calibrateur.ajouter_mesure(instant, ear, tangage)
        progression = self._calibrateur.progression(instant)
        if self._calibrateur.terminee(instant):
            # Si la calibration est jugée faussée, `construire_profil` renvoie None et repart de zéro.
            self._profil = self._calibrateur.construire_profil()
        self._dernier_etat = Etat.CALIBRATION if self._profil is None else Etat.EVEILLE
        return SortieTraitement(
            etat=self._dernier_etat, visage=visage, ear=ear, mar=mar,
            seuil_ear=self._profil.seuil_ear if self._profil else 0.0,
            progression=progression, instant=instant,
        )

    def _surveiller(self, visage: ResultatVisage, instant: float) -> SortieTraitement:
        """Régime normal : calcule les mesures, les indicateurs et l'état du conducteur.

        POURQUOI : c'est le cœur du système une fois l'utilisateur calibré.
        """
        profil = self._profil
        ear = self._calculateur_ear.calculer(visage)
        mar = self._calculateur_mar.calculer(visage)
        pose = self._estimateur_pose.estimer(visage)

        # Écart de tangage par rapport à la position de référence mesurée à la calibration.
        # `normaliser_angle` évite le saut de ±180° (le tangage est un angle cyclique).
        ecart = Geometrie.normaliser_angle(pose[0] - profil.tangage_reference) if pose else 0.0

        indicateurs = self._analyseur.mettre_a_jour(instant, ear, mar, profil.seuil_ear)
        self._dernier_etat = self._moteur.decider(indicateurs, ecart)
        return SortieTraitement(
            etat=self._dernier_etat, visage=visage, indicateurs=indicateurs, ear=ear, mar=mar,
            ecart_tangage=ecart, seuil_ear=profil.seuil_ear, instant=instant,
        )

    def _visage_perdu(self, instant: float) -> SortieTraitement:
        """Gère l'absence de visage.

        - avant la fin de la calibration : on reste en CALIBRATION ;
        - absence de plus de 3 s : état VISAGE_PERDU et analyse réinitialisée ;
        - absence brève : on conserve le dernier état (une tête qui tombe hors du champ
          pendant un endormissement ne doit pas faire cesser l'alerte).
        POURQUOI : traiter ce cas limite de façon explicite plutôt que de laisser le programme se tromper.
        """
        seuil = self._profil.seuil_ear if self._profil else 0.0
        if self._profil is None:
            return SortieTraitement(
                etat=Etat.CALIBRATION, seuil_ear=seuil,
                progression=self._calibrateur.progression(instant), instant=instant,
            )
        absent_depuis = instant - (self._instant_dernier_visage or instant)
        if absent_depuis > self._config.delai_visage_perdu:
            self._analyseur.reinitialiser()
            self._dernier_etat = Etat.VISAGE_PERDU
        return SortieTraitement(etat=self._dernier_etat, seuil_ear=seuil, instant=instant)
