"""Configuration centrale du projet de détection de somnolence.

================================================================================
CE QUE FAIT CE FICHIER : TOUS LES RÉGLAGES EN UN SEUL ENDROIT
================================================================================
AUCUN "NOMBRE MAGIQUE" dans le code. Pour ajuster le système → modifier ICI.

STRUCTURE DES RÉGLAGES (par section) :
--------------------------------------
1. FICHIERS : chemins modèle .task, sonnerie.wav, message.wav
2. CAMÉRA   : indice, résolution (640×480 = bon compromis vitesse/précision)
3. MEDIAPIPE: seuils confiance (détection/présence/suivi) - 0.5 = défaut équilibré
4. CALIBRATION : durée 5s, ratio seuil 0.70, EAR minimal valide 0.15
5. ANALYSE TEMPORELLE : α lissage 0.4, fenêtre PERCLOS 60s, fenêtre bâillements 300s
6. DÉCISION : seuils ENDORMI (1.5s / 0.8s tête penchée 20°), SOMNOLENT (PERCLOS 15%, 3 fermetures, 2 bâillements)
7. ALERTE : pause 1s, anti-spam 10s/60s
8. INTERFACE : fenêtre 1280×720, courbe 10s

IMMUTABLE (frozen=True) : une fois créé, impossible de modifier → pas de bugs de config qui change en cours d'exécution
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Dossier racine du projet : sert à construire les chemins des fichiers de ressources.
RACINE = Path(__file__).resolve().parent

MESSAGE_VOCAL = (
    "Attention. Il est déconseillé de dormir au volant. "
    "Arrêtez-vous dès que possible dans un endroit sûr, "
    "faites une pause d'au moins quinze minutes, "
    "buvez de l'eau ou prenez l'air."
)


@dataclass(frozen=True)
class Configuration:
    """Ensemble des réglages du système (objet immuable : `frozen=True`).

    APPORT : `dataclass` génère automatiquement le constructeur et l'affichage ;
             `frozen=True` empêche de modifier un réglage par erreur en cours d'exécution.
    """

    # --- Fichiers de ressources -------------------------------------------------------
    chemin_modele: Path = RACINE / "assets" / "face_landmarker.task"
    chemin_sonnerie: Path = RACINE / "assets" / "sonnerie.wav"
    chemin_voix: Path = RACINE / "assets" / "message.wav"

    # --- Caméra -----------------------------------------------------------------------
    indice_camera: int = 0
    largeur_image: int = 640
    hauteur_image: int = 480

    # --- MediaPipe (seuils de confiance) ----------------------------------------------
    confiance_detection: float = 0.5    # score minimal de BlazeFace pour accepter un visage
    confiance_presence: float = 0.5     # score minimal de présence du modèle de points
    confiance_suivi: float = 0.5        # en dessous, le suivi est abandonné et la détection relancée

    # --- Calibration ------------------------------------------------------------------
    duree_calibration: float = 5.0      # secondes de mesure yeux ouverts
    ratio_seuil_ear: float = 0.70       # seuil de fermeture = 70 % de l'EAR "yeux ouverts"
    ear_minimal_valide: float = 0.15    # médiane en dessous : calibration jugée faussée

    # --- Analyse temporelle -----------------------------------------------------------
    alpha_lissage: float = 0.4          # poids de la nouvelle valeur dans la moyenne exponentielle
    fenetre_perclos: float = 60.0       # secondes de la fenêtre glissante du PERCLOS
    fenetre_baillements: float = 300.0  # secondes sur lesquelles on compte les bâillements
    seuil_mar: float = 0.60             # au-dessus, la bouche est considérée grande ouverte
    duree_baillement: float = 1.2       # secondes d'ouverture pour compter un bâillement
    duree_fermeture_longue: float = 0.5  # secondes : en dessous, c'est un simple clignement

    # --- Décision ---------------------------------------------------------------------
    duree_endormi: float = 1.5                # fermeture continue qui déclenche "ENDORMI"
    duree_endormi_tete_penchee: float = 0.8   # durée réduite si la tête est penchée
    angle_tete_penchee: float = 20.0          # degrés d'écart de tangage jugés "tête penchée"
    seuil_perclos: float = 0.15               # 15 % du temps yeux fermés : somnolence
    nombre_fermetures_longues: int = 3
    nombre_baillements: int = 2
    delai_visage_perdu: float = 3.0           # secondes sans visage avant l'état VISAGE_PERDU

    # --- Alerte -----------------------------------------------------------------------
    pause_apres_sonnerie: float = 1.0   # pause exacte entre la sonnerie et le message vocal
    delai_entre_alertes: float = 10.0   # secondes minimales entre deux alertes complètes
    delai_entre_bips: float = 60.0      # secondes minimales entre deux bips (état SOMNOLENT)
    message_vocal: str = MESSAGE_VOCAL

    # --- Interface --------------------------------------------------------------------
    largeur_fenetre: int = 1280
    hauteur_fenetre: int = 720
    duree_courbe: float = 10.0          # secondes affichées sur la courbe de l'EAR

    @classmethod
    def par_defaut(cls) -> "Configuration":
        """Fabrique la configuration standard.

        POURQUOI : donner un point d'entrée explicite et lisible (`Configuration.par_defaut()`).
        """
        return cls()

    def valider(self) -> None:
        """Vérifie la cohérence des réglages et lève `ValueError` en cas d'erreur.

        POURQUOI : échouer tôt, avec un message clair, plutôt que de découvrir un seuil
        absurde au milieu d'une démonstration.
        """
        if not 0.0 < self.alpha_lissage <= 1.0:
            raise ValueError("alpha_lissage doit être compris entre 0 (exclu) et 1.")
        if not 0.0 < self.ratio_seuil_ear < 1.0:
            raise ValueError("ratio_seuil_ear doit être compris entre 0 et 1 (exclus).")
        if self.duree_calibration <= 0 or self.fenetre_perclos <= 0:
            raise ValueError("Les durées de calibration et de PERCLOS doivent être positives.")
        if self.duree_fermeture_longue >= self.duree_endormi:
            raise ValueError("duree_fermeture_longue doit être inférieure à duree_endormi.")
        if self.duree_endormi_tete_penchee > self.duree_endormi:
            raise ValueError("duree_endormi_tete_penchee ne peut pas dépasser duree_endormi.")
        if self.pause_apres_sonnerie < 0:
            raise ValueError("pause_apres_sonnerie ne peut pas être négative.")
