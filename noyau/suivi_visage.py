"""Détection du visage et des points de repère avec MediaPipe Face Landmarker.

================================================================================
CE QUE FAIT CE FICHIER (BLOC 2-3 : PERCEPTION)
================================================================================
Transforme une image BGR (OpenCV) en 478 points (x, y) du visage EN PIXELS.

FONCTIONNEMENT INTERNE DE MEDIAPIPE (ce qu'il faut savoir pour l'expliquer) :
---------------------------------------------------------------------------
MediaPipe Face Landmarker fait DEUX passes à la première image, puis UNE seule
passes aux images suivantes grâce au SUIVI :

  IMAGE 1 (détection complète) :
    1. BlazeFace (CNN léger, détecteur type SSD) → trouve la BOÎTE du visage
    2. Recadrage + redressement de la zone visage
    3. Face Landmark (CNN de régression) → prédit 478 points 3D (x, y, z normalisés 0-1)
       * 468 points visage + 10 points iris (5 par œil)
  
  IMAGES SUIVANTES (mode SUIVI) :
    - Le modèle UTILISE les points de l'image précédente pour prédire où est le visage
    - BlazeFace n'est relancé QUE si le suivi perd confiance (tête tourne, sort du champ)
    - ÇA rend le temps réel possible (~30 FPS sur CPU)

ENTRÉE / SORTIE :
- Entrée : image BGR (h, w, 3) + instant (secondes, time.monotonic)
- Sortie : ResultatVisage(coordonnees=[(x,y), ...], largeur, hauteur) ou None

POURQUOI CE FICHIER EST ISOLÉ :
- Seul endroit qui importe mediapipe → facile à remplacer par un autre détecteur
- Convertit les coordonnées normalisées (0-1) en PIXELS
  → Les distances (EAR, MAR) n'ont de sens qu'en pixels réels
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as taches_mediapipe
from mediapipe.tasks.python import vision

from config import Configuration
from noyau.modeles import ResultatVisage


class SuiviVisage:
    """Enveloppe de `FaceLandmarker` (BlazeFace + Face Landmark)."""

    def __init__(self, config: Configuration) -> None:
        self._config = config
        chemin = Path(config.chemin_modele)
        if not chemin.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {chemin}\n"
                "Téléchargez « face_landmarker.task » (voir le fichier README.md) "
                "et placez-le dans le dossier « assets »."
            )
        # APPORT : le chargement crée le graphe de traitement MediaPipe à partir du fichier
        # `.task` (archive contenant les modèles `.tflite`). On le fait une seule fois.
        self._detecteur = vision.FaceLandmarker.create_from_options(self._creer_options())
        self._origine: Optional[float] = None   # instant de la première image
        self._dernier_ms = -1                   # dernier horodatage envoyé (en millisecondes)

    def detecter(self, image_bgr: np.ndarray, instant: float) -> Optional[ResultatVisage]:
        """Convertit l'image, appelle le modèle et renvoie les points du visage.

        POURQUOI : c'est le cœur de la perception, de l'image brute aux points du visage.
        """
        hauteur, largeur = image_bgr.shape[:2]

        # APPORT : `cv2.cvtColor` permute les canaux B-G-R vers R-G-B, car MediaPipe attend du
        # RGB alors qu'OpenCV fournit du BGR (sans cela, les couleurs seraient inversées).
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_mediapipe = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # En mode VIDEO, MediaPipe exige des horodatages strictement croissants (en ms).
        if self._origine is None:
            self._origine = instant
        horodatage_ms = int((instant - self._origine) * 1000)
        if horodatage_ms <= self._dernier_ms:
            horodatage_ms = self._dernier_ms + 1
        self._dernier_ms = horodatage_ms

        # APPORT : `detect_for_video` utilise le SUIVI entre images : le détecteur BlazeFace
        # n'est relancé qu'en cas de besoin, ce qui rend le traitement assez rapide pour le temps réel.
        resultat = self._detecteur.detect_for_video(image_mediapipe, horodatage_ms)
        return self._convertir_resultat(resultat, largeur, hauteur)

    def fermer(self) -> None:
        """Libère le modèle (fermeture propre)."""
        self._detecteur.close()

    def _creer_options(self) -> "vision.FaceLandmarkerOptions":
        """Prépare les options de MediaPipe.

        POURQUOI : le mode VIDEO active le suivi entre images (plus rapide que le mode IMAGE) ;
        on ne demande qu'un seul visage (le conducteur) pour économiser du calcul.
        """
        return vision.FaceLandmarkerOptions(
            base_options=taches_mediapipe.BaseOptions(
                model_asset_path=str(self._config.chemin_modele)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self._config.confiance_detection,
            min_face_presence_confidence=self._config.confiance_presence,
            min_tracking_confidence=self._config.confiance_suivi,
        )

    def _convertir_resultat(self, resultat, largeur: int, hauteur: int) -> Optional[ResultatVisage]:
        """Passe des coordonnées normalisées (0 à 1) aux coordonnées en pixels.

        POURQUOI : les distances (EAR, MAR…) n'ont de sens qu'en pixels. Sur une image
        640 × 480, une variation de 0,1 vaut 64 pixels en x mais seulement 48 en y.
        """
        if not resultat.face_landmarks:
            return None
        points = [(p.x * largeur, p.y * hauteur) for p in resultat.face_landmarks[0]]
        return ResultatVisage(coordonnees=points, largeur=largeur, hauteur=hauteur)
