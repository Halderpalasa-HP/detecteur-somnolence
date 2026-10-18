"""Acquisition vidéo : lecture des images de la caméra.

================================================================================
CE QUE FAIT CE FICHIER (BLOC 1 : ACQUISITION)
================================================================================
ISOLE LE MATÉRIEL. Le reste du programme ne manipule que des images NumPy.

ENTRÉE : indice caméra (0 = défaut), résolution demandée
SORTIE : (image BGR shape=(H,W,3), instant en secondes via time.monotonic())

POURQUOI time.monotonic() ET PAS time.time() ?
- time.time() = heure système, PEUT RECULER si utilisateur change l'heure/NTP
- time.monotonic() = horloge monotone, JAMAIS RECULE = parfait pour mesurer DURÉES
  (FPS, fermeture yeux, fenêtres temporelles, délais alerte)

POURQUOI CAP_DSHOW SOUS WINDOWS ?
- DirectShow = backend natif Windows → démarrage + rapide, + fiable
- Sans ça : caméra peut mettre 3-5s à s'ouvrir ou échouer aléatoirement
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class Camera:
    """Lit les images d'une caméra avec `cv2.VideoCapture`.

    Une image est un tableau NumPy de forme (hauteur, largeur, 3), de type `uint8`
    (valeurs de 0 à 255), dont les trois canaux sont dans l'ordre B-G-R (bleu, vert, rouge).
    """

    def __init__(self, indice: int, largeur: int, hauteur: int) -> None:
        self._indice = indice
        self._largeur = largeur
        self._hauteur = hauteur
        self._capture: Optional[cv2.VideoCapture] = None

    def ouvrir(self) -> None:
        """Ouvre la caméra et règle la résolution demandée.

        POURQUOI : échouer avec un message clair si la caméra est absente ou déjà utilisée.
        """
        # APPORT : `cv2.VideoCapture` pilote la caméra via le système d'exploitation.
        # Sous Windows, le moteur DirectShow (CAP_DSHOW) démarre plus vite et plus fiablement.
        moteur = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        capture = cv2.VideoCapture(self._indice, moteur)
        if not capture.isOpened():
            raise RuntimeError(
                f"Impossible d'ouvrir la caméra n°{self._indice}. "
                "Vérifiez qu'elle est branchée et qu'aucune autre application ne l'utilise."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._largeur)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._hauteur)
        self._capture = capture

    def lire(self) -> Optional[Tuple[np.ndarray, float]]:
        """Renvoie (image BGR, instant en secondes) ou `None` si la lecture échoue.

        POURQUOI : toutes les durées du système (fermeture des yeux, fenêtres de temps) se
        calculent en secondes et non en nombre d'images, car la cadence n'est pas constante.
        """
        if self._capture is None:
            return None
        succes, image = self._capture.read()
        if not succes:
            return None
        # APPORT : `time.monotonic` est une horloge qui ne recule jamais (contrairement à
        # `time.time`, modifiable par le changement d'heure) : idéale pour mesurer des durées.
        return image, time.monotonic()

    def fermer(self) -> None:
        """Libère la caméra.

        POURQUOI : sans cela, la caméra resterait verrouillée pour les autres applications.
        """
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def est_ouverte(self) -> bool:
        """Indique si le flux vidéo est actif (condition de la boucle principale)."""
        return self._capture is not None and self._capture.isOpened()
