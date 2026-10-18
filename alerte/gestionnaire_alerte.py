"""Gestion de l'alerte : sonnerie, pause d'une seconde, puis message vocal.

================================================================================
CE QUE FAIT CE FICHIER (BLOC 9 : ALERTE SONORE)
================================================================================
RÉAGIT à 2 types d'ordres de l'Application :
- declencher() → séquence COMPLÈTE (état ENDORMI) : sonnerie → pause 1s → message vocal
- bip()        → sonnerie SEULE (état SOMNOLENT) : un bip court, max 1/min

ARCHITECTURE CRITIQUE : THREAD SÉPARÉ
-------------------------------------
La séquence audio tourne dans un Thread à part (daemon=True).
POURQUOI ? Si la boucle vidéo attendait la fin d'un son (play() + sleep) :
  - L'image se FIGERAIT pendant toute l'alerte
  - Le suivi du visage S'ARRÊTERAIT
  - Impossible de détecter la RÉOUVERTURE des yeux pour arrêter l'alerte

Avec un thread : boucle vidéo continue à 30 FPS, alerte joue en parallèle.

ARRÊT IMMÉDIAT (clé UX) :
-------------------------
- threading.Event `_arret` = drapeau partagé entre threads
- `self._arret.wait(durée)` = "dors `durée` secondes MAIS réveille-toi SI `_arret.set()`"
- Application appelle `arreter()` dès que yeux rouverts (état != ENDORMI)
- Thread audio se réveille, `wait()` renvoie True → `return` → séquence STOP instantanée

ANTI-SPAM :
- delai_entre_alertes (10s) : pas de répétition si conducteur reste endormi
- delai_entre_bips (60s)    : pas de bip intempestif en état SOMNOLENT
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import pygame

from config import Configuration


class GestionnaireAlerte:
    """Joue la séquence sonnerie → pause → message vocal, de façon interruptible."""

    def __init__(self, config: Configuration) -> None:
        self._config = config
        # APPORT : `pygame.mixer` lit les fichiers audio (WAV) et peut les arrêter à tout moment.
        pygame.mixer.init()
        self._sonnerie = pygame.mixer.Sound(str(config.chemin_sonnerie))
        self._voix = pygame.mixer.Sound(str(config.chemin_voix))

        # APPORT : un `threading.Event` est un drapeau partagé entre threads. `arret.wait(d)`
        # sert de pause interruptible : elle dort `d` secondes mais se réveille aussitôt si
        # un autre thread active le drapeau (c'est ce qui permet l'arrêt immédiat).
        self._arret = threading.Event()
        self._active = threading.Event()        # vrai pendant qu'une séquence est en cours
        self._statut = "REPOS"
        self._fil: Optional[threading.Thread] = None
        self._instant_derniere_fin = float("-inf")
        self._instant_dernier_bip = float("-inf")

    def declencher(self) -> None:
        """Lance la séquence complète, si aucune n'est active et si le délai minimal est passé.

        POURQUOI : éviter de répéter l'alerte en boucle tant que le conducteur reste endormi.
        """
        maintenant = time.monotonic()
        if self._active.is_set():
            return
        if maintenant - self._instant_derniere_fin < self._config.delai_entre_alertes:
            return
        self._arret.clear()
        self._active.set()
        # APPORT : `threading.Thread` exécute la séquence en parallèle de la boucle vidéo ;
        # `daemon=True` garantit que le thread ne bloque pas la fermeture du programme.
        self._fil = threading.Thread(target=self._sequence, daemon=True)
        self._fil.start()

    def arreter(self) -> None:
        """Interrompt la séquence en cours (appelé dès que les yeux se rouvrent).

        POURQUOI : une alerte qui continue alors que le conducteur est réveillé serait gênante.
        """
        if self._active.is_set():
            self._arret.set()

    def bip(self) -> None:
        """Joue la sonnerie seule, au plus une fois par minute (état SOMNOLENT).

        POURQUOI : avertir de façon légère que la fatigue est détectée, sans le message complet.
        """
        maintenant = time.monotonic()
        if self._active.is_set():
            return
        if maintenant - self._instant_dernier_bip < self._config.delai_entre_bips:
            return
        self._sonnerie.play()
        self._instant_dernier_bip = maintenant

    def statut(self) -> str:
        """Renvoie la phase en cours : REPOS, SONNERIE, PAUSE ou VOIX (affichée dans l'interface)."""
        return self._statut

    def fermer(self) -> None:
        """Arrête les sons et libère le lecteur audio (fermeture propre)."""
        self.arreter()
        if self._fil is not None and self._fil.is_alive():
            self._fil.join(timeout=1.0)
        pygame.mixer.quit()

    def _sequence(self) -> None:
        """Corps du thread : sonnerie, pause de 1 s, message vocal enchaîné directement.

        Chaque attente utilise `self._arret.wait(...)` : si le drapeau d'arrêt est activé,
        elle renvoie `True` et la séquence s'interrompt aussitôt.
        """
        try:
            self._statut = "SONNERIE"
            self._sonnerie.play()
            if self._arret.wait(self._sonnerie.get_length()):
                return
            self._statut = "PAUSE"
            if self._arret.wait(self._config.pause_apres_sonnerie):
                return
            self._statut = "VOIX"
            self._voix.play()
            self._arret.wait(self._voix.get_length())
        finally:
            pygame.mixer.stop()
            self._statut = "REPOS"
            self._instant_derniere_fin = time.monotonic()
            self._active.clear()
