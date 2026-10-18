"""Génération des fichiers audio de l'alerte (à faire une seule fois).

================================================================================
CE QUE FAIT CE FICHIER (PRÉPARATION BLOC 9)
================================================================================
CRÉE DEUX FICHIERS WAV une seule fois → exécution 100% HORS LIGNE ensuite.

1. SONNERIE (GenerateurSonnerie) : pattern "pi pi pi pi pi" (alarme)
   - Bip 100ms @ 880Hz + silence 100ms, répété 5 secondes
   - Enveloppe 5ms attack/release = pas de "clic" audible
   - 80% volume max = audible sans saturer

2. MESSAGE VOCAL (GenerateurVoix) : synthèse TTS française
   - Utilise edge-tts (voices Microsoft Neural en ligne) ou pyttsx3 (SAPI5 local)
   - Cherche voix française : fr-FR-DeniseNeural > fr-FR-HenriNeural > fr-CA > fr-BE
   - Sauvegarde assets/message.wav

POURQUOI GÉNÉRER UNE SEULE FOIS ?
- Pas de dépendance runtime à Internet / voix système
- Alerte = lecture simple WAV (pygame.mixer) = instantané, fiable
- Lancer : `python scripts/generer_assets.py`
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


class GenerateurSonnerie:
    """Crée une sonnerie type alarme : bips répétés (pi pi pi pi pi pi).

    PATTERN : bip de 100 ms à 880 Hz, silence de 100 ms, répété pendant ~5 secondes.
    """

    FREQUENCE_ECHANTILLONNAGE = 44100   # échantillons par seconde
    DUREE_TOTALE = 5.0                  # secondes (durée totale de la sonnerie)
    FREQUENCE_BIP = 880.0               # hertz (la aigu)
    DUREE_BIP = 0.1                     # secondes par bip
    DUREE_SILENCE = 0.1                 # secondes entre les bips

    def generer(self, chemin: Path) -> None:
        """Écrit le fichier WAV de la sonnerie à l'emplacement indiqué."""
        donnees = self._synthetiser()
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(chemin), "wb") as fichier:
            fichier.setnchannels(1)
            fichier.setsampwidth(2)
            fichier.setframerate(self.FREQUENCE_ECHANTILLONNAGE)
            fichier.writeframes(donnees.tobytes())

    def _synthetiser(self) -> np.ndarray:
        """Génère le pattern de bips répétés."""
        nb_echantillons_total = int(self.FREQUENCE_ECHANTILLONNAGE * self.DUREE_TOTALE)
        signal = np.zeros(nb_echantillons_total, dtype=np.float32)

        echantillons_bip = int(self.FREQUENCE_ECHANTILLONNAGE * self.DUREE_BIP)
        echantillons_silence = int(self.FREQUENCE_ECHANTILLONNAGE * self.DUREE_SILENCE)
        cycle = echantillons_bip + echantillons_silence

        temps_bip = np.linspace(0.0, self.DUREE_BIP, echantillons_bip, endpoint=False)
        bip_wave = np.sin(2.0 * np.pi * self.FREQUENCE_BIP * temps_bip)
        # Enveloppe pour éviter les clics (attaque/relâchement rapides)
        enveloppe = np.ones_like(bip_wave)
        fade_in = int(0.005 * self.FREQUENCE_ECHANTILLONNAGE)  # 5 ms
        fade_out = int(0.005 * self.FREQUENCE_ECHANTILLONNAGE)
        enveloppe[:fade_in] = np.linspace(0, 1, fade_in)
        enveloppe[-fade_out:] = np.linspace(1, 0, fade_out)
        bip_wave *= enveloppe

        pos = 0
        while pos + echantillons_bip <= nb_echantillons_total:
            fin_bip = pos + echantillons_bip
            signal[pos:fin_bip] = bip_wave
            pos = fin_bip + echantillons_silence

        signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
        return (0.8 * 32767 * signal).astype(np.int16)


class GenerateurVoix:
    """Crée le message vocal avec `edge-tts` (Microsoft Neural Voices) ou `pyttsx3` (fallback local)."""

    def generer(self, chemin: Path, message: str) -> None:
        """Synthétise le message et l'enregistre dans un fichier WAV.

        Essaie edge-tts en premier (voix neurones haute qualité), puis pyttsx3 en fallback.
        """
        # 1. ESSAI edge-tts (voix Microsoft Neural, nécessite Internet)
        try:
            import edge_tts
            import asyncio

            # Voix françaises disponibles (ordre de préférence)
            voix_fr = [
                "fr-FR-DeniseNeural",   # Femme, France, naturelle
                "fr-FR-HenriNeural",    # Homme, France, naturel
                "fr-FR-EloiseNeural",   # Femme, France
                "fr-FR-RemyMultilingualNeural",
                "fr-FR-VivienneMultilingualNeural",
                "fr-CA-SylvieNeural",   # Femme, Québec
                "fr-CA-AntoineNeural",  # Homme, Québec
                "fr-BE-CharlineNeural", # Femme, Belgique
                "fr-BE-GerardNeural",   # Homme, Belgique
            ]

            async def _generer_edge():
                for voix in voix_fr:
                    try:
                        communicate = edge_tts.Communicate(message, voix)
                        await communicate.save(str(chemin))
                        print(f"Message généré avec edge-tts ({voix})")
                        return True
                    except Exception:
                        continue
                return False

            if asyncio.run(_generer_edge()):
                return

        except ImportError:
            pass
        except Exception:
            pass

        # 2. FALLBACK pyttsx3 (SAPI5 Windows, hors ligne)
        import pyttsx3

        moteur = pyttsx3.init()
        self._lister_voix_disponibles(moteur)
        if not self._choisir_voix_francaise(moteur):
            print("Avertissement : aucune voix française trouvée, la voix par défaut est utilisée.")
        moteur.setProperty("rate", 150)
        moteur.setProperty("volume", 1.0)
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        moteur.save_to_file(message, str(chemin))
        moteur.runAndWait()
        print("Message généré avec pyttsx3 (fallback local)")

    def _lister_voix_disponibles(self, moteur) -> None:
        """Affiche les voix disponibles pour diagnostic."""
        print("Voix disponibles (pyttsx3/SAPI5) :")
        for i, voix in enumerate(moteur.getProperty("voices")):
            langs = getattr(voix, "languages", None) or []
            langs_str = ", ".join(
                l.decode("utf-8", "ignore") if isinstance(l, bytes) else str(l) for l in langs
            )
            print(f"  [{i}] {voix.name} (ID: {voix.id}) - Langues: {langs_str}")

    def _choisir_voix_francaise(self, moteur) -> bool:
        """Sélectionne la meilleure voix française disponible (pyttsx3)."""
        voix_fr = []
        for voix in moteur.getProperty("voices"):
            description = f"{voix.id} {voix.name}".lower()
            langues = " ".join(
                langue.decode("utf-8", "ignore") if isinstance(langue, bytes) else str(langue)
                for langue in (getattr(voix, "languages", None) or [])
            ).lower()
            if "fr" in description or "french" in description or "fr" in langues:
                voix_fr.append((voix, description, langues))

        if not voix_fr:
            return False

        # Préférer une voix avec "france" ou "hortense" (Windows) ou "fr-fr"
        for voix, desc, langs in voix_fr:
            if "france" in desc or "hortense" in desc or "fr-fr" in langs:
                moteur.setProperty("voice", voix.id)
                print(f"Voix française sélectionnée (pyttsx3) : {voix.name}")
                return True

        # Sinon prendre la première voix française trouvée
        moteur.setProperty("voice", voix_fr[0][0].id)
        print(f"Voix française sélectionnée (pyttsx3) : {voix_fr[0][0].name}")
        return True
