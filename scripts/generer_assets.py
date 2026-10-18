"""Génère les fichiers audio de l'alerte : `assets/sonnerie.wav` et `assets/message.wav`.

================================================================================
SCRIPT À LANCER UNE SEULE FOIS (après installation des dépendances)
================================================================================
Usage : `python scripts/generer_assets.py` depuis la racine du projet.

CE QU'IL FAIT :
1. Sonnerie : pattern "pi pi pi pi pi" (5s, bip 100ms/silence 100ms @ 880Hz)
2. Message vocal : synthèse TTS française du message de config.py
   - Essaie edge-tts (voix Microsoft Neural haute qualité, nécessite Internet)
   - Fallback pyttsx3 (SAPI5 Windows local, hors ligne)

PRÉREQUIS :
- Modèle `face_landmarker.task` dans `assets/` (voir README.md)
- Dépendances : `pip install -r requirements.txt` (+ `pip install edge-tts` pour voix premium)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permet d'importer les modules du projet quand on lance le script depuis le dossier "scripts".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alerte.generation_audio import GenerateurSonnerie, GenerateurVoix
from config import Configuration


def main() -> None:
    """Crée la sonnerie puis le message vocal."""
    config = Configuration.par_defaut()
    print("Génération de la sonnerie (pattern pi pi pi pi pi)...")
    GenerateurSonnerie().generer(config.chemin_sonnerie)
    print(f"  -> {config.chemin_sonnerie}")
    print("Génération du message vocal (TTS français)...")
    GenerateurVoix().generer(config.chemin_voix, config.message_vocal)
    print(f"  -> {config.chemin_voix}")
    print("Terminé. Lancez l'appli avec : python main.py")


if __name__ == "__main__":
    main()
