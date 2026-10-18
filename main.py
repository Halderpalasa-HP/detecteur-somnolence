"""Point d'entrée du programme : `python main.py`.

================================================================================
ARCHITECTURE GLOBALE DU PROJET
================================================================================
Ce projet détecte la somnolence au volant en temps réel. Voici comment les
blocs s'articulent (numéros = ordre dans le pipeline) :

  1. ACQUISITION (camera.py)         → Image BGR + horodatage (secondes)
  2. DÉTECTION VISAGE (suivi_visage.py) → 478 points du visage (x, y en pixels)
  3. MESURES (mesures_visage.py)     → EAR (yeux), MAR (bouche), pose tête
  4. CALIBRATION (calibration.py)    → Seuil EAR personnel + tangage de référence
  5. ANALYSE TEMPORELLE (analyseur_temporel.py) → Lissage, PERCLOS, compteurs
  6. DÉCISION (decision.py)          → État : CALIBRATION / ÉVEILLÉ / SOMNOLENT / ENDORMI / VISAGE_PERDU
  7. ALERTE (gestionnaire_alerte.py) → Sonnerie → pause 1s → message vocal (thread à part)
  8. INTERFACE (tableau_de_bord.py)  → Fenêtre 1280×720 : vidéo + masque + cartes + commandes

POURQUOI CETTE SÉPARATION ?
- Chaque bloc a UNE responsabilité unique → facile à tester, expliquer, modifier
- Le pipeline (pipeline.py) enchaîne les blocs 2→6 SANS connaître l'affichage
- L'interface (bloc 8) ne fait QUE de l'affichage, aucun calcul
- L'alerte (bloc 7) tourne dans un thread à part pour ne JAMAIS geler la vidéo

ERREURS GÉRÉES ICI :
- FileNotFoundError     : modèle .task ou fichiers .wav manquants
- RuntimeError          : caméra indisponible (branchement, autre appli)
- ValueError            : configuration invalide (seuils incohérents)
"""

from __future__ import annotations

import sys

from app import Application


def main() -> int:
    """Lance l'application ; renvoie 0 si tout va bien, 1 en cas d'erreur de démarrage."""
    try:
        Application().lancer()
    except (FileNotFoundError, RuntimeError, ValueError) as erreur:
        # Fichier modèle ou son absent, caméra introuvable, réglage incohérent…
        print(f"Erreur : {erreur}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
