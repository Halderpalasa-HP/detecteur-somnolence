# Détection de somnolence au volant

Projet de TP d'intelligence artificielle : une caméra filme le visage du conducteur, le système détecte
la somnolence et le sommeil, puis déclenche une **sonnerie**, attend **1 seconde**, et enchaîne un
**message vocal** qui déconseille de dormir au volant. L'interface est un tableau de bord clinique avec
un **masque de visage** en direct. Tout fonctionne hors ligne, la vidéo n'est jamais enregistrée.

> Prototype pédagogique : il ne remplace ni la vigilance du conducteur ni le repos.

## 1. Installation

```
pip install -r requirements.txt
```

Python 3.10 ou plus récent. Une webcam et un haut-parleur sont nécessaires.

## 2. Télécharger le modèle MediaPipe

Téléchargez `face_landmarker.task` et placez-le dans le dossier `assets/` :

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Si ce lien ne fonctionne plus, cherchez « MediaPipe Face Landmarker » dans la documentation officielle
(rubrique *Models*).

## 3. Générer les sons (une seule fois)

```
python scripts/generer_assets.py
```

Cela crée `assets/sonnerie.wav` (cloche synthétisée) et `assets/message.wav` (voix française, avec pyttsx3).
Si aucune voix française n'est installée sur votre système, installez-en une (Windows : Paramètres → Heure et
langue → Voix) ou enregistrez vous-même `assets/message.wav`.

## 4. Lancer

```
python main.py
```

1. **Calibration (5 s)** : regardez la route, yeux ouverts. Le système mesure votre EAR « yeux ouverts ».
2. **Surveillance** : le masque suit votre visage, les cartes affichent les mesures en direct.
3. Fermez les yeux plus de 1,5 s : état **ENDORMI** → sonnerie, pause de 1 s, message vocal.
   Rouvrez les yeux : l'alerte s'arrête aussitôt.

| Commande | Effet |
|---|---|
| Touche **M** ou bouton MAILLAGE | affiche / masque le maillage |
| Touche **G** ou bouton GUIDES | affiche / masque repères d'angle et croix |
| Touche **V** ou bouton VALEURS | affiche / masque les valeurs |
| **Espace** ou boutons DÉMARRER / ARRÊTER | démarre ou arrête la caméra (recalibre au redémarrage) |
| **Q** ou **Échap** | quitte |

## 5. Tests

```
pytest tests/
```

## 6. Organisation du code

Tous les noms (dossiers, classes, méthodes, commentaires) sont en français. Aucun fichier ne dépasse
155 lignes. Les réglages sont tous dans `config.py`.

| Dossier / fichier | Rôle |
|---|---|
| `main.py` | point d'entrée |
| `app.py` | boucle principale (caméra, pipeline, alerte, interface) |
| `pipeline.py` | calcul de bout en bout pour chaque image |
| `config.py` | tous les réglages (seuils, durées, chemins) |
| `noyau/` | `modeles.py` (objets de données), `camera.py`, `suivi_visage.py` (MediaPipe) |
| `mesures/` | `mesures_visage.py` (EAR, MAR), `pose_tete.py` (solvePnP) |
| `analyse/` | `calibration.py`, `outils_temporels.py`, `analyseur_temporel.py`, `decision.py` |
| `alerte/` | `generation_audio.py`, `gestionnaire_alerte.py` |
| `interface/` | `dessin.py`, `liaisons.py`, `rendu_masque.py`, `cartes.py`, `carte_courbe.py`, `tableau_de_bord.py` |
| `scripts/` | `generer_assets.py` |
| `tests/` | tests unitaires |
| `docs/plan_conception.html` | plan de conception (classes et méthodes) |

Chaque usage d'un modèle ou d'un mot-clé important est annoté par un commentaire `APPORT :`.

## 7. Guide pour expliquer chaque module à l'oral

Pour chaque module, répondre à : **entrée → traitement → sortie → pourquoi**.

- **camera.py** : la caméra fournit des images BGR (tableaux de pixels) avec leur instant. Les durées seront
  toutes en secondes car la cadence varie.
- **suivi_visage.py** : MediaPipe. BlazeFace (CNN léger, détection par ancres) trouve le visage, puis Face Landmark
  (CNN de régression) prédit 478 points. Aux images suivantes, le suivi évite de relancer la détection.
- **mesures_visage.py** : EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖). Un rapport de distances : indépendant de la
  distance à la caméra. MAR : même principe pour la bouche.
- **pose_tete.py** : `solvePnP` retrouve la rotation 3D de la tête à partir de 6 points ; Rodrigues donne la
  matrice, puis les angles. Une tête qui tombe confirme un endormissement.
- **calibration.py** : seuil = 0,70 × médiane de l'EAR yeux ouverts. La médiane ignore les clignements.
- **outils_temporels.py** : lissage exponentiel s = α·x + (1−α)·s ; PERCLOS = temps yeux fermés / 60 s ;
  compteurs d'événements ; chronomètre de fermeture.
- **analyseur_temporel.py** : assemble ces outils et produit les indicateurs.
- **decision.py** : règles ordonnées par gravité (ENDORMI > SOMNOLENT > ÉVEILLÉ), donc explicables.
- **gestionnaire_alerte.py** : un thread joue sonnerie → pause 1 s → voix, sans figer la vidéo ; un `Event`
  permet l'arrêt immédiat quand les yeux se rouvrent.
- **interface/** : le masque (maillage translucide, contours par élément, yeux colorés selon l'état) et les cartes
  rendent le traitement interne visible.

## 8. Limites

Lunettes (surtout de soleil), faible luminosité, visage tourné, diversité des visages (à tester sur plusieurs
personnes). Les seuils par défaut (0,70 ; 1,5 s ; PERCLOS 0,15 ; 20°) sont des points de départ à ajuster avec
vos propres mesures dans `config.py`.
