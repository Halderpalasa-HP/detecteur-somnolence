"""Tableau de bord clinique : composition de l'écran, souris et clavier.

================================================================================
CE QUE FAIT CE FICHIER (BLOC 4 : TABLEAU DE BORD = COMPOSITION FINALE)
================================================================================
ASSEMBLE en UNE image 1280×720 :
- En-titre : titre + "FLUX OPTIQUE EN DIRECT" + FPS
- GAUCHE (4:3) : vidéo caméra + masque visage superposé
- DROITE : 4 cartes empilées (ÉTAT, MESURES, COURBE, ALERTE)
- BAS : interrupteurs (MAILLAGE/GUIDES/VALEURS) + boutons (DÉMARRER/ARRÊTER)
- Mentions : "Vidéo traitée localement, jamais enregistrée"

TRADUIT les entrées utilisateur :
- Clavier : M/G/V = toggles, ESPACE = start/stop, Q/Échap = quit
- Souris : clic sur interrupteurs/boutons (via cv2.setMouseCallback)

POURQUOI CETTE CLASSE SEULE CONNAÎT LA MISE EN PAGE ?
- Un SEUL endroit pour les coordonnées (constantes VIDEO, COLONNE_X1/X2)
- Changement de résolution/disposition = modifier ICI seulement
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from config import Configuration
from interface.carte_courbe import CarteCourbe
from interface.cartes import CarteAlerte, CarteEtat, CarteMesures
from interface.dessin import Dessinateur, Theme
from interface.rendu_masque import RenduMasque
from noyau.modeles import Etat, OptionsAffichage, SortieTraitement

Rectangle = Tuple[int, int, int, int]

# Zone de la vidéo (rapport 4:3 conservé) et colonne des cartes.
VIDEO = (24, 84, 770, 644)
COLONNE_X1, COLONNE_X2 = 794, 1256


class TableauDeBord:
    """Compose l'écran complet et traduit les clics et les touches en ordres."""

    def __init__(self, config: Configuration, rendu_masque: RenduMasque) -> None:
        self._config = config
        self._rendu_masque = rendu_masque
        self._dessinateur = Dessinateur()
        self.options = OptionsAffichage()
        self.en_marche = True            # la caméra est-elle démarrée ?
        self.quitter_demande = False     # l'utilisateur a-t-il demandé à quitter ?
        self._zones: Dict[str, Rectangle] = {}   # zones cliquables de la barre basse

        x1, x2 = COLONNE_X1, COLONNE_X2
        self._carte_etat = CarteEtat("ETAT DU CONDUCTEUR", (x1, 84, x2, 196))
        self._carte_mesures = CarteMesures("MESURES EN DIRECT", (x1, 208, x2, 420))
        self._carte_courbe = CarteCourbe("ANALYSE TEMPORELLE - EAR", (x1, 432, x2, 552), config.duree_courbe)
        self._carte_alerte = CarteAlerte("ALERTE SONORE ET VOCALE", (x1, 564, x2, 644))

    def composer(self, image: np.ndarray, sortie: SortieTraitement, statut_alerte: str,
                 fps: float) -> np.ndarray:
        """Assemble l'image finale de l'écran.

        POURQUOI : dernier maillon avant l'affichage ; un seul endroit connaît la mise en page.
        """
        canevas = np.full((self._config.hauteur_fenetre, self._config.largeur_fenetre, 3),
                          Theme.FOND, dtype=np.uint8)
        self._entete(canevas, fps)

        # Vidéo : on dessine le masque sur l'image de la caméra, puis on l'adapte à la zone.
        x1, y1, x2, y2 = VIDEO
        if self.en_marche:
            self._rendu_masque.dessiner(image, sortie.visage, sortie.etat, self.options)
        canevas[y1:y2, x1:x2] = cv2.resize(image, (x2 - x1, y2 - y1))
        cv2.rectangle(canevas, (x1, y1), (x2, y2), Theme.BORDURE, 1)
        if not self.en_marche:
            self._dessinateur.texte(canevas, "CAMERA ARRETEE", ((x1 + x2) // 2, (y1 + y2) // 2),
                                    0.9, Theme.GRIS, 2, "centre")

        # Cartes de données.
        if self.en_marche and sortie.visage is not None:
            self._carte_courbe.ajouter_point(sortie.instant, sortie.ear)
        self._carte_etat.dessiner(canevas, sortie if self.en_marche else None)
        self._carte_mesures.dessiner(canevas, sortie)
        self._carte_courbe.dessiner(canevas, sortie)
        self._carte_alerte.dessiner(canevas, statut_alerte)

        self._barre_basse(canevas)
        return canevas

    def gerer_touche(self, touche: int) -> bool:
        """Traite une touche du clavier ; renvoie `True` si l'utilisateur veut quitter.

        POURQUOI : contrôle au clavier : M (maillage), G (guides), V (valeurs), espace
        (démarrer/arrêter), Q ou Échap (quitter).
        """
        if touche in (ord("q"), ord("Q"), 27):
            return True
        if touche in (ord("m"), ord("M")):
            self.options.basculer("maillage")
        elif touche in (ord("g"), ord("G")):
            self.options.basculer("guides")
        elif touche in (ord("v"), ord("V")):
            self.options.basculer("valeurs")
        elif touche == ord(" "):
            self.en_marche = not self.en_marche
        return False

    def gerer_souris(self, evenement: int, x: int, y: int, drapeaux: int = 0, parametre=None) -> None:
        """Traite un clic de souris sur un interrupteur ou un bouton.

        POURQUOI : contrôle à la souris, fidèle à l'interface de la capture. Cette méthode
        est branchée sur la fenêtre avec `cv2.setMouseCallback`.
        """
        if evenement != cv2.EVENT_LBUTTONDOWN:
            return
        for nom, (x1, y1, x2, y2) in self._zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if nom in ("maillage", "guides", "valeurs"):
                    self.options.basculer(nom)
                elif nom == "demarrer":
                    self.en_marche = True
                elif nom == "arreter":
                    self.en_marche = False
                return

    def _entete(self, canevas: np.ndarray, fps: float) -> None:
        """Dessine le titre, l'étiquette "FLUX OPTIQUE EN DIRECT" et les FPS.

        POURQUOI : reprendre l'en-tête de la capture clinique et afficher la cadence réelle.
        """
        self._dessinateur.etiquette(canevas, "FLUX OPTIQUE EN DIRECT", (24, 30))
        self._dessinateur.texte(canevas, "Moniteur de somnolence", (24, 62), 0.9, Theme.TEXTE, 2)
        self._dessinateur.texte(canevas, f"{fps:.0f} FPS", (COLONNE_X2, 40), 0.55, Theme.TEXTE_DISCRET,
                                1, "droite")

    def _barre_basse(self, canevas: np.ndarray) -> None:
        """Dessine les interrupteurs, les boutons et la mention de confidentialité.

        POURQUOI : rappeler que la vidéo est traitée localement et n'est jamais enregistrée.
        """
        y1, y2 = 664, 700
        elements = (
            ("maillage", "MAILLAGE", 24, 130, self.options.maillage),
            ("guides", "GUIDES", 142, 232, self.options.guides),
            ("valeurs", "VALEURS", 244, 340, self.options.valeurs),
            ("demarrer", "DEMARRER", 794, 920, self.en_marche),
            ("arreter", "ARRETER", 932, 1050, not self.en_marche),
        )
        self._zones = {}
        for nom, libelle, x1, x2, actif in elements:
            self._zones[nom] = (x1, y1, x2, y2)
            self._dessinateur.rectangle_arrondi(canevas, (x1, y1), (x2, y2),
                                                Theme.ACCENT if actif else Theme.BORDURE,
                                                rayon=8, epaisseur=1, rempli=False)
            couleur_texte = Theme.ACCENT if actif else Theme.TEXTE_DISCRET
            self._dessinateur.texte(canevas, libelle, ((x1 + x2) // 2, y1 + 23), 0.45, couleur_texte, 1, "centre")
        self._dessinateur.texte(canevas, "Video traitee localement, jamais enregistree.",
                                (24, 712), 0.42, Theme.TEXTE_DISCRET)
