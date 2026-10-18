"""Tests de l'analyse : lissage, PERCLOS, chronomètre, calibration, analyseur et décision.

Lancer avec :  pytest tests/        (depuis la racine du projet)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse.analyseur_temporel import AnalyseurTemporel
from analyse.calibration import Calibrateur
from analyse.decision import MoteurDecision
from analyse.outils_temporels import ChronometreEtat, CompteurEvenements, FenetrePerclos, LisseurExponentiel
from config import Configuration
from noyau.modeles import Etat, Indicateurs

CONFIG = Configuration.par_defaut()


def test_lissage_exponentiel_exemple_du_rapport() -> None:
    lisseur = LisseurExponentiel(0.4)
    lisseur.lisser(0.30)                          # valeur initiale
    assert abs(lisseur.lisser(0.10) - 0.22) < 1e-9
    assert abs(lisseur.lisser(0.09) - 0.168) < 1e-9


def test_perclos_et_purge_de_la_fenetre() -> None:
    fenetre = FenetrePerclos(duree=60.0)
    for seconde in range(100):                    # 100 observations, une par seconde
        fenetre.ajouter(float(seconde), ferme=(seconde % 4 == 0))   # 1 image sur 4 est « fermée »
    assert abs(fenetre.valeur() - 0.25) < 0.03
    assert len(fenetre._observations) <= 61       # seules les ~60 dernières secondes sont gardées


def test_compteur_evenements_et_purge() -> None:
    compteur = CompteurEvenements(duree=60.0)
    compteur.enregistrer(0.0)
    compteur.enregistrer(30.0)
    assert compteur.nombre(31.0) == 2
    assert compteur.nombre(70.0) == 1             # l'événement de l'instant 0 est sorti de la fenêtre


def test_chronometre_duree_en_cours_et_terminee() -> None:
    chrono = ChronometreEtat()
    assert chrono.mettre_a_jour(0.0, False) == (0.0, None)
    assert chrono.mettre_a_jour(1.0, True) == (0.0, None)
    assert chrono.mettre_a_jour(2.5, True) == (1.5, None)
    assert chrono.mettre_a_jour(3.0, False) == (0.0, 2.0)


def test_calibration_mediane_et_seuil() -> None:
    calibrateur = Calibrateur(CONFIG)
    for numero in range(60):                      # 60 mesures sur 5,9 s
        ear = 0.30 if numero % 10 else 0.05       # quelques clignements (EAR très bas)
        calibrateur.ajouter_mesure(numero * 0.1, ear, 4.0)
    assert calibrateur.terminee(5.9)
    profil = calibrateur.construire_profil()
    assert profil is not None
    assert abs(profil.seuil_ear - 0.21) < 1e-9    # 0,70 × médiane (0,30)
    assert profil.tangage_reference == 4.0


def test_calibration_faussee_est_rejetee() -> None:
    calibrateur = Calibrateur(CONFIG)
    for numero in range(60):
        calibrateur.ajouter_mesure(numero * 0.1, 0.10, 0.0)   # yeux presque fermés
    assert calibrateur.construire_profil() is None
    assert calibrateur.progression(100.0) == 0.0  # la calibration repart de zéro


def test_analyseur_detecte_une_fermeture_prolongee() -> None:
    analyseur = AnalyseurTemporel(CONFIG)
    indicateurs = None
    for dixieme in range(40):                     # 4 secondes, une image tous les 0,1 s
        instant = dixieme * 0.1
        ear = 0.30 if instant < 1.0 else 0.05     # yeux ouverts puis fermés dès t = 1 s
        indicateurs = analyseur.mettre_a_jour(instant, ear, 0.05, seuil_ear=0.21)
    assert indicateurs.ferme
    assert indicateurs.duree_fermeture > 1.5


def test_un_clignement_bref_n_est_pas_une_fermeture_longue() -> None:
    analyseur = AnalyseurTemporel(CONFIG)
    for dixieme in range(30):
        instant = dixieme * 0.1
        ear = 0.05 if 1.0 <= instant < 1.2 else 0.30    # fermeture de 0,2 s seulement
        indicateurs = analyseur.mettre_a_jour(instant, ear, 0.05, seuil_ear=0.21)
    assert indicateurs.fermetures_longues == 0


def indicateurs(duree=0.0, perclos=0.0, longues=0, baillements=0) -> Indicateurs:
    return Indicateurs(ferme=duree > 0, duree_fermeture=duree, perclos=perclos,
                       fermetures_longues=longues, baillements=baillements, ear_lisse=0.3)


def test_decision_endormi() -> None:
    moteur = MoteurDecision(CONFIG)
    assert moteur.decider(indicateurs(duree=1.6), 0.0) is Etat.ENDORMI
    assert moteur.decider(indicateurs(duree=0.9), 25.0) is Etat.ENDORMI     # tête penchée : durée réduite
    assert moteur.decider(indicateurs(duree=0.9), 5.0) is not Etat.ENDORMI


def test_decision_somnolent_et_eveille() -> None:
    moteur = MoteurDecision(CONFIG)
    assert moteur.decider(indicateurs(perclos=0.16), 0.0) is Etat.SOMNOLENT
    assert moteur.decider(indicateurs(longues=3), 0.0) is Etat.SOMNOLENT
    assert moteur.decider(indicateurs(baillements=2), 0.0) is Etat.SOMNOLENT
    assert moteur.decider(indicateurs(perclos=0.05), 0.0) is Etat.EVEILLE


def test_configuration_incoherente_est_refusee() -> None:
    import dataclasses
    mauvaise = dataclasses.replace(CONFIG, alpha_lissage=1.5)
    try:
        mauvaise.valider()
    except ValueError:
        return
    raise AssertionError("Une configuration incohérente aurait dû être refusée.")
