"""
Tableau de bord d'investissement final - Partie 1: Simulation et préparation des données
Version finale avec toutes les améliorations de lisibilité et de visualisation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import sys
from datetime import datetime
import matplotlib.ticker as mtick

# Importer nos modules personnalisés
sys.path.append('..')
from src.actifs_enrichis import get_actifs_enrichis
from src.fiscalite import get_fiscalite_par_actif
from src.visualisation_utils import configurer_style, creer_tableau_metriques, creer_graphique_barres, creer_graphique_ligne

# Définir le répertoire de sortie
repertoire_sortie = "../output"
if not os.path.exists(repertoire_sortie):
    os.makedirs(repertoire_sortie)

# Configurer le style
colors = configurer_style()

# Paramètres de la simulation
np.random.seed(42)  # Pour la reproductibilité
montant_pret = 200_000
taux_interet_annuel = 0.0099  # Prêt étudiant avantageux à 0,99%
mois_differe = 18
mois_remboursement = 102
total_mois = mois_differe + mois_remboursement
n_simulations = 10_000
duree_ans = total_mois / 12

# Calculer le prêt
taux_interet_mensuel = taux_interet_annuel / 12
montant_differe = montant_pret * (1 + taux_interet_mensuel) ** mois_differe
paiement_mensuel = montant_differe * (taux_interet_mensuel * (1 + taux_interet_mensuel)**mois_remboursement) / ((1 + taux_interet_mensuel)**mois_remboursement - 1)
cout_total_interet = (paiement_mensuel * mois_remboursement) - montant_pret

# Afficher les détails du prêt
print(f"Montant initial du prêt : {montant_pret:,.2f} €")
print(f"Montant différé après {mois_differe} mois : {montant_differe:,.2f} €")
print(f"Paiement mensuel pendant la période de remboursement : {paiement_mensuel:,.2f} €")
print(f"Total des paiements sur la durée du prêt : {paiement_mensuel * mois_remboursement:,.2f} €")
print(f"Coût total du prêt (intérêts) : {cout_total_interet:,.2f} €")

# Obtenir les classes d'actifs enrichies
actifs = get_actifs_enrichis()
fiscalite_par_actif = get_fiscalite_par_actif()

# Fonction pour simuler l'évolution d'un portefeuille
def simuler_portefeuille(actif, n_simulations, total_mois, montant_initial):
    rendement_mensuel = actif['rendement_annuel'] / 12
    volatilite_mensuelle = actif['volatilite_annuelle'] / np.sqrt(12)
    
    # Initialiser un array pour stocker les résultats des simulations
    resultats_simulation = np.zeros((n_simulations, total_mois+1))
    resultats_simulation[:, 0] = montant_initial
    
    # Simuler n_simulations trajectoires
    for i in range(n_simulations):
        for mois in range(1, total_mois+1):
            # Utiliser la loi log-normale pour les rendements mensuels
            rendement = np.random.normal(rendement_mensuel, volatilite_mensuelle)
            resultats_simulation[i, mois] = resultats_simulation[i, mois-1] * (1 + rendement)
    
    return resultats_simulation

# Simuler tous les actifs
resultats = {}
donnees_mensuelles = {}

for nom_actif, actif in actifs.items():
    print(f"Simulation de {nom_actif}...")
    # Simuler le portefeuille
    simulations = simuler_portefeuille(actif, n_simulations, total_mois, montant_pret)
    
    # Stocker les données mensuelles pour les graphiques
    donnees_mensuelles[nom_actif] = simulations
    
    # Calculer les métriques de base
    valeurs_finales = simulations[:, -1]
    gain_brut_moyen = np.mean(valeurs_finales) - montant_pret
    ecart_type_gain = np.std(valeurs_finales)
    rendement_annualise = (np.mean(valeurs_finales) / montant_pret) ** (1 / duree_ans) - 1
    volatilite_annuelle = actif['volatilite_annuelle']
    
    # Calculer le ratio de Sharpe (rendement excédentaire / volatilité)
    # Utilisation d'un taux sans risque de 0% pour simplifier
    ratio_sharpe = rendement_annualise / volatilite_annuelle if volatilite_annuelle > 0 else 0
    
    # Calculer les métriques de réussite
    cout_total = montant_pret + cout_total_interet
    taux_reussite = np.mean(valeurs_finales > cout_total) * 100
    
    # Calculer le drawdown maximum
    drawdowns = []
    for sim in simulations:
        # Calculer les drawdowns pour chaque simulation
        peak = sim[0]
        sim_drawdowns = []
        for val in sim[1:]:
            if val > peak:
                peak = val
            sim_drawdowns.append((val - peak) / peak * 100)  # En pourcentage
        drawdowns.append(min(sim_drawdowns) if sim_drawdowns else 0)
    drawdown_max = np.mean(drawdowns)
    
    # Calculer les métriques fiscales
    fiscalite = fiscalite_par_actif.get(nom_actif, {})
    taux_imposition = fiscalite.get('taux_imposition', 0.30)  # Taux par défaut: 30%
    taux_abattement = fiscalite.get('abattement', 0)  # Abattement par défaut: 0%
    
    # Calculer les impôts et le gain net après impôts
    base_imposable = gain_brut_moyen * (1 - taux_abattement)
    impots = base_imposable * taux_imposition
    gain_net_avant_impots = gain_brut_moyen - cout_total_interet
    gain_net_apres_impots = gain_brut_moyen - impots - cout_total_interet
    
    # Calculer le taux d'imposition effectif (impôts / gain brut)
    taux_imposition_effectif = (impots / gain_brut_moyen) * 100 if gain_brut_moyen > 0 else 0
    
    # Calculer le rendement net annualisé (après impôts)
    rendement_net_annualise = ((montant_pret + gain_net_apres_impots) / montant_pret) ** (1 / duree_ans) - 1
    
    # Calculer le ratio gain/coût (combien d'euros gagnés pour chaque euro d'intérêt payé)
    ratio_gain_cout = gain_net_apres_impots / cout_total_interet if cout_total_interet > 0 else 0
    
    # Stocker les résultats
    resultats[nom_actif] = {
        'rendement_annuel': actif['rendement_annuel'] * 100,
        'volatilite_annuelle': actif['volatilite_annuelle'] * 100,
        'ratio_sharpe': ratio_sharpe,
        'gain_brut_moyen': gain_brut_moyen,
        'ecart_type_gain': ecart_type_gain,
        'rendement_annualise': rendement_annualise * 100,
        'taux_reussite': taux_reussite,
        'drawdown_max': drawdown_max,
        'taux_imposition': taux_imposition * 100,
        'taux_abattement': taux_abattement * 100,
        'impots': impots,
        'gain_net_avant_impots': gain_net_avant_impots,
        'gain_net_apres_impots': gain_net_apres_impots,
        'taux_imposition_effectif': taux_imposition_effectif,
        'rendement_net_annualise': rendement_net_annualise * 100,
        'ratio_gain_cout': ratio_gain_cout,
        'couleur': colors.get(nom_actif, '#333333')
    }

# Convertir les résultats en DataFrame pour faciliter la manipulation
resultats_df = pd.DataFrame.from_dict(resultats, orient='index')

# Trier par gain net après impôts (du plus élevé au plus bas)
resultats_df = resultats_df.sort_values('gain_net_apres_impots', ascending=False)

# Enregistrer les résultats dans un fichier CSV pour référence
resultats_df.to_csv(f"{repertoire_sortie}/recap_actifs_avec_fiscalite.csv")

# Continuer dans la partie 2...
print("Données préparées avec succès")

# Exporter les données pour être utilisées dans la partie 2
# Utiliser un format plus fiable pour l'échange de données
resultats_df.to_pickle(f"{repertoire_sortie}/resultats_df.pkl")

# Sauvegarder les données mensuelles et paramètres sous forme de dictionnaire
donnees_a_sauvegarder = {
    'donnees_mensuelles': donnees_mensuelles,
    'montant_pret': montant_pret,
    'cout_total_interet': cout_total_interet,
    'duree_ans': duree_ans,
    'mois_differe': mois_differe,
    'mois_remboursement': mois_remboursement,
    'total_mois': total_mois,
    'taux_interet_annuel': taux_interet_annuel
}
pd.to_pickle(donnees_a_sauvegarder, f"{repertoire_sortie}/parametres_simulation.pkl")
