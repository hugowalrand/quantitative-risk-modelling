"""
Génère uniquement le tableau des métriques clés pour vérifier sa lisibilité
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Importer nos modules personnalisés
from actifs_enrichis import get_actifs_enrichis
from fiscalite import get_fiscalite_par_actif
from visualisations import configurer_style, creer_tableau_metriques

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

# Créer un DataFrame de résultats fictif pour tester la lisibilité
resultats = {
    'PEA ETF Européens': {
        'rendement_annuel': 6.5,
        'volatilite_annuelle': 17.0,
        'ratio_sharpe': 0.26,
        'taux_imposition_effectif': 17.2,
        'gain_net_apres_impots': 140951,
        'rendement_net_annualise': 5.8,
        'ratio_gain_cout': 12.01,
        'taux_reussite': 80.0,
        'drawdown_max': -25.3
    },
    'ETF d\'Actions': {
        'rendement_annuel': 7.0,
        'volatilite_annuelle': 16.0,
        'ratio_sharpe': 0.31,
        'taux_imposition_effectif': 30.0,
        'gain_net_apres_impots': 127541,
        'rendement_net_annualise': 5.5,
        'ratio_gain_cout': 10.87,
        'taux_reussite': 84.0,
        'drawdown_max': -23.5
    },
    'ETF Smart Beta': {
        'rendement_annuel': 6.5,
        'volatilite_annuelle': 15.0,
        'ratio_sharpe': 0.30,
        'taux_imposition_effectif': 30.0,
        'gain_net_apres_impots': 116060,
        'rendement_net_annualise': 5.2,
        'ratio_gain_cout': 9.89,
        'taux_reussite': 84.9,
        'drawdown_max': -22.1
    },
    'Portefeuille Mixte (80/20)': {
        'rendement_annuel': 6.2,
        'volatilite_annuelle': 14.0,
        'ratio_sharpe': 0.30,
        'taux_imposition_effectif': 30.0,
        'gain_net_apres_impots': 106937,
        'rendement_net_annualise': 4.9,
        'ratio_gain_cout': 9.11,
        'taux_reussite': 85.0,
        'drawdown_max': -20.8
    },
    'ETF d\'Actions à Faible Volatilité': {
        'rendement_annuel': 5.5,
        'volatilite_annuelle': 11.0,
        'ratio_sharpe': 0.32,
        'taux_imposition_effectif': 30.0,
        'gain_net_apres_impots': 90283,
        'rendement_net_annualise': 4.3,
        'ratio_gain_cout': 7.69,
        'taux_reussite': 89.3,
        'drawdown_max': -16.5
    },
    'ETF Obligations High Yield': {
        'rendement_annuel': 5.5,
        'volatilite_annuelle': 12.0,
        'ratio_sharpe': 0.29,
        'taux_imposition_effectif': 30.0,
        'gain_net_apres_impots': 89915,
        'rendement_net_annualise': 4.3,
        'ratio_gain_cout': 7.66,
        'taux_reussite': 86.6,
        'drawdown_max': -17.8
    },
    'Portefeuille Mixte (60/40)': {
        'rendement_annuel': 5.4,
        'volatilite_annuelle': 12.0,
        'ratio_sharpe': 0.28,
        'taux_imposition_effectif': 30.0,
        'gain_net_apres_impots': 88324,
        'rendement_net_annualise': 4.2,
        'ratio_gain_cout': 7.53,
        'taux_reussite': 86.7,
        'drawdown_max': -17.9
    },
    'Portefeuille Équilibré': {
        'rendement_annuel': 5.2,
        'volatilite_annuelle': 11.2,
        'ratio_sharpe': 0.29,
        'taux_imposition_effectif': 30.0,
        'gain_net_apres_impots': 83230,
        'rendement_net_annualise': 4.0,
        'ratio_gain_cout': 7.09,
        'taux_reussite': 87.0,
        'drawdown_max': -16.8
    },
    'Portefeuille Mixte (40/60)': {
        'rendement_annuel': 4.6,
        'volatilite_annuelle': 10.0,
        'ratio_sharpe': 0.26,
        'taux_imposition_effectif': 30.0,
        'gain_net_apres_impots': 70866,
        'rendement_net_annualise': 3.5,
        'ratio_gain_cout': 6.04,
        'taux_reussite': 86.8,
        'drawdown_max': -15.1
    },
    'Immobilier': {
        'rendement_annuel': 5.0,
        'volatilite_annuelle': 10.0,
        'ratio_sharpe': 0.30,
        'taux_imposition_effectif': 37.0,
        'gain_net_apres_impots': 69330,
        'rendement_net_annualise': 3.4,
        'ratio_gain_cout': 5.91,
        'taux_reussite': 89.7,
        'drawdown_max': -15.0
    },
    'SCPI': {
        'rendement_annuel': 4.5,
        'volatilite_annuelle': 6.0,
        'ratio_sharpe': 0.42,
        'taux_imposition_effectif': 33.0,
        'gain_net_apres_impots': 64752,
        'rendement_net_annualise': 3.2,
        'ratio_gain_cout': 5.52,
        'taux_reussite': 97.6,
        'drawdown_max': -9.1
    },
    'ETF Obligations d\'Entreprises': {
        'rendement_annuel': 4.0,
        'volatilite_annuelle': 8.0,
        'ratio_sharpe': 0.25,
        'taux_imposition_effectif': 30.0,
        'gain_net_apres_impots': 56406,
        'rendement_net_annualise': 2.8,
        'ratio_gain_cout': 4.81,
        'taux_reussite': 89.0,
        'drawdown_max': -12.1
    },
    'Obligations d\'État': {
        'rendement_annuel': 3.0,
        'volatilite_annuelle': 6.0,
        'ratio_sharpe': 0.17,
        'taux_imposition_effectif': 30.0,
        'gain_net_apres_impots': 37260,
        'rendement_net_annualise': 1.9,
        'ratio_gain_cout': 3.18,
        'taux_reussite': 88.4,
        'drawdown_max': -9.2
    }
}

# Convertir en DataFrame
resultats_df = pd.DataFrame.from_dict(resultats, orient='index')
resultats_df = resultats_df.sort_values('gain_net_apres_impots', ascending=False)

# Définir les métriques à afficher avec des descriptions très concises
metriques = [
    ('rendement_annuel', 'pct', 'Rend.', 'Perf/an'),
    ('volatilite_annuelle', 'pct', 'Volat.', 'Risque'),
    ('ratio_sharpe', 'ratio', 'Sharpe', '>1=bon'),
    ('taux_imposition_effectif', 'pct', 'Impôt', '%'),
    ('gain_net_apres_impots', 'eur', 'Gain', 'Net'),
    ('rendement_net_annualise', 'pct', 'R.Net', '%/an'),
    ('ratio_gain_cout', 'ratio', 'G/C', 'Ratio'),
    ('taux_reussite', 'pct', 'Réus.', '%'),
    ('drawdown_max', 'pct', 'DD', 'Baisse')
]

# Créer uniquement le tableau des métriques
plt.figure(figsize=(16, 10))
ax = plt.gca()
creer_tableau_metriques(ax, resultats_df, metriques, 
                       titre="Tableau des métriques clés (triées par gain net après impôts)")

# Ajouter un texte explicatif sous le tableau
ax.text(0.5, -0.05, 
       "Ce tableau présente les principales métriques pour chaque classe d'actif, triées par gain net après impôts.\n"
       "Les couleurs indiquent les performances relatives : vert = bonne performance, rouge = risque élevé, bleu = gain élevé.",
       transform=ax.transAxes, fontsize=10, ha='center',
       bbox=dict(facecolor='#f8f9fa', alpha=0.7, edgecolor='#dddddd'))

# Sauvegarder l'image
plt.tight_layout()
plt.savefig(f"{repertoire_sortie}/tableau_metriques_simplifie.png", dpi=300, bbox_inches='tight')
print(f"Tableau simplifié enregistré dans {repertoire_sortie}/tableau_metriques_simplifie.png")

# Afficher le chemin vers le fichier généré
print(f"Vous pouvez consulter le tableau simplifié ici: {os.path.abspath(f'{repertoire_sortie}/tableau_metriques_simplifie.png')}")
