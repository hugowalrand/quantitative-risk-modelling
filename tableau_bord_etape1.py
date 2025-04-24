"""
Tableau de bord d'investissement enrichi - Étape 1
Ce script simule différents scénarios d'investissement avec un prêt étudiant à 0,99%
et génère un tableau de bord visuel pour comparer les options.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from actifs_enrichis import get_actifs_enrichis

# Définir le style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.figsize'] = (20, 24)

# Créer le répertoire de sortie s'il n'existe pas
repertoire_sortie = "../output"
if not os.path.exists(repertoire_sortie):
    os.makedirs(repertoire_sortie)

# Paramètres de la simulation
np.random.seed(42)  # Pour la reproductibilité
montant_pret = 200_000
taux_interet_annuel = 0.0099  # Prêt étudiant avantageux à 0,99%
mois_differe = 18
mois_remboursement = 102
total_mois = mois_differe + mois_remboursement
n_simulations = 10_000

# Calculer le prêt
taux_interet_mensuel = taux_interet_annuel / 12
montant_differe = montant_pret * (1 + taux_interet_mensuel) ** mois_differe
paiement_mensuel = montant_differe * (taux_interet_mensuel * (1 + taux_interet_mensuel)**mois_remboursement) / ((1 + taux_interet_mensuel)**mois_remboursement - 1)
cout_total_interet = (paiement_mensuel * mois_remboursement) - montant_pret

# Afficher les informations sur le prêt
print(f"Montant initial du prêt : {montant_pret:,.2f} €")
print(f"Montant différé après {mois_differe} mois : {montant_differe:,.2f} €")
print(f"Paiement mensuel pendant la période de remboursement : {paiement_mensuel:,.2f} €")
print(f"Total des paiements sur la durée du prêt : {paiement_mensuel * mois_remboursement:,.2f} €")
print(f"Coût total du prêt (intérêts) : {cout_total_interet:,.2f} €")

# Obtenir les actifs enrichis
actifs = get_actifs_enrichis()

# Fonction pour calculer le ratio de Sharpe
def calculer_ratio_sharpe(rendement_annuel, volatilite_annuelle, taux_sans_risque=0.02):
    """Calcule le ratio de Sharpe (rendement excédentaire par unité de risque)"""
    return (rendement_annuel - taux_sans_risque) / volatilite_annuelle if volatilite_annuelle > 0 else float('inf')

# Fonction pour simuler l'évolution du portefeuille
def simuler_portefeuille(actif, n_simulations, total_mois, montant_initial):
    """Simuler l'évolution d'un portefeuille avec des rendements aléatoires"""
    rendement_mensuel = actif['rendement_annuel'] / 12
    volatilite_mensuelle = actif['volatilite_annuelle'] / np.sqrt(12)
    
    # Initialiser les tableaux pour les simulations
    valeurs = np.zeros((n_simulations, total_mois + 1))
    valeurs[:, 0] = montant_initial
    
    # Simuler les rendements mensuels
    for t in range(total_mois):
        # Générer des rendements aléatoires pour ce mois
        r = np.random.normal(rendement_mensuel, volatilite_mensuelle, n_simulations)
        
        # Appliquer les rendements au portefeuille
        valeurs[:, t+1] = valeurs[:, t] * (1 + r)
    
    return valeurs

# Fonction pour calculer le drawdown maximum
def calculer_drawdown_max(valeurs):
    """Calcule le drawdown maximum pour chaque simulation"""
    drawdowns = np.zeros(valeurs.shape[0])
    
    for i in range(valeurs.shape[0]):
        # Calculer les drawdowns pour cette simulation
        serie = valeurs[i, :]
        max_jusqu_ici = np.maximum.accumulate(serie)
        drawdown = (serie - max_jusqu_ici) / max_jusqu_ici
        drawdowns[i] = np.min(drawdown)
    
    return np.mean(drawdowns) * 100  # En pourcentage

# Simuler tous les actifs
resultats = {}
donnees_mensuelles = {}

# Sélectionner les actifs les plus pertinents pour l'analyse
actifs_a_simuler = [
    'ETF d\'Actions', 
    'Obligations d\'État', 
    'Immobilier',
    'Portefeuille Mixte (60/40)',
    'Portefeuille Mixte (80/20)',
    'Portefeuille Mixte (40/60)',
    'Portefeuille Équilibré',
    'ETF d\'Actions à Faible Volatilité',
    'ETF Obligations d\'Entreprises',
    'ETF Obligations High Yield',
    'SCPI',
    'ETF Smart Beta',
    'PEA ETF Européens'
]

# Filtrer les actifs à simuler
actifs_filtres = {k: v for k, v in actifs.items() if k in actifs_a_simuler}

for nom_actif, actif in actifs_filtres.items():
    print(f"Simulation de {nom_actif}...")
    valeurs = simuler_portefeuille(actif, n_simulations, total_mois, montant_pret)
    
    # Calculer les statistiques sur les valeurs finales
    valeurs_finales = valeurs[:, -1]
    
    # Calculer les résultats pour cet actif
    resultats[nom_actif] = {
        'nom': actif['nom'],
        'description': actif['description'],
        'profil_risque': actif['profil_risque'],
        'horizon_recommande': actif['horizon_recommande'],
        'rendement_annuel': actif['rendement_annuel'] * 100,
        'volatilite_annuelle': actif['volatilite_annuelle'] * 100,
        'ratio_sharpe': calculer_ratio_sharpe(actif['rendement_annuel'], actif['volatilite_annuelle']),
        'moyenne_valeur_finale': np.mean(valeurs_finales),
        'mediane_valeur_finale': np.median(valeurs_finales),
        'ecart_type_valeur_finale': np.std(valeurs_finales),
        'gain_brut_moyen': np.mean(valeurs_finales) - montant_pret,
        'gain_net_moyen': np.mean(valeurs_finales) - montant_pret - cout_total_interet,
        'rendement_net_annualise': ((np.mean(valeurs_finales) / montant_pret) ** (1/10) - 1) * 100,
        'ratio_gain_cout': (np.mean(valeurs_finales) - montant_pret - cout_total_interet) / cout_total_interet,
        'valeur_risque_5': np.percentile(valeurs_finales, 5),
        'valeur_risque_1': np.percentile(valeurs_finales, 1),
        'percentile_25': np.percentile(valeurs_finales, 25),
        'percentile_75': np.percentile(valeurs_finales, 75),
        'percentile_95': np.percentile(valeurs_finales, 95),
        'min': np.min(valeurs_finales),
        'max': np.max(valeurs_finales),
        'taux_reussite': np.mean(valeurs_finales > (montant_pret + cout_total_interet)) * 100,
        'drawdown_max': calculer_drawdown_max(valeurs),
        'couleur': actif['couleur'] if 'couleur' in actif else '#333333'
    }
    
    # Stocker les données mensuelles
    donnees_mensuelles[nom_actif] = {
        'moyenne': np.mean(valeurs, axis=0),
        'mediane': np.median(valeurs, axis=0),
        'min': np.min(valeurs, axis=0),
        'max': np.max(valeurs, axis=0),
        'percentile_5': np.percentile(valeurs, 5, axis=0),
        'percentile_25': np.percentile(valeurs, 25, axis=0),
        'percentile_75': np.percentile(valeurs, 75, axis=0),
        'percentile_95': np.percentile(valeurs, 95, axis=0),
        'valeurs': valeurs
    }

# Convertir les résultats en DataFrame pour faciliter l'analyse
resultats_df = pd.DataFrame.from_dict(resultats, orient='index')
resultats_df = resultats_df.sort_values('gain_net_moyen', ascending=False)

# Créer un tableau de bord visuel complet
plt.figure(figsize=(20, 24))
gs = GridSpec(6, 2, figure=plt.gcf(), height_ratios=[0.8, 1, 1, 1, 1, 1])

# Titre et paramètres
plt.suptitle("TABLEAU DE BORD - ANALYSE DES OPTIONS D'INVESTISSEMENT ENRICHIE", fontsize=24, y=0.995)
texte_param = (
    f"Paramètres : Prêt étudiant {montant_pret:,.0f} € à {taux_interet_annuel*100:.2f}% | "
    f"Différé : {mois_differe} mois (1,5 ans) | "
    f"Remboursement : {mois_remboursement} mois (8,5 ans) | "
    f"Total : 10 ans | "
    f"Coût total des intérêts : {cout_total_interet:,.0f} €"
)
plt.figtext(0.5, 0.975, texte_param, ha="center", fontsize=14)

# 1. Tableau des métriques clés avec focus sur le gain réel et le risque
ax_metrics = plt.subplot(gs[0, :])
ax_metrics.axis('off')

# Préparer les données pour le tableau
metrics_data = []
for nom_actif in resultats_df.index:
    metrics_data.append([
        nom_actif,
        f"{resultats_df.loc[nom_actif, 'rendement_annuel']:.1f}%",
        f"{resultats_df.loc[nom_actif, 'volatilite_annuelle']:.1f}%",
        f"{resultats_df.loc[nom_actif, 'ratio_sharpe']:.2f}",
        f"{resultats_df.loc[nom_actif, 'gain_net_moyen']:,.0f} €",
        f"{resultats_df.loc[nom_actif, 'rendement_net_annualise']:.2f}%",
        f"{resultats_df.loc[nom_actif, 'ratio_gain_cout']:.2f}",
        f"{resultats_df.loc[nom_actif, 'taux_reussite']:.1f}%",
        f"{resultats_df.loc[nom_actif, 'drawdown_max']:.1f}%"
    ])

metric_labels = [
    'Classe d\'Actif', 'Rend. Annuel', 'Volatilité', 'Ratio Sharpe',
    'Gain Net Moyen', 'Rend. Net Annualisé', 'Ratio Gain/Coût', 
    'Taux de Réussite', 'Drawdown Max'
]

metrics_table = ax_metrics.table(
    cellText=metrics_data,
    colLabels=metric_labels,
    loc='center',
    cellLoc='center'
)
metrics_table.auto_set_font_size(False)
metrics_table.set_fontsize(10)
metrics_table.scale(1, 1.5)

# Mettre en forme le tableau
for i in range(len(resultats_df.index)):
    for j in range(len(metric_labels)):
        cell = metrics_table[(i+1, j)]
        
        # Colorer les cellules selon les valeurs
        if j == 2:  # Volatilité
            val = resultats_df.iloc[i]['volatilite_annuelle']
            min_val = resultats_df['volatilite_annuelle'].min()
            max_val = resultats_df['volatilite_annuelle'].max()
            norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            color = (norm_val, 1-norm_val, 0.3)  # Rouge = plus volatile
            cell.set_facecolor(color)
        elif j == 3:  # Ratio Sharpe
            val = resultats_df.iloc[i]['ratio_sharpe']
            min_val = resultats_df['ratio_sharpe'].min()
            max_val = resultats_df['ratio_sharpe'].max()
            norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            color = (1-norm_val, norm_val, 0.3)  # Vert = meilleur ratio
            cell.set_facecolor(color)
        elif j == 7:  # Taux de réussite
            val = resultats_df.iloc[i]['taux_reussite']
            color = (1-val/100, val/100, 0.3)
            cell.set_facecolor(color)
        elif j == 8:  # Drawdown Max
            val = abs(resultats_df.iloc[i]['drawdown_max'])
            min_val = abs(resultats_df['drawdown_max'].min())
            max_val = abs(resultats_df['drawdown_max'].max())
            norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            color = (norm_val, 1-norm_val, 0.3)  # Rouge = drawdown plus important
            cell.set_facecolor(color)

# Mettre en forme l'en-tête
for j in range(len(metric_labels)):
    cell = metrics_table[(0, j)]
    cell.set_facecolor('#4c72b0')
    cell.set_text_props(weight='bold', color='white')

# 2. Graphique comparatif des gains nets
ax_gains = plt.subplot(gs[1, 0])
gains_nets = resultats_df['gain_net_moyen']
bars = ax_gains.bar(gains_nets.index, gains_nets.values, color=[resultats_df.loc[nom, 'couleur'] for nom in gains_nets.index])

ax_gains.set_title('Gain Net Moyen par Classe d\'Actif', fontsize=16)
ax_gains.set_ylabel('Gain Net (€)', fontsize=14)
ax_gains.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax_gains.grid(True, alpha=0.3)

# Ajouter les valeurs sur les barres
for i, v in enumerate(gains_nets):
    ax_gains.text(i, v + (5000 if v > 0 else -5000), 
                 f"{v:,.0f} €", 
                 ha='center', fontsize=10)

plt.setp(ax_gains.get_xticklabels(), rotation=45, ha='right')

# 3. Graphique comparatif des ratios de Sharpe
ax_sharpe = plt.subplot(gs[1, 1])
sharpe_ratios = resultats_df['ratio_sharpe'].sort_values(ascending=False)
bars = ax_sharpe.bar(sharpe_ratios.index, sharpe_ratios.values, 
                    color=[resultats_df.loc[nom, 'couleur'] for nom in sharpe_ratios.index])

ax_sharpe.set_title('Ratio de Sharpe par Classe d\'Actif', fontsize=16)
ax_sharpe.set_ylabel('Ratio de Sharpe', fontsize=14)
ax_sharpe.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax_sharpe.grid(True, alpha=0.3)

# Ajouter les valeurs sur les barres
for i, v in enumerate(sharpe_ratios):
    ax_sharpe.text(i, v + 0.05, 
                  f"{v:.2f}", 
                  ha='center', fontsize=10)

plt.setp(ax_sharpe.get_xticklabels(), rotation=45, ha='right')

# 4. Distribution des valeurs finales
ax_dist = plt.subplot(gs[2, :])
for nom_actif in resultats_df.index[:7]:  # Limiter aux 7 premiers pour la lisibilité
    valeurs_finales = donnees_mensuelles[nom_actif]['valeurs'][:, -1]
    sns.kdeplot(valeurs_finales, label=nom_actif, fill=True, alpha=0.3, color=resultats_df.loc[nom_actif, 'couleur'])

# Ajouter des lignes verticales pour les points de référence
ax_dist.axvline(x=montant_pret, color='r', linestyle='--', label='Montant Initial du Prêt')
ax_dist.axvline(x=montant_pret + cout_total_interet, color='orange', linestyle='--', 
               label='Montant Prêt + Intérêts')

ax_dist.set_title('Distribution des Valeurs Finales par Classe d\'Actif', fontsize=16)
ax_dist.set_xlabel('Valeur Finale (€)', fontsize=14)
ax_dist.set_ylabel('Densité', fontsize=14)
ax_dist.legend()
ax_dist.grid(True, alpha=0.3)

# 5. Évolution temporelle des valeurs moyennes
ax_time = plt.subplot(gs[3, :])
mois = np.arange(total_mois + 1)
for nom_actif in resultats_df.index[:7]:  # Limiter aux 7 premiers pour la lisibilité
    ax_time.plot(mois, donnees_mensuelles[nom_actif]['moyenne'], 
                label=nom_actif, linewidth=2, color=resultats_df.loc[nom_actif, 'couleur'])

# Calculer et tracer le solde du prêt
solde_pret = np.zeros(total_mois + 1)
solde_pret[0] = montant_pret
for t in range(1, mois_differe + 1):
    solde_pret[t] = solde_pret[t-1] * (1 + taux_interet_mensuel)

for t in range(mois_differe + 1, total_mois + 1):
    paiement_interet = solde_pret[t-1] * taux_interet_mensuel
    paiement_principal = paiement_mensuel - paiement_interet
    solde_pret[t] = max(0, solde_pret[t-1] - paiement_principal)

ax_time.plot(mois, solde_pret, 'r--', label='Solde du Prêt', linewidth=2)
ax_time.axvline(x=mois_differe, color='k', linestyle=':', label='Fin du Différé')

# Tracer la ligne du capital initial + intérêts
capital_plus_interets = np.ones(total_mois + 1) * (montant_pret + cout_total_interet)
ax_time.plot(mois, capital_plus_interets, 'k--', label='Capital + Intérêts', linewidth=1.5)

ax_time.set_title('Évolution Temporelle de la Valeur Moyenne du Portefeuille', fontsize=16)
ax_time.set_xlabel('Mois', fontsize=14)
ax_time.set_ylabel('Valeur (€)', fontsize=14)
ax_time.legend()
ax_time.grid(True, alpha=0.3)

# 6. Graphique de risque/rendement
ax_risk = plt.subplot(gs[4, 0])
for nom_actif in resultats_df.index:
    ax_risk.scatter(
        resultats_df.loc[nom_actif, 'volatilite_annuelle'],
        resultats_df.loc[nom_actif, 'rendement_annuel'],
        s=200,
        color=resultats_df.loc[nom_actif, 'couleur'],
        label=nom_actif,
        alpha=0.7
    )
    ax_risk.annotate(
        nom_actif,
        (resultats_df.loc[nom_actif, 'volatilite_annuelle'] + 0.2, 
         resultats_df.loc[nom_actif, 'rendement_annuel']),
        fontsize=9
    )

ax_risk.set_title('Profil Risque/Rendement des Classes d\'Actifs', fontsize=16)
ax_risk.set_xlabel('Volatilité Annuelle (%)', fontsize=14)
ax_risk.set_ylabel('Rendement Annuel (%)', fontsize=14)
ax_risk.grid(True, alpha=0.3)

# 7. Value at Risk (VaR) par classe d'actif
ax_var = plt.subplot(gs[4, 1])
var_data = resultats_df[['valeur_risque_5', 'valeur_risque_1']].copy()
var_data['perte_var_5'] = (var_data['valeur_risque_5'] - montant_pret) / montant_pret * 100
var_data['perte_var_1'] = (var_data['valeur_risque_1'] - montant_pret) / montant_pret * 100
var_data = var_data.sort_values('perte_var_5')

# Créer un graphique à barres pour la VaR
bars1 = ax_var.bar(var_data.index, var_data['perte_var_5'], width=0.4, 
                  label='VaR 5%', color='orange', alpha=0.7)
bars2 = ax_var.bar([x + 0.4 for x in range(len(var_data))], var_data['perte_var_1'], 
                  width=0.4, label='VaR 1%', color='red', alpha=0.7)

ax_var.set_title('Value at Risk (VaR) par Classe d\'Actif', fontsize=16)
ax_var.set_ylabel('Perte Potentielle (%)', fontsize=14)
ax_var.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax_var.grid(True, alpha=0.3)
ax_var.legend()

# Ajouter les valeurs sur les barres
for i, v in enumerate(var_data['perte_var_5']):
    ax_var.text(i, v - 2, f"{v:.1f}%", ha='center', fontsize=9, color='white')
for i, v in enumerate(var_data['perte_var_1']):
    ax_var.text(i + 0.4, v - 2, f"{v:.1f}%", ha='center', fontsize=9, color='white')

plt.setp(ax_var.get_xticklabels(), rotation=45, ha='right')

# 8. Tableau explicatif des métriques de risque
ax_explanation = plt.subplot(gs[5, :])
ax_explanation.axis('off')

# Texte explicatif
explanation_text = (
    "GUIDE DES MÉTRIQUES DE RISQUE ET D'INVESTISSEMENT\n\n"
    "• Ratio de Sharpe : Mesure du rendement excédentaire par unité de risque. Plus le ratio est élevé, meilleur est le rendement ajusté au risque.\n"
    "  Un ratio > 1 est généralement considéré comme bon, > 2 comme très bon.\n\n"
    "• Taux de Réussite : Pourcentage des simulations où la valeur finale du portefeuille dépasse le montant total du prêt + intérêts.\n"
    "  Indique la probabilité de réaliser un gain net positif.\n\n"
    "• Drawdown Maximum : La plus grande baisse entre un pic et un creux durant la période. Mesure l'ampleur des baisses temporaires.\n"
    "  Important pour évaluer la résistance psychologique nécessaire face aux fluctuations.\n\n"
    "• Value at Risk (VaR) : Perte potentielle maximale avec un niveau de confiance donné.\n"
    "  VaR 5% = perte maximale dans 95% des scénarios, VaR 1% = perte maximale dans 99% des scénarios.\n\n"
    "• Ratio Gain/Coût : Combien d'euros de gain net pour chaque euro d'intérêt payé.\n"
    "  Un ratio > 1 signifie que l'effet de levier est positif et que l'investissement génère plus que le coût du prêt.\n\n"
    "CONSIDÉRATIONS IMPORTANTES POUR LA PRISE DE DÉCISION :\n"
    "1. Votre tolérance au risque - Êtes-vous prêt à supporter des fluctuations importantes pour un rendement potentiellement plus élevé?\n"
    "2. Votre horizon d'investissement - Pouvez-vous maintenir votre stratégie pendant toute la durée du prêt (10 ans)?\n"
    "3. Votre capacité à ne pas paniquer lors des baisses temporaires du marché.\n"
    "4. Votre situation financière globale - Avez-vous d'autres sources de revenus ou d'épargne en cas de besoin?\n\n"
    f"AVANTAGE UNIQUE : Avec un prêt étudiant à seulement {taux_interet_annuel*100:.2f}%, l'effet de levier est très favorable pour presque toutes les classes d'actifs."
)

ax_explanation.text(0.5, 0.5, explanation_text, ha='center', va='center', fontsize=14,
                  bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=1'))

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3)
plt.savefig(f"{repertoire_sortie}/tableau_bord_enrichi_etape1.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{repertoire_sortie}/tableau_bord_enrichi_etape1.pdf", bbox_inches='tight')
print(f"\nTableau de bord enrichi (étape 1) enregistré dans {repertoire_sortie}/tableau_bord_enrichi_etape1.png et .pdf")

# Afficher le chemin vers le fichier généré
print(f"\nVous pouvez consulter le tableau de bord enrichi ici: {os.path.abspath(f'{repertoire_sortie}/tableau_bord_enrichi_etape1.png')}")

# Créer un tableau récapitulatif des profils d'actifs
print("\nRécapitulatif des classes d'actifs :")
print("------------------------------------")
recap_df = resultats_df[['nom', 'description', 'profil_risque', 'horizon_recommande', 
                         'rendement_annuel', 'volatilite_annuelle', 'ratio_sharpe', 
                         'gain_net_moyen', 'ratio_gain_cout', 'taux_reussite']]

# Formater les colonnes numériques
recap_df['rendement_annuel'] = recap_df['rendement_annuel'].apply(lambda x: f"{x:.1f}%")
recap_df['volatilite_annuelle'] = recap_df['volatilite_annuelle'].apply(lambda x: f"{x:.1f}%")
recap_df['ratio_sharpe'] = recap_df['ratio_sharpe'].apply(lambda x: f"{x:.2f}")
recap_df['gain_net_moyen'] = recap_df['gain_net_moyen'].apply(lambda x: f"{x:,.0f} €")
recap_df['ratio_gain_cout'] = recap_df['ratio_gain_cout'].apply(lambda x: f"{x:.2f}")
recap_df['taux_reussite'] = recap_df['taux_reussite'].apply(lambda x: f"{x:.1f}%")

# Enregistrer le récapitulatif au format CSV
recap_df.to_csv(f"{repertoire_sortie}/recap_actifs_enrichis.csv", index=True)
print(f"Récapitulatif des actifs enregistré dans {repertoire_sortie}/recap_actifs_enrichis.csv")
