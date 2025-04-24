import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# Définir le style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Créer le répertoire de sortie s'il n'existe pas
repertoire_sortie = "../output"
if not os.path.exists(repertoire_sortie):
    os.makedirs(repertoire_sortie)

# Paramètres de la simulation
np.random.seed(42)  # Pour la reproductibilité
montant_pret = 200_000
taux_interet_annuel = 0.0099
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

# Définir les actifs avec leurs caractéristiques
actifs = {
    'ETF d\'Actions': {
        'rendement_annuel': 0.07,
        'volatilite_annuelle': 0.15,
        'rendement_mensuel': 0.07/12,
        'volatilite_mensuelle': 0.15/np.sqrt(12),
        'couleur': '#1f77b4'
    },
    'Obligations d\'État': {
        'rendement_annuel': 0.025,
        'volatilite_annuelle': 0.05,
        'rendement_mensuel': 0.025/12,
        'volatilite_mensuelle': 0.05/np.sqrt(12),
        'couleur': '#ff7f0e'
    },
    'Immobilier': {
        'rendement_annuel': 0.07,
        'volatilite_annuelle': 0.10,
        'rendement_mensuel': 0.07/12,
        'volatilite_mensuelle': 0.10/np.sqrt(12),
        'couleur': '#2ca02c'
    },
    'Portefeuille Mixte (60/40)': {
        'rendement_annuel': 0.07*0.6 + 0.025*0.4,
        'volatilite_annuelle': 0.15*0.6 + 0.05*0.4,
        'rendement_mensuel': (0.07*0.6 + 0.025*0.4)/12,
        'volatilite_mensuelle': (0.15*0.6 + 0.05*0.4)/np.sqrt(12),
        'couleur': '#d62728'
    },
    'Portefeuille Mixte (80/20)': {
        'rendement_annuel': 0.07*0.8 + 0.025*0.2,
        'volatilite_annuelle': 0.15*0.8 + 0.05*0.2,
        'rendement_mensuel': (0.07*0.8 + 0.025*0.2)/12,
        'volatilite_mensuelle': (0.15*0.8 + 0.05*0.2)/np.sqrt(12),
        'couleur': '#9467bd'
    },
    'Portefeuille Mixte (40/60)': {
        'rendement_annuel': 0.07*0.4 + 0.025*0.6,
        'volatilite_annuelle': 0.15*0.4 + 0.05*0.6,
        'rendement_mensuel': (0.07*0.4 + 0.025*0.6)/12,
        'volatilite_mensuelle': (0.15*0.4 + 0.05*0.6)/np.sqrt(12),
        'couleur': '#8c564b'
    },
    'Portefeuille Équilibré': {
        'rendement_annuel': 0.07*0.4 + 0.025*0.3 + 0.07*0.3,
        'volatilite_annuelle': 0.15*0.4 + 0.05*0.3 + 0.10*0.3,
        'rendement_mensuel': (0.07*0.4 + 0.025*0.3 + 0.07*0.3)/12,
        'volatilite_mensuelle': (0.15*0.4 + 0.05*0.3 + 0.10*0.3)/np.sqrt(12),
        'couleur': '#e377c2'
    }
}

# Fonction pour simuler l'évolution du portefeuille
def simuler_portefeuille(actif, n_simulations, total_mois, montant_initial):
    """Simuler l'évolution d'un portefeuille avec des rendements aléatoires"""
    rendements = actif['rendement_mensuel']
    volatilite = actif['volatilite_mensuelle']
    
    # Initialiser les tableaux pour les simulations
    valeurs = np.zeros((n_simulations, total_mois + 1))
    valeurs[:, 0] = montant_initial
    
    # Simuler les rendements mensuels
    for t in range(total_mois):
        # Générer des rendements aléatoires pour ce mois
        r = np.random.normal(rendements, volatilite, n_simulations)
        
        # Appliquer les rendements au portefeuille
        valeurs[:, t+1] = valeurs[:, t] * (1 + r)
    
    return valeurs

# Simuler tous les actifs
resultats = {}
donnees_mensuelles = {}

for nom_actif, actif in actifs.items():
    print(f"Simulation de {nom_actif}...")
    valeurs = simuler_portefeuille(actif, n_simulations, total_mois, montant_pret)
    
    # Calculer les statistiques sur les valeurs finales
    valeurs_finales = valeurs[:, -1]
    
    # Calculer les résultats pour cet actif
    resultats[nom_actif] = {
        'rendement_annuel': actif['rendement_annuel'] * 100,
        'volatilite_annuelle': actif['volatilite_annuelle'] * 100,
        'moyenne_valeur_finale': np.mean(valeurs_finales),
        'mediane_valeur_finale': np.median(valeurs_finales),
        'ecart_type_valeur_finale': np.std(valeurs_finales),
        'gain_brut_moyen': np.mean(valeurs_finales) - montant_pret,
        'gain_net_moyen': np.mean(valeurs_finales) - montant_pret - cout_total_interet,
        'rendement_net_annualise': ((np.mean(valeurs_finales) / montant_pret) ** (1/10) - 1) * 100,
        'ratio_gain_cout': (np.mean(valeurs_finales) - montant_pret - cout_total_interet) / cout_total_interet,
        'valeur_risque_5': np.percentile(valeurs_finales, 5),
        'percentile_25': np.percentile(valeurs_finales, 25),
        'percentile_75': np.percentile(valeurs_finales, 75),
        'percentile_95': np.percentile(valeurs_finales, 95),
        'min': np.min(valeurs_finales),
        'max': np.max(valeurs_finales),
        'taux_reussite': np.mean(valeurs_finales > (montant_pret + cout_total_interet)) * 100,
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
resultats_df = pd.DataFrame(resultats).T
resultats_df = resultats_df.sort_values('gain_net_moyen', ascending=False)

# Créer un tableau de bord visuel complet
plt.figure(figsize=(20, 24))
gs = GridSpec(5, 2, figure=plt.gcf(), height_ratios=[0.8, 1, 1, 1, 1])

# Titre et paramètres
plt.suptitle("TABLEAU DE BORD - ANALYSE DES OPTIONS D'INVESTISSEMENT (VERSION FINALE)", fontsize=24, y=0.995)
texte_param = (
    f"Paramètres : Prêt {montant_pret:,.0f} € à {taux_interet_annuel*100:.2f}% | "
    f"Différé : {mois_differe} mois (1,5 ans) | "
    f"Remboursement : {mois_remboursement} mois (8,5 ans) | "
    f"Total : 10 ans | "
    f"Coût total des intérêts : {cout_total_interet:,.0f} €"
)
plt.figtext(0.5, 0.975, texte_param, ha="center", fontsize=14)

# 1. Tableau des métriques clés avec focus sur le gain réel
ax_metrics = plt.subplot(gs[0, :])
ax_metrics.axis('off')

# Préparer les données pour le tableau
metrics_data = []
for nom_actif in resultats_df.index:
    metrics_data.append([
        nom_actif,
        f"{resultats_df.loc[nom_actif, 'rendement_annuel']:.1f}%",
        f"{resultats_df.loc[nom_actif, 'taux_reussite']:.1f}%",
        f"{resultats_df.loc[nom_actif, 'gain_net_moyen']:,.0f} €",
        f"{resultats_df.loc[nom_actif, 'rendement_net_annualise']:.2f}%",
        f"{resultats_df.loc[nom_actif, 'ratio_gain_cout']:.2f}",
        f"{resultats_df.loc[nom_actif, 'valeur_risque_5']:,.0f} €"
    ])

metric_labels = [
    'Classe d\'Actif', 'Rend. Annuel', 'Taux de Réussite', 'Gain Net Moyen', 
    'Rend. Net Annualisé', 'Ratio Gain/Coût', 'VaR (5%)'
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
        
        # Colorer les cellules selon les valeurs (vert = bon, rouge = mauvais)
        if j == 2:  # Taux de réussite
            val = resultats_df.iloc[i]['taux_reussite']
            color = (1-val/100, val/100, 0.3)
            cell.set_facecolor(color)
        elif j == 4:  # Rendement net annualisé
            val = resultats_df.iloc[i]['rendement_net_annualise']
            min_val = resultats_df['rendement_net_annualise'].min()
            max_val = resultats_df['rendement_net_annualise'].max()
            norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            color = (1-norm_val, norm_val, 0.3)
            cell.set_facecolor(color)

# Mettre en forme l'en-tête
for j in range(len(metric_labels)):
    cell = metrics_table[(0, j)]
    cell.set_facecolor('#4c72b0')
    cell.set_text_props(weight='bold', color='white')

# 2. Graphique comparatif des gains nets
ax_gains = plt.subplot(gs[1, 0])
gains_nets = resultats_df['gain_net_moyen']
bars = ax_gains.bar(gains_nets.index, gains_nets.values)

# Colorer les barres en fonction de la valeur
for i, bar in enumerate(bars):
    val = gains_nets.iloc[i]
    if val > 0:
        bar.set_color('green')
    else:
        bar.set_color('red')

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

# 3. Graphique comparatif des rendements nets annualisés
ax_returns = plt.subplot(gs[1, 1])
rendements_nets = resultats_df['rendement_net_annualise']
bars = ax_returns.bar(rendements_nets.index, rendements_nets.values)

# Colorer les barres en fonction de la valeur
for i, bar in enumerate(bars):
    val = rendements_nets.iloc[i]
    if val > 0:
        bar.set_color('green')
    else:
        bar.set_color('red')

ax_returns.set_title('Rendement Net Annualisé par Classe d\'Actif', fontsize=16)
ax_returns.set_ylabel('Rendement Net Annualisé (%)', fontsize=14)
ax_returns.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax_returns.grid(True, alpha=0.3)

# Ajouter les valeurs sur les barres
for i, v in enumerate(rendements_nets):
    ax_returns.text(i, v + (0.5 if v > 0 else -0.5), 
                   f"{v:.2f}%", 
                   ha='center', fontsize=10)

plt.setp(ax_returns.get_xticklabels(), rotation=45, ha='right')

# 4. Distribution des valeurs finales
ax_dist = plt.subplot(gs[2, :])
for nom_actif in resultats_df.index:
    valeurs_finales = donnees_mensuelles[nom_actif]['valeurs'][:, -1]
    sns.kdeplot(valeurs_finales, label=nom_actif, fill=True, alpha=0.3, color=actifs[nom_actif]['couleur'])

# Ajouter des lignes verticales pour les points de référence
ax_dist.axvline(x=montant_pret, color='r', linestyle='--', label='Montant Initial du Prêt')
ax_dist.axvline(x=montant_pret + cout_total_interet, color='orange', linestyle='--', 
               label='Montant Prêt + Intérêts')

ax_dist.set_title('Distribution des Valeurs Finales par Classe d\'Actif', fontsize=16)
ax_dist.set_xlabel('Valeur Finale (€)', fontsize=14)
ax_dist.set_ylabel('Densité', fontsize=14)
ax_dist.legend()
ax_dist.grid(True, alpha=0.3)

# Ajouter une annotation pour expliquer le gain net
ax_dist.annotate('Gain Net = Valeur Finale - (Prêt + Intérêts)', 
                xy=(montant_pret + cout_total_interet, 0.000005),
                xytext=(montant_pret + cout_total_interet + 50000, 0.000008),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                ha='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))

# 5. Évolution temporelle des valeurs moyennes
ax_time = plt.subplot(gs[3, :])
mois = np.arange(total_mois + 1)
for nom_actif in resultats_df.index:
    ax_time.plot(mois, donnees_mensuelles[nom_actif]['moyenne'], 
                label=nom_actif, linewidth=2, color=actifs[nom_actif]['couleur'])

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

# 6. Tableau explicatif des coûts et bénéfices
ax_explanation = plt.subplot(gs[4, :])
ax_explanation.axis('off')

# Texte explicatif
explanation_text = (
    "EXPLICATION DES MÉTRIQUES ET RÉSULTATS\n\n"
    "• Rendement Annuel : Le taux de rendement annuel moyen attendu pour chaque classe d'actif\n\n"
    "• Taux de Réussite : Pourcentage des simulations où la valeur finale du portefeuille dépasse le montant total du prêt + intérêts\n\n"
    "• Gain Net Moyen : Valeur finale moyenne - Montant du prêt - Intérêts payés\n"
    f"  (Intérêts totaux payés sur la durée du prêt : {cout_total_interet:,.0f} €)\n\n"
    "• Rendement Net Annualisé : Taux de rendement annuel équivalent basé sur la valeur finale du portefeuille\n\n"
    "• Ratio Gain/Coût : Combien d'euros de gain net pour chaque euro d'intérêt payé (> 1 signifie que l'effet de levier est positif)\n\n"
    "• VaR (5%) : Value at Risk - La valeur du portefeuille dans le pire des 5% des cas\n\n"
    "VÉRIFICATION DE LA COHÉRENCE :\n"
    f"Valeur théorique pour ETF Actions après 10 ans (7%) : 200 000 € × (1,07)¹⁰ = 393 430 €\n"
    f"Gain net théorique ETF Actions : 393 430 € - 200 000 € - 11 735 € = 181 695 €\n\n"
    f"Valeur théorique Obligations après 10 ans (2,5%) : 200 000 € × (1,025)¹⁰ = 256 017 €\n"
    f"Gain net théorique Obligations : 256 017 € - 200 000 € - 11 735 € = 44 282 €\n\n"
    "Les résultats des simulations devraient désormais être cohérents avec ces calculs théoriques."
)

ax_explanation.text(0.5, 0.5, explanation_text, ha='center', va='center', fontsize=14,
                  bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=1'))

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3)
plt.savefig(f"{repertoire_sortie}/tableau_bord_final.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{repertoire_sortie}/tableau_bord_final.pdf", bbox_inches='tight')
print(f"\nTableau de bord final enregistré dans {repertoire_sortie}/tableau_bord_final.png et .pdf")

# Afficher le chemin vers le fichier généré
print(f"\nVous pouvez consulter le tableau de bord final ici: {os.path.abspath(f'{repertoire_sortie}/tableau_bord_final.png')}")

# Pour vérifier la cohérence, afficher les valeurs calculées par la simulation
print("\nVérification de la cohérence des résultats :")
print("---------------------------------------------")
print("ETF d'Actions (7% annuel) :")
print(f"Valeur théorique (calcul direct) : 200 000 € × (1,07)¹⁰ = {200000 * (1.07)**10:,.2f} €")
print(f"Valeur finale moyenne (simulation) : {resultats['ETF d\'Actions']['moyenne_valeur_finale']:,.2f} €")
print(f"Gain net théorique : {200000 * (1.07)**10 - 200000 - cout_total_interet:,.2f} €")
print(f"Gain net moyen (simulation) : {resultats['ETF d\'Actions']['gain_net_moyen']:,.2f} €")

print("\nObligations d'État (2,5% annuel) :")
print(f"Valeur théorique (calcul direct) : 200 000 € × (1,025)¹⁰ = {200000 * (1.025)**10:,.2f} €")
print(f"Valeur finale moyenne (simulation) : {resultats['Obligations d\'État']['moyenne_valeur_finale']:,.2f} €")
print(f"Gain net théorique : {200000 * (1.025)**10 - 200000 - cout_total_interet:,.2f} €")
print(f"Gain net moyen (simulation) : {resultats['Obligations d\'État']['gain_net_moyen']:,.2f} €")
