import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from investment_sim import SimulationInvestissement
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick

# Définir le style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Créer le répertoire de sortie s'il n'existe pas
repertoire_sortie = "../output"
if not os.path.exists(repertoire_sortie):
    os.makedirs(repertoire_sortie)

# Initialiser la simulation
sim = SimulationInvestissement(mois_remboursement=102, n_simulations=10000)  # 8,5 ans, 10000 simulations
print(f"Montant initial du prêt : {sim.montant_pret:,.2f} €")
print(f"Montant différé après {sim.mois_differe} mois : {sim.montant_differe:,.2f} €")
print(f"Paiement mensuel pendant la période de remboursement : {sim.paiement_mensuel:,.2f} €")

# Exécuter la simulation
liste_resultats = []
donnees_portefeuille = {}
donnees_impots = {}
donnees_frais_transaction = {}
donnees_mensuelles = {}

# Exécuter les simulations pour chaque classe d'actif
for cle_actif, actif in sim.actifs.items():
    print(f"\nSimulation de {actif.nom}...")
    valeurs_portefeuille, impots_payes, frais_transaction = sim.simuler_portefeuille(actif)
    resultat = sim.analyser_resultats(valeurs_portefeuille, impots_payes, frais_transaction, actif.nom)
    liste_resultats.append(resultat)
    
    # Stocker les données pour une utilisation ultérieure
    donnees_portefeuille[actif.nom] = valeurs_portefeuille
    donnees_impots[actif.nom] = impots_payes
    donnees_frais_transaction[actif.nom] = frais_transaction
    
    # Calculer et stocker les données mensuelles pour chaque actif
    donnees_mensuelles[actif.nom] = {
        'moyenne': np.nanmean(valeurs_portefeuille, axis=0),
        'mediane': np.nanmedian(valeurs_portefeuille, axis=0),
        'ecart_type': np.nanstd(valeurs_portefeuille, axis=0),
        'percentile_25': np.nanpercentile(valeurs_portefeuille, 25, axis=0),
        'percentile_75': np.nanpercentile(valeurs_portefeuille, 75, axis=0),
        'percentile_5': np.nanpercentile(valeurs_portefeuille, 5, axis=0),
        'percentile_95': np.nanpercentile(valeurs_portefeuille, 95, axis=0)
    }

# Créer un DataFrame de résultats
resultats_df = pd.DataFrame(liste_resultats)
resultats_df.set_index('nom_actif', inplace=True)

# Calcul des métriques de risque supplémentaires
for nom_actif in resultats_df.index:
    valeurs_finales = donnees_portefeuille[nom_actif][:, -1]
    valeurs_valides = valeurs_finales[~np.isnan(valeurs_finales)]
    
    # Ratio de Sharpe (approximation simplifiée)
    taux_sans_risque = 0.01  # 1% annuel
    rendement_moyen = resultats_df.loc[nom_actif, 'rendement_annualise_moyen'] / 100
    volatilite = np.nanstd(valeurs_valides) / np.nanmean(valeurs_valides)
    ratio_sharpe = (rendement_moyen - taux_sans_risque) / volatilite if volatilite != 0 else 0
    resultats_df.loc[nom_actif, 'ratio_sharpe'] = ratio_sharpe
    
    # Drawdown maximal (approximation pour les valeurs finales)
    resultats_df.loc[nom_actif, 'drawdown_max'] = (np.nanmax(valeurs_valides) - np.nanmin(valeurs_valides)) / np.nanmax(valeurs_valides) * 100
    
    # Probabilité de perte
    resultats_df.loc[nom_actif, 'probabilite_perte'] = np.sum(valeurs_valides < sim.montant_differe) / len(valeurs_valides) * 100
    
    # Perte maximale
    resultats_df.loc[nom_actif, 'perte_maximale'] = (np.nanmin(valeurs_valides) - sim.montant_differe) / sim.montant_differe * 100
    
    # Gain maximal
    resultats_df.loc[nom_actif, 'gain_maximal'] = (np.nanmax(valeurs_valides) - sim.montant_differe) / sim.montant_differe * 100

# Calculer un score pour chaque classe d'actif basé sur plusieurs facteurs
scores = pd.DataFrame(index=resultats_df.index)

# Normaliser les valeurs pour le score
scores['score_rendement'] = (resultats_df['rendement_annualise_moyen'] - resultats_df['rendement_annualise_moyen'].min()) / (resultats_df['rendement_annualise_moyen'].max() - resultats_df['rendement_annualise_moyen'].min()) if resultats_df['rendement_annualise_moyen'].max() != resultats_df['rendement_annualise_moyen'].min() else 0
scores['score_risque'] = resultats_df['taux_reussite'] / 100
scores['score_valeur'] = (resultats_df['moyenne_valeur_finale'] - resultats_df['moyenne_valeur_finale'].min()) / (resultats_df['moyenne_valeur_finale'].max() - resultats_df['moyenne_valeur_finale'].min()) if resultats_df['moyenne_valeur_finale'].max() != resultats_df['moyenne_valeur_finale'].min() else 0
scores['score_sharpe'] = (resultats_df['ratio_sharpe'] - resultats_df['ratio_sharpe'].min()) / (resultats_df['ratio_sharpe'].max() - resultats_df['ratio_sharpe'].min()) if resultats_df['ratio_sharpe'].max() != resultats_df['ratio_sharpe'].min() else 0

# Calculer les coûts en pourcentage
couts_df = pd.DataFrame(index=resultats_df.index)
for nom_actif in resultats_df.index:
    impots_totaux = resultats_df.loc[nom_actif, 'impots_moyens_payes']
    frais_totaux = resultats_df.loc[nom_actif, 'frais_transaction_moyens']
    couts_totaux = impots_totaux + frais_totaux
    valeur_finale = resultats_df.loc[nom_actif, 'moyenne_valeur_finale']
    couts_df.loc[nom_actif, 'pourcentage'] = (couts_totaux / valeur_finale) * 100 if valeur_finale > 0 else 0

scores['score_couts'] = 1 - (couts_df['pourcentage'] - couts_df['pourcentage'].min()) / (couts_df['pourcentage'].max() - couts_df['pourcentage'].min()) if couts_df['pourcentage'].max() != couts_df['pourcentage'].min() else 0

# Pondération des scores (ajustez selon vos préférences)
poids = {
    'score_rendement': 0.3,
    'score_risque': 0.3,
    'score_valeur': 0.2,
    'score_sharpe': 0.1,
    'score_couts': 0.1
}

# Calculer le score total
scores['score_total'] = sum(scores[col] * poids[col] for col in poids.keys())

# Trier par score total
scores_tries = scores.sort_values('score_total', ascending=False)

# Créer un tableau de bord visuel complet
plt.figure(figsize=(20, 24))
gs = GridSpec(6, 2, figure=plt.gcf(), height_ratios=[0.8, 1, 1, 1, 1, 1])

# Titre et paramètres
plt.suptitle("TABLEAU DE BORD COMPLET - ANALYSE DES OPTIONS D'INVESTISSEMENT", fontsize=24, y=0.995)
texte_param = (
    f"Paramètres : Prêt {sim.montant_pret:,.0f} € à {sim.taux_interet_annuel*100:.2f}% | "
    f"Différé : {sim.mois_differe} mois (1,5 ans) | "
    f"Remboursement : {sim.mois_remboursement} mois (8,5 ans) | "
    f"Total : 10 ans | "
    f"Simulations : {sim.n_simulations}"
)
plt.figtext(0.5, 0.975, texte_param, ha="center", fontsize=14)

# 1. Tableau récapitulatif des scores et recommandations
ax_scores = plt.subplot(gs[0, :])
ax_scores.axis('off')

# Créer un mini-tableau pour les scores
scores_data = []
for nom_actif in scores_tries.index:
    scores_data.append([
        nom_actif,
        f"{scores_tries.loc[nom_actif, 'score_rendement']:.2f}",
        f"{scores_tries.loc[nom_actif, 'score_risque']:.2f}",
        f"{scores_tries.loc[nom_actif, 'score_valeur']:.2f}",
        f"{scores_tries.loc[nom_actif, 'score_sharpe']:.2f}",
        f"{scores_tries.loc[nom_actif, 'score_couts']:.2f}",
        f"{scores_tries.loc[nom_actif, 'score_total']:.2f}"
    ])

col_labels = ['Classe d\'Actif', 'Rendement (30%)', 'Risque (30%)', 'Valeur (20%)', 'Sharpe (10%)', 'Coûts (10%)', 'SCORE TOTAL']
score_table = ax_scores.table(
    cellText=scores_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
    colWidths=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
score_table.auto_set_font_size(False)
score_table.set_fontsize(12)
score_table.scale(1, 1.5)

# Mettre en forme le tableau
for i, nom_actif in enumerate(scores_tries.index):
    # Colorer la ligne en fonction du score total (du vert au rouge)
    score = scores_tries.loc[nom_actif, 'score_total']
    color_intensity = min(1.0, max(0.0, score))
    bg_color = (1-color_intensity, color_intensity, 0.3)
    
    for j in range(7):
        cell = score_table[(i+1, j)]
        cell.set_facecolor(bg_color)
        
        # Mettre en gras la meilleure option
        if i == 0:
            cell.set_text_props(weight='bold')

# Mettre en forme l'en-tête
for j in range(7):
    cell = score_table[(0, j)]
    cell.set_facecolor('#4c72b0')
    cell.set_text_props(weight='bold', color='white')

# 2. Graphique Risque vs Rendement
ax_risk_return = plt.subplot(gs[1, 0])
x = resultats_df['rendement_annualise_moyen']
y = 100 - resultats_df['taux_reussite']  # Convertir le taux de réussite en risque d'échec
taille = resultats_df['moyenne_valeur_finale'] / 50000  # Taille proportionnelle à la valeur finale

scatter = ax_risk_return.scatter(x, y, s=taille, alpha=0.7, c=scores_tries['score_total'], cmap='RdYlGn')
for i, nom in enumerate(resultats_df.index):
    ax_risk_return.annotate(nom, (x.iloc[i], y.iloc[i]), fontsize=12)

ax_risk_return.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Seuil de risque 10%')
ax_risk_return.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='Seuil de rendement 0%')

ax_risk_return.set_title('Analyse Risque vs Rendement', fontsize=16)
ax_risk_return.set_xlabel('Rendement Annualisé Moyen (%)', fontsize=14)
ax_risk_return.set_ylabel('Risque d\'Échec (%)', fontsize=14)
ax_risk_return.grid(True, alpha=0.3)
ax_risk_return.legend()

# Ajouter une légende pour la taille des bulles
handles, labels = ax_risk_return.get_legend_handles_labels()
legend1 = ax_risk_return.legend(handles, labels, loc='upper right')
ax_risk_return.add_artist(legend1)

# Créer une légende personnalisée pour la taille des bulles
sizes = [50000, 200000, 500000, 1000000]
labels_size = [f"{s/1000:.0f}k€" for s in sizes]
scatter_size = [s/50000 for s in sizes]

handles = [plt.scatter([], [], s=s, color='gray', alpha=0.7) for s in scatter_size]
legend2 = ax_risk_return.legend(handles, labels_size, loc='upper left', title="Valeur Finale Moyenne")
ax_risk_return.add_artist(legend2)

# 3. Distribution des valeurs finales
ax_dist = plt.subplot(gs[1, 1])
for i, nom_actif in enumerate(donnees_portefeuille.keys()):
    valeurs_finales = donnees_portefeuille[nom_actif][:, -1]
    valeurs_valides = valeurs_finales[~np.isnan(valeurs_finales)]
    if len(valeurs_valides) > 0:
        sns.kdeplot(valeurs_valides, label=nom_actif, fill=True, alpha=0.3)

ax_dist.axvline(x=sim.montant_differe, color='r', linestyle='--', label='Montant Initial Différé')
ax_dist.set_title('Distribution des Valeurs Finales par Classe d\'Actif', fontsize=16)
ax_dist.set_xlabel('Valeur Finale (€)', fontsize=14)
ax_dist.set_ylabel('Densité', fontsize=14)
ax_dist.legend()
ax_dist.grid(True, alpha=0.3)

# Limiter l'axe x pour une meilleure lisibilité (exclure les valeurs extrêmes)
ax_dist.set_xlim(0, 500000)
ax_dist.text(0.95, 0.95, "Note: Certaines valeurs extrêmes\nne sont pas affichées", 
             transform=ax_dist.transAxes, fontsize=10, ha='right',
             bbox=dict(facecolor='white', alpha=0.8))

# 4. Tableau des métriques clés
ax_metrics = plt.subplot(gs[2, :])
ax_metrics.axis('off')

# Préparer les données pour le tableau
metrics_data = []
for nom_actif in resultats_df.index:
    metrics_data.append([
        nom_actif,
        f"{resultats_df.loc[nom_actif, 'taux_reussite']:.1f}%",
        f"{resultats_df.loc[nom_actif, 'rendement_annualise_moyen']:.2f}%",
        f"{resultats_df.loc[nom_actif, 'moyenne_valeur_finale']:,.0f} €",
        f"{resultats_df.loc[nom_actif, 'mediane_valeur_finale']:,.0f} €",
        f"{resultats_df.loc[nom_actif, 'var_95']:,.0f} €",
        f"{resultats_df.loc[nom_actif, 'ratio_sharpe']:.4f}",
        f"{resultats_df.loc[nom_actif, 'probabilite_perte']:.1f}%",
        f"{resultats_df.loc[nom_actif, 'perte_maximale']:.1f}%",
        f"{resultats_df.loc[nom_actif, 'gain_maximal']:.1f}%",
        f"{resultats_df.loc[nom_actif, 'impots_moyens_payes']:,.0f} €",
        f"{couts_df.loc[nom_actif, 'pourcentage']:.1f}%"
    ])

metric_labels = [
    'Classe d\'Actif', 'Taux de Réussite', 'Rendement Annualisé', 'Valeur Finale Moyenne', 
    'Valeur Finale Médiane', 'VaR 5%', 'Ratio de Sharpe', 'Probabilité de Perte', 
    'Perte Maximale', 'Gain Maximal', 'Impôts Moyens', 'Coûts (%)'
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
        if j == 1:  # Taux de réussite
            val = resultats_df.iloc[i]['taux_reussite']
            color = (1-val/100, val/100, 0.3)
            cell.set_facecolor(color)
        elif j == 2:  # Rendement annualisé
            val = resultats_df.iloc[i]['rendement_annualise_moyen']
            norm_val = (val - resultats_df['rendement_annualise_moyen'].min()) / (resultats_df['rendement_annualise_moyen'].max() - resultats_df['rendement_annualise_moyen'].min())
            color = (1-norm_val, norm_val, 0.3)
            cell.set_facecolor(color)
        elif j == 6:  # Ratio de Sharpe
            val = resultats_df.iloc[i]['ratio_sharpe']
            norm_val = (val - resultats_df['ratio_sharpe'].min()) / (resultats_df['ratio_sharpe'].max() - resultats_df['ratio_sharpe'].min())
            color = (1-norm_val, norm_val, 0.3)
            cell.set_facecolor(color)
        elif j == 11:  # Coûts
            val = couts_df.iloc[i]['pourcentage']
            max_val = couts_df['pourcentage'].max()
            norm_val = 1 - (val / max_val if max_val > 0 else 0)
            color = (1-norm_val, norm_val, 0.3)
            cell.set_facecolor(color)

# Mettre en forme l'en-tête
for j in range(len(metric_labels)):
    cell = metrics_table[(0, j)]
    cell.set_facecolor('#4c72b0')
    cell.set_text_props(weight='bold', color='white')

# 5. Évolution temporelle des valeurs moyennes
ax_time = plt.subplot(gs[3, :])
mois = np.arange(sim.total_mois + 1)
for nom_actif in donnees_mensuelles.keys():
    ax_time.plot(mois, donnees_mensuelles[nom_actif]['moyenne'], label=nom_actif, linewidth=2)

ax_time.plot(mois, sim.solde_pret, 'r--', label='Solde du Prêt', linewidth=2)
ax_time.axvline(x=sim.mois_differe, color='k', linestyle=':', label='Fin du Différé')
ax_time.set_title('Évolution Temporelle de la Valeur Moyenne du Portefeuille', fontsize=16)
ax_time.set_xlabel('Mois', fontsize=14)
ax_time.set_ylabel('Valeur (€)', fontsize=14)
ax_time.legend()
ax_time.grid(True, alpha=0.3)

# Ajouter des annotations pour les périodes clés
ax_time.annotate('Période de différé', xy=(sim.mois_differe/2, sim.montant_pret*1.1), 
                xytext=(sim.mois_differe/2, sim.montant_pret*1.3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                ha='center', fontsize=12)

ax_time.annotate('Période de remboursement', xy=(sim.mois_differe + (sim.total_mois-sim.mois_differe)/2, sim.montant_pret*0.6), 
                xytext=(sim.mois_differe + (sim.total_mois-sim.mois_differe)/2, sim.montant_pret*0.3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                ha='center', fontsize=12)

# 6. Graphiques de performance par profil d'investisseur
ax_profiles = plt.subplot(gs[4, :])
ax_profiles.axis('off')

# Définir les profils d'investisseurs et leurs préférences
profils = {
    'Prudent': {'rendement': 0.1, 'risque': 0.9},
    'Équilibré': {'rendement': 0.5, 'risque': 0.5},
    'Dynamique': {'rendement': 0.7, 'risque': 0.3},
    'Agressif': {'rendement': 0.9, 'risque': 0.1}
}

# Calculer les scores pour chaque profil
scores_profils = pd.DataFrame(index=resultats_df.index)

for profil, poids in profils.items():
    for nom_actif in resultats_df.index:
        score_rendement = scores.loc[nom_actif, 'score_rendement']
        score_risque = scores.loc[nom_actif, 'score_risque']
        scores_profils.loc[nom_actif, profil] = (score_rendement * poids['rendement'] + score_risque * poids['risque'])

# Créer un graphique à barres pour chaque profil
width = 0.15
x = np.arange(len(resultats_df.index))

# Créer un sous-graphique pour les profils
ax_profiles_bars = plt.subplot(gs[4, :])

for i, (profil, _) in enumerate(profils.items()):
    ax_profiles_bars.bar(x + i*width - width*1.5, scores_profils[profil], width, label=profil)

ax_profiles_bars.set_title('Adéquation des Classes d\'Actifs par Profil d\'Investisseur', fontsize=16)
ax_profiles_bars.set_xticks(x)
ax_profiles_bars.set_xticklabels(resultats_df.index, rotation=0)
ax_profiles_bars.set_ylabel('Score d\'Adéquation', fontsize=14)
ax_profiles_bars.set_ylim(0, 1)
ax_profiles_bars.legend()
ax_profiles_bars.grid(True, alpha=0.3)

# 7. Recommandations finales et résumé
ax_recommendations = plt.subplot(gs[5, :])
ax_recommendations.axis('off')

# Déterminer les recommandations par profil
reco_prudent = scores_profils['Prudent'].idxmax()
reco_equilibre = scores_profils['Équilibré'].idxmax()
reco_dynamique = scores_profils['Dynamique'].idxmax()
reco_agressif = scores_profils['Agressif'].idxmax()

# Texte de recommandation
reco_text = (
    "RECOMMANDATIONS FINALES\n\n"
    f"Meilleure option globale: {scores_tries.index[0]} (Score: {scores_tries['score_total'].iloc[0]:.2f})\n"
    f"Alternative recommandée: {scores_tries.index[1]} (Score: {scores_tries['score_total'].iloc[1]:.2f})\n\n"
    "Recommandations par profil d'investisseur:\n"
    f"• Profil Prudent: {reco_prudent} - Privilégie la sécurité et la préservation du capital\n"
    f"• Profil Équilibré: {reco_equilibre} - Recherche un bon compromis entre rendement et risque\n"
    f"• Profil Dynamique: {reco_dynamique} - Accepte un risque modéré pour un meilleur rendement\n"
    f"• Profil Agressif: {reco_agressif} - Vise la maximisation du rendement, peu sensible au risque\n\n"
    "POINTS CLÉS À CONSIDÉRER:\n"
    f"• La Cryptomonnaie offre le meilleur rendement potentiel mais avec un risque élevé (taux d'échec: {100-resultats_df.loc['Cryptomonnaie', 'taux_reussite']:.1f}%)\n"
    f"• Les ETF d'Actions offrent un bon équilibre entre rendement et sécurité\n"
    f"• L'Immobilier présente une bonne stabilité mais des coûts élevés\n"
    f"• Les Obligations d'État sont les plus sûres mais avec le rendement le plus faible\n"
    f"• Le Portefeuille Mixte offre une diversification qui peut réduire le risque global\n\n"
    "CONCLUSION:\n"
    "Le choix optimal dépend de votre tolérance au risque, de votre horizon d'investissement et de vos objectifs financiers."
)

ax_recommendations.text(0.5, 0.5, reco_text, ha='center', va='center', fontsize=14,
                      bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=1'))

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3)
plt.savefig(f"{repertoire_sortie}/tableau_bord_complet.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{repertoire_sortie}/tableau_bord_complet.pdf", bbox_inches='tight')
print(f"\nTableau de bord complet enregistré dans {repertoire_sortie}/tableau_bord_complet.png et .pdf")

# Afficher le chemin vers le fichier généré
print(f"\nVous pouvez consulter le tableau de bord complet ici: {os.path.abspath(f'{repertoire_sortie}/tableau_bord_complet.png')}")
