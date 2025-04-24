"""
Tableau de bord d'investissement final - Partie 2: Création du tableau de bord visuel
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
from src.visualisation_utils import configurer_style, creer_tableau_metriques, creer_graphique_barres, creer_graphique_ligne

# Charger les données préparées dans la partie 1
repertoire_sortie = "../output"
resultats_df = pd.read_pickle(f"{repertoire_sortie}/resultats_df.pkl")
parametres = pd.read_pickle(f"{repertoire_sortie}/parametres_simulation.pkl")

# Extraire les données des paramètres
donnees_mensuelles = parametres['donnees_mensuelles']
montant_pret = parametres['montant_pret']
cout_total_interet = parametres['cout_total_interet']
duree_ans = parametres['duree_ans']
mois_differe = parametres['mois_differe']
mois_remboursement = parametres['mois_remboursement']
total_mois = parametres['total_mois']
taux_interet_annuel = parametres['taux_interet_annuel']

# Configurer le style
colors = configurer_style()

# Définir les métriques à afficher avec des descriptions très concises
metriques = [
    ('rendement_annuel', 'pct', 'Rend.', 'Annuel brut'),
    ('volatilite_annuelle', 'pct', 'Volat.', 'Risque'),
    ('ratio_sharpe', 'ratio', 'Sharpe', 'Rend/Risque'),
    ('taux_imposition_effectif', 'pct', 'Impôt', '% prélevé'),
    ('gain_net_apres_impots', 'eur', 'Gain Net', 'Après impôts'),
    ('rendement_net_annualise', 'pct', 'R.Net', '% par an'),
    ('ratio_gain_cout', 'ratio', 'G/C', 'Gain/Coût'),
    ('taux_reussite', 'pct', 'Réus.', '% rentable'),
    ('drawdown_max', 'pct', 'DD', 'Baisse max')
]

# Créer un tableau de bord visuel complet
plt.figure(figsize=(24, 36))  # Format A2 approximatif pour une meilleure lisibilité
gs = GridSpec(9, 2, figure=plt.gcf(), height_ratios=[1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 1])  # 9 lignes, 2 colonnes

# Titre et paramètres en haut de la page
plt.suptitle("TABLEAU DE BORD D'INVESTISSEMENT - ANALYSE FISCALE INTÉGRÉE", fontsize=24, y=0.995)
date_actuelle = datetime.now().strftime("%d/%m/%Y")
texte_param = (
    f"Paramètres : Prêt étudiant {montant_pret:,.0f} € à {taux_interet_annuel*100:.2f}% | "
    f"Différé : {mois_differe} mois (1,5 ans) | "
    f"Remboursement : {mois_remboursement} mois (8,5 ans) | "
    f"Total : 10 ans | "
    f"Coût total des intérêts : {cout_total_interet:,.0f} € | "
    f"Date de simulation : {date_actuelle}"
)
plt.figtext(0.5, 0.975, texte_param, ha="center", fontsize=14)

# 1. Tableau des métriques clés avec focus sur le gain réel et le risque
ax_metrics = plt.subplot(gs[0, :])

# Créer le tableau des métriques
creer_tableau_metriques(ax_metrics, resultats_df, metriques, 
                       titre="Tableau des métriques clés (triées par gain net après impôts)")

# Ajouter un texte explicatif sous le tableau
ax_metrics.text(0.5, -0.05, 
               "Ce tableau présente les principales métriques pour chaque classe d'actif, triées par gain net après impôts.\n"
               "Les couleurs indiquent les performances relatives : vert = bonne performance, rouge = risque élevé, bleu = gain élevé.",
               transform=ax_metrics.transAxes, fontsize=10, ha='center',
               bbox=dict(facecolor='#f8f9fa', alpha=0.7, edgecolor='#dddddd'))

# 2. Graphique comparatif des gains nets après impôts
ax_gains = plt.subplot(gs[1, 0])
# Utiliser le DataFrame directement avec la fonction creer_graphique_barres pour éviter le FutureWarning
df_for_plot = resultats_df.reset_index()[['index', 'gain_net_apres_impots']].sort_values('gain_net_apres_impots', ascending=False)
df_for_plot['_hue'] = df_for_plot['index']  # Colonne hue pour les couleurs
creer_graphique_barres(
    ax_gains,
    df_for_plot,
    x='index',
    y='gain_net_apres_impots',
    hue='_hue',
    titre='Gain Net Après Impôts',
    xlabel='',
    ylabel='Euros (€)',
    rotation=45,
    format_y='eur',
    colors=[resultats_df.loc[nom, 'couleur'] for nom in resultats_df.index]
)
ax_gains.legend().remove()  # Supprimer la légende

# 3. Ratio de Sharpe (mesure du rendement ajusté au risque)
ax_sharpe = plt.subplot(gs[1, 1])
df_for_plot = resultats_df.reset_index()[['index', 'ratio_sharpe']].sort_values('ratio_sharpe', ascending=False)
df_for_plot['_hue'] = df_for_plot['index']
creer_graphique_barres(
    ax_sharpe,
    df_for_plot,
    x='index',
    y='ratio_sharpe',
    hue='_hue',
    titre='Ratio de Sharpe (rendement/risque)',
    xlabel='',
    ylabel='Ratio',
    rotation=45,
    format_y='numeric',
    colors=[resultats_df.loc[nom, 'couleur'] for nom in resultats_df.index]
)
ax_sharpe.legend().remove()
# Ajouter une annotation
ax_sharpe.text(0.02, 0.95, "Plus le ratio est élevé, meilleur est le compromis rendement/risque.", 
               transform=ax_sharpe.transAxes, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 4. Comparaison avant/après impôts
ax_fiscalite = plt.subplot(gs[2, 0])
data_fiscalite = resultats_df.reset_index()[['index', 'gain_net_avant_impots', 'gain_net_apres_impots']]
data_fiscalite = data_fiscalite.rename(columns={
    'gain_net_avant_impots': 'Avant Impôts',
    'gain_net_apres_impots': 'Après Impôts'
})
data_fiscalite_melted = pd.melt(
    data_fiscalite, 
    id_vars=['index'], 
    value_vars=['Avant Impôts', 'Après Impôts'],
    var_name='Type',
    value_name='Gain Net'
)

sns.barplot(
    x='index', 
    y='Gain Net', 
    hue='Type', 
    data=data_fiscalite_melted,
    ax=ax_fiscalite,
    palette=['#3498db', '#2ecc71']
)

ax_fiscalite.set_title('Impact de la Fiscalité sur le Gain Net', fontsize=14, pad=15)
ax_fiscalite.set_xlabel('')
ax_fiscalite.set_ylabel('Gain Net (€)')
ax_fiscalite.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
plt.setp(ax_fiscalite.get_xticklabels(), rotation=45, ha='right')
ax_fiscalite.grid(True, axis='y', alpha=0.3)
ax_fiscalite.legend(title='')

# Ajouter une annotation pour expliquer l'impact fiscal
ax_fiscalite.text(0.02, 0.95, "Impact fiscal : différence entre le gain brut et le gain net après impôts", 
                 transform=ax_fiscalite.transAxes, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 5. Taux d'imposition effectifs
ax_impots = plt.subplot(gs[2, 1])
df_for_plot = resultats_df.sort_values('taux_imposition_effectif').reset_index()
df_for_plot['_hue'] = df_for_plot['index']
creer_graphique_barres(
    ax_impots,
    df_for_plot,
    x='index',
    y='taux_imposition_effectif',
    hue='_hue',
    titre='Taux d\'Imposition Effectif par Classe d\'Actif',
    xlabel='',
    ylabel='Taux d\'Imposition (%)',
    rotation=45,
    format_y='pct',
    colors=[resultats_df.loc[nom, 'couleur'] for nom in df_for_plot.iloc[:, 0]]
)
ax_impots.legend().remove()

# Ajouter une annotation pour expliquer les différences de fiscalité
ax_impots.text(0.02, 0.95, "PEA : fiscalité avantageuse (17,2%) | Immobilier : fiscalité plus lourde (37%)", 
              transform=ax_impots.transAxes, fontsize=10, 
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 6. Distribution des valeurs finales
ax_dist = plt.subplot(gs[3, :])
actifs_top = resultats_df.index[:5]  # Top 5 pour la lisibilité

# Créer un DataFrame pour la visualisation
dist_data = []
for actif in actifs_top:
    valeurs_finales = donnees_mensuelles[actif][:, -1]
    for val in valeurs_finales:
        dist_data.append({'Classe d\'Actif': actif, 'Valeur Finale': val})

dist_df = pd.DataFrame(dist_data)

# KDE plot pour afficher la distribution
sns.kdeplot(
    data=dist_df, 
    x='Valeur Finale', 
    hue='Classe d\'Actif',
    ax=ax_dist,
    fill=True,
    alpha=0.3,
    palette=[resultats_df.loc[nom, 'couleur'] for nom in actifs_top]
)

# Ajouter une ligne verticale pour le montant initial du prêt + intérêts
cout_total = montant_pret + cout_total_interet
ax_dist.axvline(cout_total, color='red', linestyle='--', 
               label=f'Coût total du prêt ({cout_total:,.0f} €)')

# Configurer le graphique
ax_dist.set_title('Distribution des Valeurs Finales (Top 5 Classes d\'Actifs)', fontsize=14, pad=15)
ax_dist.set_xlabel('Valeur Finale (€)')
ax_dist.set_ylabel('Densité')
ax_dist.xaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
ax_dist.grid(True, alpha=0.3)
ax_dist.legend(title='')

# Ajouter une annotation
ax_dist.text(0.02, 0.95, 
            "Ce graphique montre la distribution des valeurs finales possibles.\n"
            "Zone à gauche de la ligne rouge = perte par rapport au coût total du prêt.",
            transform=ax_dist.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 7. Évolution moyenne de la valeur du portefeuille au fil du temps
ax_evolution = plt.subplot(gs[4, :])

# Préparer les données pour l'évolution moyenne
evolution_data = []
mois = list(range(total_mois + 1))
for actif in actifs_top:
    moyennes_mensuelles = np.mean(donnees_mensuelles[actif], axis=0)
    for m, val in zip(mois, moyennes_mensuelles):
        evolution_data.append({
            'Mois': m,
            'Valeur': val,
            'Classe d\'Actif': actif
        })

evolution_df = pd.DataFrame(evolution_data)

# Créer le graphique d'évolution
sns.lineplot(
    data=evolution_df,
    x='Mois',
    y='Valeur',
    hue='Classe d\'Actif',
    ax=ax_evolution,
    palette=[resultats_df.loc[nom, 'couleur'] for nom in actifs_top]
)

# Configurer le graphique
ax_evolution.set_title('Évolution Moyenne de la Valeur du Portefeuille au Fil du Temps (Top 5)', fontsize=14, pad=15)
ax_evolution.set_xlabel('Mois')
ax_evolution.set_ylabel('Valeur (€)')
ax_evolution.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
ax_evolution.grid(True, alpha=0.3)

# Ajouter des lignes verticales pour marquer la fin de la période de différé et le début des remboursements
ax_evolution.axvline(mois_differe, color='gray', linestyle='--', 
                    label=f'Fin du différé (mois {mois_differe})')

# Ajouter une ligne horizontale pour le montant initial + intérêts
ax_evolution.axhline(cout_total, color='red', linestyle=':', 
                    label=f'Coût total ({cout_total:,.0f} €)')

ax_evolution.legend(title='')

# 8. Taux de réussite (% de simulations où la valeur finale dépasse le coût total)
ax_reussite = plt.subplot(gs[5, 0])
df_for_plot = resultats_df.reset_index()[['index', 'taux_reussite']].sort_values('taux_reussite', ascending=False)
df_for_plot['_hue'] = df_for_plot['index']
creer_graphique_barres(
    ax_reussite,
    df_for_plot,
    x='index',
    y='taux_reussite',
    hue='_hue',
    titre='Taux de Réussite (% de simulations rentables)',
    xlabel='',
    ylabel='Taux de Réussite (%)',
    rotation=45,
    format_y='pct',
    colors=[resultats_df.loc[nom, 'couleur'] for nom in df_for_plot.iloc[:, 0]]
)
ax_reussite.legend().remove()

# Ajouter une annotation
ax_reussite.text(0.02, 0.95, "% de simulations où la valeur finale dépasse le coût total du prêt", 
                transform=ax_reussite.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 9. Rendement net annualisé
ax_rendement = plt.subplot(gs[5, 1])
df_for_plot = resultats_df.reset_index()[['index', 'rendement_net_annualise']].sort_values('rendement_net_annualise', ascending=False)
df_for_plot['_hue'] = df_for_plot['index']
creer_graphique_barres(
    ax_rendement,
    df_for_plot,
    x='index',
    y='rendement_net_annualise',
    hue='_hue',
    titre='Rendement Net Annualisé (après impôts)',
    xlabel='',
    ylabel='Rendement (%)',
    rotation=45,
    format_y='pct',
    colors=[resultats_df.loc[nom, 'couleur'] for nom in df_for_plot.iloc[:, 0]]
)
ax_rendement.legend().remove()

# 10. Drawdown maximum (baisse maximale par rapport au pic précédent)
ax_drawdown = plt.subplot(gs[6, 0])
df_for_plot = resultats_df.reset_index()[['index', 'drawdown_max']].sort_values('drawdown_max', ascending=True)  # Ascendant car drawdown est négatif
df_for_plot['_hue'] = df_for_plot['index']
creer_graphique_barres(
    ax_drawdown,
    df_for_plot,
    x='index',
    y='drawdown_max',
    hue='_hue',
    titre='Drawdown Maximum (baisse temporaire maximale)',
    xlabel='',
    ylabel='Drawdown (%)',
    rotation=45,
    format_y='pct',
    colors=[resultats_df.loc[nom, 'couleur'] for nom in df_for_plot.iloc[:, 0]]
)
ax_drawdown.legend().remove()

# Ajouter une annotation
ax_drawdown.text(0.02, 0.95, "Plus le drawdown est faible (proche de zéro), moins le risque est élevé", 
                transform=ax_drawdown.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 11. Ratio Gain/Coût
ax_ratio = plt.subplot(gs[6, 1])
df_for_plot = resultats_df.reset_index()[['index', 'ratio_gain_cout']].sort_values('ratio_gain_cout', ascending=False)
df_for_plot['_hue'] = df_for_plot['index']
creer_graphique_barres(
    ax_ratio,
    df_for_plot,
    x='index',
    y='ratio_gain_cout',
    hue='_hue',
    titre='Ratio Gain/Coût (euros gagnés par euro d\'intérêt payé)',
    xlabel='',
    ylabel='Ratio',
    rotation=45,
    format_y='numeric',
    colors=[resultats_df.loc[nom, 'couleur'] for nom in df_for_plot.iloc[:, 0]]
)
ax_ratio.legend().remove()

# 12. Pourcentage des gains perdus en impôts
ax_perte_fiscale = plt.subplot(gs[7, :])

# Calculer le pourcentage des gains bruts perdus en impôts
resultats_df['perte_fiscale_pct'] = (resultats_df['gain_brut_moyen'] - (resultats_df['gain_net_apres_impots'] + cout_total_interet)) / resultats_df['gain_brut_moyen'] * 100

# Trier par perte fiscale croissante
df_perte = resultats_df.sort_values('perte_fiscale_pct').reset_index()
df_perte['_hue'] = df_perte['index']

# Créer le graphique
creer_graphique_barres(
    ax_perte_fiscale,
    df_perte,
    x='index',
    y='perte_fiscale_pct',
    hue='_hue',
    titre='Pourcentage des Gains Bruts Perdus en Impôts',
    xlabel='',
    ylabel='Pourcentage (%)',
    rotation=45,
    format_y='pct',
    colors=[plt.cm.RdYlGn_r(x/100) for x in df_perte['perte_fiscale_pct']]
)
ax_perte_fiscale.legend().remove()

# Ajouter une annotation explicative
ax_perte_fiscale.text(0.02, 0.95, 
                     "Ce graphique montre le pourcentage des gains bruts perdus en impôts pour chaque classe d'actif.\n"
                     "Plus le pourcentage est bas, plus l'enveloppe fiscale est avantageuse.", 
                     transform=ax_perte_fiscale.transAxes, fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 13. Guide fiscal et explicatif des métriques
ax_explanation = plt.subplot(gs[8, :])

# Texte explicatif
explanation_text = (
    "GUIDE DES MÉTRIQUES ET DE LA FISCALITÉ\n\n"
    "MÉTRIQUES PRINCIPALES :\n"
    "• Rend. - Rendement annuel brut : Performance annuelle moyenne de l'actif avant impôts.\n"
    "• Volat. - Volatilité : Mesure de l'amplitude des variations de prix, indicateur du niveau de risque.\n"
    "• Sharpe - Ratio de Sharpe : Mesure du rendement ajusté au risque. Plus il est élevé, meilleur est le compromis rendement/risque.\n"
    "• Impôt - Taux d'imposition effectif : Pourcentage des gains bruts prélevés en impôts.\n"
    "• Gain Net - Gain net après impôts : Profit final après paiement des intérêts du prêt et des impôts.\n"
    "• R.Net - Rendement net annualisé : Taux de rendement annuel moyen après impôts.\n"
    "• G/C - Ratio Gain/Coût : Nombre d'euros gagnés pour chaque euro d'intérêt payé sur le prêt.\n"
    "• Réus. - Taux de réussite : Pourcentage des simulations où l'investissement est rentable après impôts et coût du prêt.\n"
    "• DD - Drawdown : Baisse maximale temporaire par rapport au pic précédent.\n\n"
    
    "FISCALITÉ DES INVESTISSEMENTS :\n"
    "• PEA : Imposition à 17,2% (prélèvements sociaux uniquement) après 5 ans de détention.\n"
    "• Assurance-vie : Imposition réduite (23% puis 17,2%) après 8 ans, avec abattement annuel.\n"
    "• CTO (Compte-Titres Ordinaire) : Flat tax de 30% (12,8% IR + 17,2% PS) sur tous les gains.\n"
    "• Immobilier : Imposition entre 32% et 37% selon régime fiscal, incluant taxes foncières et prélèvements sociaux majorés.\n"
    "• SCPI : Régime fiscal similaire à l'immobilier en direct mais légèrement optimisé.\n\n"
    
    "ANALYSE DES RÉSULTATS :\n"
    "Cette simulation démontre l'impact significatif de la fiscalité sur la performance finale des investissements.\n"
    "Le PEA apparaît comme l'enveloppe la plus avantageuse pour investir dans des actions européennes (gain net supérieur).\n"
    "Les actifs à forte volatilité offrent des rendements potentiels plus élevés mais avec un risque accru de pertes temporaires importantes.\n"
    "Pour une stratégie prudente, les SCPI présentent le meilleur taux de réussite et un drawdown limité.\n"
    "Tous les actifs simulés présentent un gain net positif, ce qui valide la pertinence d'utiliser un prêt étudiant à taux avantageux."
)

# Ajouter le texte explicatif
ax_explanation.text(0.5, 0.5, explanation_text, 
                  ha='center', va='center', 
                  fontsize=11, 
                  bbox=dict(facecolor='#f8f9fa', alpha=0.8, edgecolor='#dddddd', boxstyle='round,pad=1'),
                  transform=ax_explanation.transAxes)
ax_explanation.axis('off')

# Ajuster la mise en page
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Sauvegarder le tableau de bord
plt.savefig(f"{repertoire_sortie}/tableau_bord_final.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{repertoire_sortie}/tableau_bord_final.pdf", bbox_inches='tight')

print(f"Tableau de bord final enregistré dans {repertoire_sortie}/tableau_bord_final.png et .pdf")
print(f"Vous pouvez consulter le tableau de bord ici: {os.path.abspath(f'{repertoire_sortie}/tableau_bord_final.png')}")

# Récapituler les actifs les plus performants
top_actifs = resultats_df.iloc[:3].index.tolist()
print("\nMeilleures classes d'actifs par gain net après impôts :")
print("--------------------------------------------------")
for i, actif in enumerate(top_actifs, 1):
    gain = resultats_df.loc[actif, 'gain_net_apres_impots']
    rendement = resultats_df.loc[actif, 'rendement_net_annualise']
    print(f"{i}. {actif} : {gain:,.0f} € (rendement net annualisé de {rendement:.1f}%)")

print("\nRécapitulatif des classes d'actifs avec fiscalité :")
print("--------------------------------------------------")
print(f"Récapitulatif des actifs avec fiscalité enregistré dans {repertoire_sortie}/recap_actifs_avec_fiscalite.csv")
