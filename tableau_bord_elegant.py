"""
Tableau de bord d'investissement élégant avec fiscalité intégrée
Ce script génère un tableau de bord visuel amélioré pour comparer différentes options d'investissement
en tenant compte de la fiscalité applicable en France.
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
from actifs_enrichis import get_actifs_enrichis
from fiscalite import get_fiscalite_par_actif, appliquer_fiscalite
from visualisations import (configurer_style, creer_tableau_metriques, creer_graphique_barres,
                           creer_graphique_ligne, creer_graphique_distribution, 
                           creer_graphique_scatter, creer_texte_explicatif)

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

# Afficher les informations sur le prêt
print(f"Montant initial du prêt : {montant_pret:,.2f} €")
print(f"Montant différé après {mois_differe} mois : {montant_differe:,.2f} €")
print(f"Paiement mensuel pendant la période de remboursement : {paiement_mensuel:,.2f} €")
print(f"Total des paiements sur la durée du prêt : {paiement_mensuel * mois_remboursement:,.2f} €")
print(f"Coût total du prêt (intérêts) : {cout_total_interet:,.2f} €")

# Obtenir les actifs enrichis et la fiscalité
actifs = get_actifs_enrichis()
fiscalite_par_actif = get_fiscalite_par_actif()

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

# Fonction pour appliquer la fiscalité aux gains
def appliquer_fiscalite_aux_gains(gain_brut, nom_actif, duree_detention):
    """Applique la fiscalité appropriée aux gains selon le type d'actif"""
    if nom_actif not in fiscalite_par_actif:
        # Par défaut, appliquer le PFU (30%)
        return gain_brut * 0.7
    
    fiscalite = fiscalite_par_actif[nom_actif]
    taux_effectif = fiscalite.get('taux_effectif_global', 0.30)
    
    # Cas spéciaux selon la durée de détention
    if nom_actif == 'PEA ETF Européens' and duree_detention < 5:
        taux_effectif = 0.30  # PFU avant 5 ans
    elif nom_actif == 'Assurance-Vie Fonds Euro':
        if duree_detention >= 8:
            taux_effectif = 0.247  # 17.2% PS + 7.5% IR après 8 ans
        elif duree_detention >= 4:
            taux_effectif = 0.297  # 17.2% PS + 12.5% IR entre 4 et 8 ans
        else:
            taux_effectif = 0.30  # PFU avant 4 ans
    
    return gain_brut * (1 - taux_effectif)

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

# Simuler tous les actifs
resultats = {}
donnees_mensuelles = {}

for nom_actif, actif in actifs_filtres.items():
    print(f"Simulation de {nom_actif}...")
    valeurs = simuler_portefeuille(actif, n_simulations, total_mois, montant_pret)
    
    # Calculer les statistiques sur les valeurs finales
    valeurs_finales = valeurs[:, -1]
    
    # Calculer les gains bruts et nets
    gain_brut_moyen = np.mean(valeurs_finales) - montant_pret
    gain_net_avant_impots = gain_brut_moyen - cout_total_interet
    
    # Appliquer la fiscalité
    gain_net_apres_impots = appliquer_fiscalite_aux_gains(gain_brut_moyen, nom_actif, duree_ans)
    gain_net_apres_impots_et_cout = gain_net_apres_impots - cout_total_interet
    
    # Calculer les résultats pour cet actif
    resultats[nom_actif] = {
        'nom': actif['nom'],
        'description': actif.get('description', ''),
        'profil_risque': actif.get('profil_risque', ''),
        'horizon_recommande': actif.get('horizon_recommande', ''),
        'rendement_annuel': actif['rendement_annuel'] * 100,
        'volatilite_annuelle': actif['volatilite_annuelle'] * 100,
        'ratio_sharpe': calculer_ratio_sharpe(actif['rendement_annuel'], actif['volatilite_annuelle']),
        'moyenne_valeur_finale': np.mean(valeurs_finales),
        'mediane_valeur_finale': np.median(valeurs_finales),
        'ecart_type_valeur_finale': np.std(valeurs_finales),
        'gain_brut_moyen': gain_brut_moyen,
        'gain_net_avant_impots': gain_net_avant_impots,
        'gain_net_apres_impots': gain_net_apres_impots - cout_total_interet,
        'rendement_net_annualise': ((np.mean(valeurs_finales) / montant_pret) ** (1/10) - 1) * 100,
        'ratio_gain_cout': gain_net_apres_impots_et_cout / cout_total_interet,
        'taux_imposition_effectif': fiscalite_par_actif.get(nom_actif, {}).get('taux_effectif_global', 0.30) * 100,
        'valeur_risque_5': np.percentile(valeurs_finales, 5),
        'valeur_risque_1': np.percentile(valeurs_finales, 1),
        'percentile_25': np.percentile(valeurs_finales, 25),
        'percentile_75': np.percentile(valeurs_finales, 75),
        'percentile_95': np.percentile(valeurs_finales, 95),
        'min': np.min(valeurs_finales),
        'max': np.max(valeurs_finales),
        'taux_reussite': np.mean(valeurs_finales > (montant_pret + cout_total_interet)) * 100,
        'drawdown_max': calculer_drawdown_max(valeurs),
        'couleur': actif.get('couleur', colors[actifs_a_simuler.index(nom_actif) % len(colors)])
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
resultats_df = resultats_df.sort_values('gain_net_apres_impots', ascending=False)

# Créer un tableau de bord visuel complet
plt.figure(figsize=(24, 34))  # Augmenter davantage la taille
gs = GridSpec(8, 2, figure=plt.gcf(), height_ratios=[1, 1, 1, 1, 1, 1, 0.8, 1])  # Ajouter une ligne supplémentaire

# Titre et paramètres
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

# Définir les métriques à afficher
metriques = [
    ('rendement_annuel', 'pct', 'Rend.', 'Perf. annuelle'),
    ('volatilite_annuelle', 'pct', 'Volat.', 'Risque'),
    ('ratio_sharpe', 'ratio', 'Sharpe', '>1 = bon'),
    ('taux_imposition_effectif', 'pct', 'Impôt', '% prélevé'),
    ('gain_net_apres_impots', 'eur', 'Gain Net', 'Après impôts'),
    ('rendement_net_annualise', 'pct', 'Rend. Net', 'Annualisé'),
    ('ratio_gain_cout', 'ratio', 'G/C', 'Gain/coût prêt'),
    ('taux_reussite', 'pct', 'Réussite', '% profitable'),
    ('drawdown_max', 'pct', 'Drawdown', 'Baisse max')
]

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
creer_graphique_barres(
    ax_gains, 
    resultats_df.reset_index(), 
    x='index', 
    y='gain_net_apres_impots',
    titre='Gain Net Après Impôts par Classe d\'Actif',
    xlabel='',
    ylabel='Gain Net (€)',
    colors=[resultats_df.loc[nom, 'couleur'] for nom in resultats_df.index],
    rotation=45
)

# Ajouter une ligne horizontale à zéro
ax_gains.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 3. Graphique comparatif des ratios de Sharpe
ax_sharpe = plt.subplot(gs[1, 1])
sharpe_df = resultats_df.sort_values('ratio_sharpe', ascending=False)
creer_graphique_barres(
    ax_sharpe,
    sharpe_df.reset_index(),
    x='index',
    y='ratio_sharpe',
    titre='Ratio de Sharpe par Classe d\'Actif',
    xlabel='',
    ylabel='Ratio de Sharpe',
    colors=[sharpe_df.loc[nom, 'couleur'] for nom in sharpe_df.index],
    rotation=45,
    format_y='none'
)

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
max_y = data_fiscalite_melted['Gain Net'].max()
ax_fiscalite.text(0.02, 0.95, "Impact fiscal : différence entre le gain brut et le gain net après impôts", 
                 transform=ax_fiscalite.transAxes, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 5. Taux d'imposition effectifs
ax_impots = plt.subplot(gs[2, 1])
creer_graphique_barres(
    ax_impots,
    resultats_df.sort_values('taux_imposition_effectif').reset_index(),
    x='index',
    y='taux_imposition_effectif',
    titre='Taux d\'Imposition Effectif par Classe d\'Actif',
    xlabel='',
    ylabel='Taux d\'Imposition (%)',
    rotation=45,
    format_y='pct'
)

# Ajouter une annotation pour expliquer les différences de fiscalité
ax_impots.text(0.02, 0.95, "PEA : fiscalité avantageuse (17,2%) | Immobilier : fiscalité plus lourde (37%)", 
              transform=ax_impots.transAxes, fontsize=10, 
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 6. Distribution des valeurs finales
ax_dist = plt.subplot(gs[3, :])
actifs_top = resultats_df.index[:7]  # Top 7 pour la lisibilité
valeurs_finales_dict = {nom: donnees_mensuelles[nom]['valeurs'][:, -1] for nom in actifs_top}
labels = [f"{nom} (μ={np.mean(valeurs_finales_dict[nom]):,.0f} €)" for nom in actifs_top]
couleurs = [resultats_df.loc[nom, 'couleur'] for nom in actifs_top]

references = [
    (montant_pret, 'Montant Initial du Prêt', 'red', '--'),
    (montant_pret + cout_total_interet, 'Montant Prêt + Intérêts', 'orange', '--')
]

creer_graphique_distribution(
    ax_dist,
    valeurs_finales_dict,
    actifs_top,
    labels,
    titre='Distribution des Valeurs Finales par Classe d\'Actif',
    xlabel='Valeur Finale (€)',
    ylabel='Densité',
    colors=couleurs,
    references=references
)

# 7. Évolution temporelle des valeurs moyennes
ax_time = plt.subplot(gs[4, :])
mois = np.arange(total_mois + 1)
donnees_evolution = {nom: donnees_mensuelles[nom]['moyenne'] for nom in actifs_top}
couleurs = [resultats_df.loc[nom, 'couleur'] for nom in actifs_top]

# Calculer le solde du prêt
solde_pret = np.zeros(total_mois + 1)
solde_pret[0] = montant_pret
for t in range(1, mois_differe + 1):
    solde_pret[t] = solde_pret[t-1] * (1 + taux_interet_mensuel)

for t in range(mois_differe + 1, total_mois + 1):
    paiement_interet = solde_pret[t-1] * taux_interet_mensuel
    paiement_principal = paiement_mensuel - paiement_interet
    solde_pret[t] = max(0, solde_pret[t-1] - paiement_principal)

# Ajouter le solde du prêt aux données d'évolution
donnees_evolution['Solde du Prêt'] = solde_pret
actifs_top_avec_pret = list(actifs_top) + ['Solde du Prêt']
couleurs_avec_pret = couleurs + ['red']

creer_graphique_ligne(
    ax_time,
    donnees_evolution,
    mois,
    actifs_top_avec_pret,
    actifs_top_avec_pret,
    titre='Évolution Temporelle de la Valeur Moyenne du Portefeuille',
    xlabel='Mois',
    ylabel='Valeur (€)',
    colors=couleurs_avec_pret
)

# Ajouter une ligne verticale pour la fin du différé
ax_time.axvline(x=mois_differe, color='black', linestyle=':', label='Fin du Différé')

# Ajouter une ligne horizontale pour le capital + intérêts
ax_time.axhline(y=montant_pret + cout_total_interet, color='orange', linestyle='--', 
               label='Capital + Intérêts')

# 8. Graphique de risque/rendement
ax_risk = plt.subplot(gs[5, 0])
creer_graphique_scatter(
    ax_risk,
    resultats_df,
    x='volatilite_annuelle',
    y='rendement_annuel',
    titre='Profil Risque/Rendement des Classes d\'Actifs',
    xlabel='Volatilité Annuelle (%)',
    ylabel='Rendement Annuel (%)',
    colors=[resultats_df.loc[nom, 'couleur'] for nom in resultats_df.index]
)

# 9. Graphique risque/rendement après impôts
ax_risk_apres = plt.subplot(gs[5, 1])
# Calculer le rendement net après impôts annualisé
resultats_df['rendement_net_apres_impots_annualise'] = resultats_df.apply(
    lambda x: ((1 + x['gain_net_apres_impots'] / montant_pret) ** (1/10) - 1) * 100, 
    axis=1
)

creer_graphique_scatter(
    ax_risk_apres,
    resultats_df,
    x='volatilite_annuelle',
    y='rendement_net_apres_impots_annualise',
    titre='Profil Risque/Rendement Net Après Impôts',
    xlabel='Volatilité Annuelle (%)',
    ylabel='Rendement Net Après Impôts (%)',
    colors=[resultats_df.loc[nom, 'couleur'] for nom in resultats_df.index]
)

# 10. Nouveau graphique : Pourcentage des gains perdus en impôts
ax_perte_fiscale = plt.subplot(gs[6, :])

# Calculer le pourcentage des gains bruts perdus en impôts
resultats_df['perte_fiscale_pct'] = (resultats_df['gain_brut_moyen'] - (resultats_df['gain_net_apres_impots'] + cout_total_interet)) / resultats_df['gain_brut_moyen'] * 100

# Trier par perte fiscale croissante
df_perte = resultats_df.sort_values('perte_fiscale_pct').reset_index()

# Créer le graphique
bars = sns.barplot(x='index', y='perte_fiscale_pct', data=df_perte, ax=ax_perte_fiscale, 
                  palette=[plt.cm.RdYlGn_r(x/100) for x in df_perte['perte_fiscale_pct']])

# Configurer le graphique
ax_perte_fiscale.set_title('Pourcentage des Gains Bruts Perdus en Impôts', fontsize=14, pad=15)
ax_perte_fiscale.set_xlabel('')
ax_perte_fiscale.set_ylabel('Pourcentage (%)')
ax_perte_fiscale.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.setp(ax_perte_fiscale.get_xticklabels(), rotation=45, ha='right')
ax_perte_fiscale.grid(True, axis='y', alpha=0.3)

# Annoter les barres
for i, p in enumerate(ax_perte_fiscale.patches):
    height = p.get_height()
    ax_perte_fiscale.annotate(f"{height:.1f}%",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9)

# Ajouter une annotation explicative
ax_perte_fiscale.text(0.02, 0.95, 
                     "Ce graphique montre le pourcentage des gains bruts perdus en impôts pour chaque classe d'actif.\nPlus le pourcentage est bas, plus l'enveloppe fiscale est avantageuse.", 
                     transform=ax_perte_fiscale.transAxes, fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 11. Guide fiscal et explicatif des métriques
ax_explanation = plt.subplot(gs[7, :])

# Texte explicatif
explanation_text = (
    "GUIDE DES MÉTRIQUES ET DE LA FISCALITÉ DE L'INVESTISSEMENT\n\n"
    "MÉTRIQUES CLÉS EXPLIQUÉES :\n"
    "• Rendement Annuel : Performance moyenne attendue par an avant impôts et frais.\n"
    "• Volatilité : Mesure de l'ampleur des fluctuations de valeur. Plus elle est élevée, plus l'investissement est risqué.\n"
    "• Ratio de Sharpe : Mesure du rendement excédentaire par unité de risque. Un ratio > 1 est généralement bon, > 2 est excellent.\n"
    "• Taux d'Imposition : Pourcentage des gains prélevés par la fiscalité selon le régime applicable à chaque actif.\n"
    "• Gain Net : Profit final après déduction des impôts et du coût du prêt (intérêts).\n"
    "• Rendement Net Annualisé : Taux de croissance annuel moyen après impôts et frais.\n"
    "• Ratio Gain/Coût : Combien d'euros de profit net sont générés pour chaque euro d'intérêt payé sur le prêt.\n"
    "• Taux de Réussite : Pourcentage des simulations où l'investissement génère un profit net positif.\n"
    "• Drawdown Maximum : Plus grande baisse temporaire de valeur. Indique l'ampleur des pertes potentielles à court terme.\n\n"
    "FISCALITÉ DES INVESTISSEMENTS EN FRANCE (2025) :\n"
    "- PFU (Flat Tax) : 30% sur les revenus de capitaux mobiliers et plus-values (ETF, obligations)\n"
    "- PEA : Exonération d'impôt sur le revenu après 5 ans (uniquement 17,2% de prélèvements sociaux)\n"
    "- Immobilier : Imposition au barème progressif de l'IR après abattements (environ 37% en moyenne)\n"
    "- SCPI : Revenus imposés comme des revenus fonciers, plus-values comme des plus-values mobilières\n"
    "- Assurance-Vie : Fiscalité avantageuse après 8 ans (7,5% + 17,2% PS) avec abattement annuel\n\n"
    "CONSIDÉRATIONS IMPORTANTES POUR LA PRISE DE DÉCISION :\n"
    "1. Votre tolérance au risque - Êtes-vous prêt à supporter des fluctuations importantes pour un rendement potentiellement plus élevé?\n"
    "2. Votre horizon d'investissement - Pouvez-vous maintenir votre stratégie pendant toute la durée du prêt (10 ans)?\n"
    "3. Votre situation fiscale personnelle - Les avantages fiscaux de certains produits dépendent de votre tranche marginale d'imposition.\n"
    "4. L'enveloppe fiscale - Le choix de l'enveloppe (CTO, PEA, Assurance-vie) peut avoir un impact significatif sur le rendement net final.\n\n"
    f"AVANTAGE UNIQUE : Avec un prêt étudiant à seulement {taux_interet_annuel*100:.2f}%, l'effet de levier reste très favorable même après impôts."
)

creer_texte_explicatif(ax_explanation, explanation_text, titre="GUIDE FISCAL ET MÉTRIQUES D'INVESTISSEMENT")

# Finaliser et sauvegarder
plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=4, w_pad=3)  # Augmenter l'espacement
plt.savefig(f"{repertoire_sortie}/tableau_bord_elegant.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{repertoire_sortie}/tableau_bord_elegant.pdf", bbox_inches='tight')
print(f"\nTableau de bord élégant enregistré dans {repertoire_sortie}/tableau_bord_elegant.png et .pdf")

# Afficher le chemin vers le fichier généré
print(f"\nVous pouvez consulter le tableau de bord élégant ici: {os.path.abspath(f'{repertoire_sortie}/tableau_bord_elegant.png')}")

# Créer un tableau récapitulatif des profils d'actifs avec fiscalité
print("\nRécapitulatif des classes d'actifs avec fiscalité :")
print("--------------------------------------------------")
recap_df = resultats_df[['nom', 'description', 'profil_risque', 'horizon_recommande', 
                         'rendement_annuel', 'volatilite_annuelle', 'ratio_sharpe', 
                         'taux_imposition_effectif', 'gain_net_apres_impots', 
                         'ratio_gain_cout', 'taux_reussite']]

# Formater les colonnes numériques
recap_df_formatted = recap_df.copy()
recap_df_formatted['rendement_annuel'] = recap_df['rendement_annuel'].apply(lambda x: f"{x:.1f}%")
recap_df_formatted['volatilite_annuelle'] = recap_df['volatilite_annuelle'].apply(lambda x: f"{x:.1f}%")
recap_df_formatted['ratio_sharpe'] = recap_df['ratio_sharpe'].apply(lambda x: f"{x:.2f}")
recap_df_formatted['taux_imposition_effectif'] = recap_df['taux_imposition_effectif'].apply(lambda x: f"{x:.1f}%")
recap_df_formatted['gain_net_apres_impots'] = recap_df['gain_net_apres_impots'].apply(lambda x: f"{x:,.0f} €")
recap_df_formatted['ratio_gain_cout'] = recap_df['ratio_gain_cout'].apply(lambda x: f"{x:.2f}")
recap_df_formatted['taux_reussite'] = recap_df['taux_reussite'].apply(lambda x: f"{x:.1f}%")

# Enregistrer le récapitulatif au format CSV
recap_df_formatted.to_csv(f"{repertoire_sortie}/recap_actifs_avec_fiscalite.csv", index=True)
print(f"Récapitulatif des actifs avec fiscalité enregistré dans {repertoire_sortie}/recap_actifs_avec_fiscalite.csv")
