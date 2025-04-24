import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from investment_sim import SimulationInvestissement
import os
from tabulate import tabulate
from scipy import stats

# Définir le style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (14, 8)

# Créer le répertoire de sortie s'il n'existe pas
repertoire_sortie = "../output"
if not os.path.exists(repertoire_sortie):
    os.makedirs(repertoire_sortie)

# Initialiser la simulation avec plus de simulations pour une meilleure précision
sim = SimulationInvestissement(mois_remboursement=102, n_simulations=10000)  # 8,5 ans, 10000 simulations
print(f"Montant initial du prêt : {sim.montant_pret:,.2f} €")
print(f"Montant différé après {sim.mois_differe} mois : {sim.montant_differe:,.2f} €")
print(f"Paiement mensuel pendant la période de remboursement : {sim.paiement_mensuel:,.2f} €")
print(f"Total des paiements sur la durée du prêt : {sim.paiement_mensuel * sim.mois_remboursement:,.2f} €")
print(f"Coût total du prêt : {(sim.paiement_mensuel * sim.mois_remboursement) - sim.montant_pret:,.2f} €")

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
    
    # Ratio de Sortino (approximation simplifiée)
    rendements_negatifs = valeurs_valides[valeurs_valides < sim.montant_differe]
    volatilite_negative = np.std(rendements_negatifs) / sim.montant_differe if len(rendements_negatifs) > 0 else 0
    ratio_sortino = (rendement_moyen - taux_sans_risque) / volatilite_negative if volatilite_negative != 0 else 0
    resultats_df.loc[nom_actif, 'ratio_sortino'] = ratio_sortino
    
    # Probabilité de perte
    resultats_df.loc[nom_actif, 'probabilite_perte'] = np.sum(valeurs_valides < sim.montant_differe) / len(valeurs_valides) * 100
    
    # Perte maximale
    resultats_df.loc[nom_actif, 'perte_maximale'] = (np.nanmin(valeurs_valides) - sim.montant_differe) / sim.montant_differe * 100
    
    # Gain maximal
    resultats_df.loc[nom_actif, 'gain_maximal'] = (np.nanmax(valeurs_valides) - sim.montant_differe) / sim.montant_differe * 100
    
    # Skewness (asymétrie) et Kurtosis (aplatissement)
    resultats_df.loc[nom_actif, 'asymetrie'] = stats.skew(valeurs_valides)
    resultats_df.loc[nom_actif, 'aplatissement'] = stats.kurtosis(valeurs_valides)

# 1. Tableau récapitulatif complet
print("\n=== TABLEAU RÉCAPITULATIF COMPLET ===")
tableau_complet = resultats_df[['taux_reussite', 'moyenne_valeur_finale', 'mediane_valeur_finale', 
                              'var_95', 'rendement_annualise_moyen', 'impots_moyens_payes', 
                              'frais_transaction_moyens', 'ratio_sharpe', 'ratio_sortino', 
                              'probabilite_perte', 'perte_maximale', 'gain_maximal']].copy()

# Formater les colonnes pour l'affichage
tableau_formatte = tableau_complet.copy()
for col in ['moyenne_valeur_finale', 'mediane_valeur_finale', 'var_95', 'impots_moyens_payes', 'frais_transaction_moyens']:
    tableau_formatte[col] = tableau_formatte[col].apply(lambda x: f"{x:,.2f} €")
for col in ['taux_reussite', 'rendement_annualise_moyen', 'probabilite_perte', 'perte_maximale', 'gain_maximal']:
    tableau_formatte[col] = tableau_formatte[col].apply(lambda x: f"{x:.2f}%")
for col in ['ratio_sharpe', 'ratio_sortino', 'asymetrie', 'aplatissement']:
    if col in tableau_formatte.columns:
        tableau_formatte[col] = tableau_formatte[col].apply(lambda x: f"{x:.4f}")

# Renommer les colonnes pour une meilleure lisibilité
tableau_formatte.columns = [
    'Taux de Réussite', 'Valeur Finale Moyenne', 'Valeur Finale Médiane', 'VaR 5%', 
    'Rendement Annualisé Moyen', 'Impôts Moyens Payés', 'Frais de Transaction Moyens',
    'Ratio de Sharpe', 'Ratio de Sortino', 'Probabilité de Perte', 'Perte Maximale (%)', 'Gain Maximal (%)'
]

print(tabulate(tableau_formatte, headers='keys', tablefmt='pretty'))

# 2. Analyse du risque vs rendement
plt.figure(figsize=(12, 8))
x = resultats_df['rendement_annualise_moyen']
y = 100 - resultats_df['taux_reussite']  # Convertir le taux de réussite en risque d'échec
taille = resultats_df['moyenne_valeur_finale'] / 50000  # Taille proportionnelle à la valeur finale

plt.scatter(x, y, s=taille, alpha=0.7)
for i, nom in enumerate(resultats_df.index):
    plt.annotate(nom, (x[i], y[i]), fontsize=12)

plt.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Seuil de risque 10%')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='Seuil de rendement 0%')

plt.title('Analyse Risque vs Rendement', fontsize=16)
plt.xlabel('Rendement Annualisé Moyen (%)', fontsize=14)
plt.ylabel('Risque d\'Échec (%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{repertoire_sortie}/analyse_risque_rendement.png", dpi=300)
print(f"\nAnalyse risque vs rendement enregistrée dans {repertoire_sortie}/analyse_risque_rendement.png")

# 3. Comparaison des distributions de valeurs finales
plt.figure(figsize=(14, 8))
for i, nom_actif in enumerate(donnees_portefeuille.keys()):
    valeurs_finales = donnees_portefeuille[nom_actif][:, -1]
    valeurs_valides = valeurs_finales[~np.isnan(valeurs_finales)]
    if len(valeurs_valides) > 0:
        sns.kdeplot(valeurs_valides, label=nom_actif, fill=True, alpha=0.3)

plt.axvline(x=sim.montant_differe, color='r', linestyle='--', label='Montant Initial Différé')
plt.title('Distribution des Valeurs Finales par Classe d\'Actif', fontsize=16)
plt.xlabel('Valeur Finale (€)', fontsize=14)
plt.ylabel('Densité', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{repertoire_sortie}/distribution_valeurs_finales.png", dpi=300)
print(f"Distribution des valeurs finales enregistrée dans {repertoire_sortie}/distribution_valeurs_finales.png")

# 4. Évolution temporelle des valeurs moyennes
plt.figure(figsize=(14, 8))
mois = np.arange(sim.total_mois + 1)
for nom_actif in donnees_mensuelles.keys():
    plt.plot(mois, donnees_mensuelles[nom_actif]['moyenne'], label=nom_actif)

plt.plot(mois, sim.solde_pret, 'r--', label='Solde du Prêt')
plt.axvline(x=sim.mois_differe, color='k', linestyle=':', label='Fin du Différé')
plt.title('Évolution Temporelle de la Valeur Moyenne du Portefeuille', fontsize=16)
plt.xlabel('Mois', fontsize=14)
plt.ylabel('Valeur (€)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{repertoire_sortie}/evolution_temporelle.png", dpi=300)
print(f"Évolution temporelle enregistrée dans {repertoire_sortie}/evolution_temporelle.png")

# 5. Analyse des rendements par percentile
print("\n=== ANALYSE DES RENDEMENTS PAR PERCENTILE ===")
percentiles_df = pd.DataFrame(index=resultats_df.index)
percentiles = [5, 25, 50, 75, 95]

for p in percentiles:
    for nom_actif in resultats_df.index:
        valeurs_finales = donnees_portefeuille[nom_actif][:, -1]
        percentile_val = np.nanpercentile(valeurs_finales, p)
        rendement = ((percentile_val / sim.montant_differe) ** (1/10) - 1) * 100  # Rendement annualisé sur 10 ans
        percentiles_df.loc[nom_actif, f'p{p}'] = rendement

# Formater pour l'affichage
percentiles_df_formatte = percentiles_df.applymap(lambda x: f"{x:.2f}%")
percentiles_df_formatte.columns = [f'Percentile {p}%' for p in percentiles]
print(tabulate(percentiles_df_formatte, headers='keys', tablefmt='pretty'))

# 6. Analyse de sensibilité au temps d'investissement
print("\n=== ANALYSE DE SENSIBILITÉ AU TEMPS D'INVESTISSEMENT ===")
horizons = [3, 5, 7, 10]  # Années
sensibilite_df = pd.DataFrame(index=resultats_df.index)

for horizon in horizons:
    mois_horizon = horizon * 12
    if mois_horizon <= sim.total_mois:
        for nom_actif in resultats_df.index:
            valeurs_horizon = donnees_portefeuille[nom_actif][:, mois_horizon]
            valeurs_valides = valeurs_horizon[~np.isnan(valeurs_horizon)]
            if len(valeurs_valides) > 0:
                taux_reussite = np.mean(valeurs_valides >= sim.solde_pret[mois_horizon]) * 100
                sensibilite_df.loc[nom_actif, f'h{horizon}'] = taux_reussite

# Formater pour l'affichage
sensibilite_df_formatte = sensibilite_df.applymap(lambda x: f"{x:.2f}%")
sensibilite_df_formatte.columns = [f'Horizon {h} ans' for h in horizons if h * 12 <= sim.total_mois]
print(tabulate(sensibilite_df_formatte, headers='keys', tablefmt='pretty'))

# 7. Analyse des coûts cumulés (impôts + frais de transaction)
print("\n=== ANALYSE DES COÛTS CUMULÉS ===")
couts_df = pd.DataFrame(index=resultats_df.index)

for nom_actif in resultats_df.index:
    impots_totaux = resultats_df.loc[nom_actif, 'impots_moyens_payes']
    frais_totaux = resultats_df.loc[nom_actif, 'frais_transaction_moyens']
    couts_totaux = impots_totaux + frais_totaux
    valeur_finale = resultats_df.loc[nom_actif, 'moyenne_valeur_finale']
    
    couts_df.loc[nom_actif, 'impots'] = impots_totaux
    couts_df.loc[nom_actif, 'frais'] = frais_totaux
    couts_df.loc[nom_actif, 'total'] = couts_totaux
    couts_df.loc[nom_actif, 'pourcentage'] = (couts_totaux / valeur_finale) * 100

# Formater pour l'affichage
couts_formatte = couts_df.copy()
for col in ['impots', 'frais', 'total']:
    couts_formatte[col] = couts_formatte[col].apply(lambda x: f"{x:,.2f} €")
couts_formatte['pourcentage'] = couts_formatte['pourcentage'].apply(lambda x: f"{x:.2f}%")

couts_formatte.columns = ['Impôts Totaux', 'Frais de Transaction', 'Coûts Totaux', 'Pourcentage de la Valeur Finale']
print(tabulate(couts_formatte, headers='keys', tablefmt='pretty'))

# 8. Analyse des ratios risque/rendement
print("\n=== ANALYSE DES RATIOS RISQUE/RENDEMENT ===")
ratios_df = resultats_df[['ratio_sharpe', 'ratio_sortino', 'asymetrie', 'aplatissement']].copy()
ratios_df['rendement_par_risque'] = resultats_df['rendement_annualise_moyen'] / (100 - resultats_df['taux_reussite'])
ratios_df['rendement_par_var'] = resultats_df['rendement_annualise_moyen'] / ((resultats_df['moyenne_valeur_finale'] - resultats_df['var_95']) / resultats_df['moyenne_valeur_finale'] * 100)

# Formater pour l'affichage
ratios_formatte = ratios_df.copy()
for col in ratios_formatte.columns:
    ratios_formatte[col] = ratios_formatte[col].apply(lambda x: f"{x:.4f}")

ratios_formatte.columns = ['Ratio de Sharpe', 'Ratio de Sortino', 'Asymétrie', 'Aplatissement', 
                          'Rendement/Risque', 'Rendement/VaR']
print(tabulate(ratios_formatte, headers='keys', tablefmt='pretty'))

# 9. Conclusion et recommandation
print("\n=== CONCLUSION ET RECOMMANDATION ===")

# Calculer un score pour chaque classe d'actif basé sur plusieurs facteurs
scores = pd.DataFrame(index=resultats_df.index)

# Normaliser les valeurs pour le score
scores['score_rendement'] = (resultats_df['rendement_annualise_moyen'] - resultats_df['rendement_annualise_moyen'].min()) / (resultats_df['rendement_annualise_moyen'].max() - resultats_df['rendement_annualise_moyen'].min()) if resultats_df['rendement_annualise_moyen'].max() != resultats_df['rendement_annualise_moyen'].min() else 0
scores['score_risque'] = resultats_df['taux_reussite'] / 100
scores['score_valeur'] = (resultats_df['moyenne_valeur_finale'] - resultats_df['moyenne_valeur_finale'].min()) / (resultats_df['moyenne_valeur_finale'].max() - resultats_df['moyenne_valeur_finale'].min()) if resultats_df['moyenne_valeur_finale'].max() != resultats_df['moyenne_valeur_finale'].min() else 0
scores['score_sharpe'] = (resultats_df['ratio_sharpe'] - resultats_df['ratio_sharpe'].min()) / (resultats_df['ratio_sharpe'].max() - resultats_df['ratio_sharpe'].min()) if resultats_df['ratio_sharpe'].max() != resultats_df['ratio_sharpe'].min() else 0
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

# Afficher les scores
scores_formatte = scores.copy()
for col in scores_formatte.columns:
    scores_formatte[col] = scores_formatte[col].apply(lambda x: f"{x:.4f}")

scores_formatte.columns = ['Score Rendement', 'Score Risque', 'Score Valeur', 'Score Sharpe', 'Score Coûts', 'Score Total']
print(tabulate(scores_formatte.sort_values('Score Total', ascending=False), headers='keys', tablefmt='pretty'))

# Recommandation
meilleur_actif = scores_tries.index[0]
deuxieme_actif = scores_tries.index[1]
print(f"\nRECOMMANDATION PRINCIPALE: {meilleur_actif}")
print(f"Alternative recommandée: {deuxieme_actif}")

# Profils d'investisseurs
print("\nRECOMMANDATIONS PAR PROFIL D'INVESTISSEUR:")
print(f"- Profil Prudent: {resultats_df['taux_reussite'].idxmax()}")
print(f"- Profil Équilibré: {resultats_df.loc[resultats_df['taux_reussite'] > 85, 'rendement_annualise_moyen'].idxmax() if any(resultats_df['taux_reussite'] > 85) else 'Portefeuille Mixte'}")
print(f"- Profil Dynamique: {resultats_df.loc[resultats_df['taux_reussite'] > 50, 'rendement_annualise_moyen'].idxmax() if any(resultats_df['taux_reussite'] > 50) else 'ETF d\'Actions'}")
print(f"- Profil Agressif: {resultats_df['rendement_annualise_moyen'].idxmax()}")

# Enregistrer les résultats dans un fichier
with open(f"{repertoire_sortie}/analyse_complete.txt", "w") as f:
    f.write("=== ANALYSE COMPLÈTE DES OPTIONS D'INVESTISSEMENT ===\n\n")
    f.write(f"Montant initial du prêt : {sim.montant_pret:,.2f} €\n")
    f.write(f"Montant différé après {sim.mois_differe} mois : {sim.montant_differe:,.2f} €\n")
    f.write(f"Paiement mensuel pendant la période de remboursement : {sim.paiement_mensuel:,.2f} €\n\n")
    
    f.write("=== TABLEAU RÉCAPITULATIF COMPLET ===\n")
    f.write(tabulate(tableau_formatte, headers='keys', tablefmt='grid'))
    f.write("\n\n")
    
    f.write("=== ANALYSE DES RENDEMENTS PAR PERCENTILE ===\n")
    f.write(tabulate(percentiles_df_formatte, headers='keys', tablefmt='grid'))
    f.write("\n\n")
    
    f.write("=== ANALYSE DE SENSIBILITÉ AU TEMPS D'INVESTISSEMENT ===\n")
    f.write(tabulate(sensibilite_df_formatte, headers='keys', tablefmt='grid'))
    f.write("\n\n")
    
    f.write("=== ANALYSE DES COÛTS CUMULÉS ===\n")
    f.write(tabulate(couts_formatte, headers='keys', tablefmt='grid'))
    f.write("\n\n")
    
    f.write("=== ANALYSE DES RATIOS RISQUE/RENDEMENT ===\n")
    f.write(tabulate(ratios_formatte, headers='keys', tablefmt='grid'))
    f.write("\n\n")
    
    f.write("=== RECOMMANDATION ===\n")
    f.write(tabulate(scores_formatte.sort_values('Score Total', ascending=False), headers='keys', tablefmt='grid'))
    f.write("\n\n")
    
    f.write(f"RECOMMANDATION PRINCIPALE: {meilleur_actif}\n")
    f.write(f"Alternative recommandée: {deuxieme_actif}\n\n")
    
    f.write("RECOMMANDATIONS PAR PROFIL D'INVESTISSEUR:\n")
    f.write(f"- Profil Prudent: {resultats_df['taux_reussite'].idxmax()}\n")
    f.write(f"- Profil Équilibré: {resultats_df.loc[resultats_df['taux_reussite'] > 85, 'rendement_annualise_moyen'].idxmax() if any(resultats_df['taux_reussite'] > 85) else 'Portefeuille Mixte'}\n")
    f.write(f"- Profil Dynamique: {resultats_df.loc[resultats_df['taux_reussite'] > 50, 'rendement_annualise_moyen'].idxmax() if any(resultats_df['taux_reussite'] > 50) else 'ETF d\'Actions'}\n")
    f.write(f"- Profil Agressif: {resultats_df['rendement_annualise_moyen'].idxmax()}\n")

print(f"\nAnalyse complète enregistrée dans {repertoire_sortie}/analyse_complete.txt")
