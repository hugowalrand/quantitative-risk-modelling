import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from investment_sim import SimulationInvestissement
import os

# Définir le style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Créer le répertoire de sortie s'il n'existe pas
repertoire_sortie = "../output"
if not os.path.exists(repertoire_sortie):
    os.makedirs(repertoire_sortie)

# Initialiser la simulation
sim = SimulationInvestissement(mois_remboursement=102)  # 8,5 ans
print(f"Montant initial du prêt : {sim.montant_pret:,.2f} €")
print(f"Montant différé après {sim.mois_differe} mois : {sim.montant_differe:,.2f} €")
print(f"Paiement mensuel pendant la période de remboursement : {sim.paiement_mensuel:,.2f} €")

# Exécuter la simulation
liste_resultats = []
donnees_portefeuille = {}
donnees_impots = {}
donnees_frais_transaction = {}

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

# Créer un DataFrame de résultats
resultats_df = pd.DataFrame(liste_resultats)
resultats_df.set_index('nom_actif', inplace=True)

# Créer une grande figure pour le rapport complet
plt.figure(figsize=(24, 36))
gs = GridSpec(8, 2, figure=plt.gcf())  # Taille de grille augmentée pour plus d'espace

# Titre et paramètres
plt.suptitle("Rapport Complet de Simulation d'Investissement", fontsize=24, y=0.995)
texte_param = (
    f"Paramètres : Prêt {sim.montant_pret:,.2f} € à {sim.taux_interet_annuel*100:.2f}% | "
    f"Différé : {sim.mois_differe} mois (1,5 ans) | "
    f"Remboursement : {sim.mois_remboursement} mois (8,5 ans) | "
    f"Total : 10 ans | "
    f"Inflation : {sim.inflation_mensuelle*1200:.1f}%"  # Convertir mensuel en annuel et en pourcentage
)
plt.figtext(0.5, 0.975, texte_param, ha="center", fontsize=14)

# 1. Tableau récapitulatif
ax_table = plt.subplot(gs[0, :])
ax_table.axis('tight')
ax_table.axis('off')

# Formater les données pour le tableau
donnees_tableau = resultats_df.copy()
donnees_tableau = donnees_tableau.round(2)
# Formater les colonnes monétaires
for col in ['moyenne_valeur_finale', 'mediane_valeur_finale', 'var_95', 'valeur_max', 
           'valeur_min', 'profit_perte_moyen', 'impots_moyens_payes', 'frais_transaction_moyens']:
    donnees_tableau[col] = donnees_tableau[col].apply(lambda x: f"{x:,.2f} €")
# Formater les colonnes de pourcentage
for col in ['taux_reussite', 'rendement_annualise_moyen']:
    donnees_tableau[col] = donnees_tableau[col].apply(lambda x: f"{x:.2f}%")

# Créer le tableau
etiquettes_colonnes = [
    'Taux de Réussite', 'Valeur Finale Moyenne', 'Valeur Finale Médiane', 'VaR 5%', 
    'Valeur Max', 'Valeur Min', 'Profit/Perte Moyen', 'Rendement Annuel Moyen', 
    'Impôts Moyens Payés', 'Frais de Transaction Moyens'
]
tableau = ax_table.table(
    cellText=donnees_tableau.values,
    rowLabels=donnees_tableau.index,
    colLabels=etiquettes_colonnes,
    cellLoc='center',
    loc='center',
    colWidths=[0.08] * len(etiquettes_colonnes)
)
tableau.auto_set_font_size(False)
tableau.set_fontsize(10)
tableau.scale(1, 1.5)
for (i, j), cell in tableau.get_celld().items():
    if i == 0:  # Ligne d'en-tête
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4c72b0')
    elif j == -1:  # Colonne d'étiquettes de ligne
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f0f0f0')

# 2. Comparaison des taux de réussite
ax_success = plt.subplot(gs[1, 0])
bars = sns.barplot(x=resultats_df.index, y='taux_reussite', data=resultats_df, ax=ax_success)
ax_success.set_title('Taux de Réussite par Classe d\'Actif', fontsize=14)
ax_success.set_xlabel('Classe d\'Actif', fontsize=12)
ax_success.set_ylabel('Taux de Réussite (%)', fontsize=12)
ax_success.set_ylim(0, 100)
for i, v in enumerate(resultats_df['taux_reussite']):
    ax_success.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=10)
plt.setp(ax_success.get_xticklabels(), rotation=45, ha='right')

# 3. Comparaison des valeurs finales moyennes
ax_final = plt.subplot(gs[1, 1])
bars = sns.barplot(x=resultats_df.index, y='moyenne_valeur_finale', data=resultats_df, ax=ax_final)
ax_final.set_title('Valeur Finale Moyenne par Classe d\'Actif', fontsize=14)
ax_final.set_xlabel('Classe d\'Actif', fontsize=12)
ax_final.set_ylabel('Valeur Finale Moyenne (€)', fontsize=12)
for i, v in enumerate(resultats_df['moyenne_valeur_finale']):
    ax_final.text(i, v + 5000, f"{v:,.0f} €", ha='center', fontsize=10)
ax_final.axhline(y=sim.montant_pret, color='r', linestyle='--', label='Prêt Initial')
ax_final.legend()
plt.setp(ax_final.get_xticklabels(), rotation=45, ha='right')

# Obtenir les noms d'actifs et configurer les couleurs
noms_actifs = list(donnees_portefeuille.keys())
couleurs = plt.cm.viridis(np.linspace(0, 1, len(noms_actifs)))

# 4. Chemins d'échantillon pour chaque classe d'actif - une ligne par actif (5 actifs = 5 lignes)
for i, nom_actif in enumerate(noms_actifs):
    ax_paths = plt.subplot(gs[2+i, 0])
    
    # Obtenir les valeurs du portefeuille pour cet actif
    valeurs_portefeuille = donnees_portefeuille[nom_actif]
    
    # Tracer le solde du prêt
    x = np.arange(sim.total_mois + 1)
    ax_paths.plot(x, sim.solde_pret, 'r-', linewidth=2, label='Solde du Prêt')
    
    # Tracer les chemins d'échantillon (seulement 50 pour la clarté)
    n_chemins = 50
    for j in range(min(n_chemins, sim.n_simulations)):
        ax_paths.plot(x, valeurs_portefeuille[j], alpha=0.1, color=couleurs[i])
    
    # Ajouter une ligne verticale pour la fin de la période de différé
    ax_paths.axvline(x=sim.mois_differe, color='r', linestyle='--',
                   label='Fin de la Période de Différé')
    
    # Ajouter des annotations
    ax_paths.set_title(f'{nom_actif} - Chemins de Simulation Échantillons', fontsize=14)
    ax_paths.set_xlabel('Mois', fontsize=12)
    ax_paths.set_ylabel('Valeur du Portefeuille (€)', fontsize=12)
    
    # Ajouter l'annotation du taux de réussite
    taux_reussite = resultats_df.loc[nom_actif, 'taux_reussite']
    ax_paths.text(0.05, 0.95, f"Taux de Réussite : {taux_reussite:.1f}%", 
                 transform=ax_paths.transAxes, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8))
    
    # Ajouter la légende
    if i == 0:  # Ajouter la légende uniquement au premier graphique
        ax_paths.legend(loc='upper right')

    # 5. Distributions finales pour chaque actif (dans la colonne de droite)
    ax_dist = plt.subplot(gs[2+i, 1])
    
    # Obtenir les valeurs du portefeuille pour cet actif
    valeurs_finales = valeurs_portefeuille[:, -1]
    valeurs_valides = valeurs_finales[~np.isnan(valeurs_finales)]
    
    if len(valeurs_valides) > 0:
        # Tracer l'histogramme avec KDE
        sns.histplot(valeurs_valides, kde=True, ax=ax_dist, color=couleurs[i])
        
        # Ajouter des lignes verticales
        ax_dist.axvline(x=sim.montant_pret, color='r', linestyle='--',
                       label='Montant Initial du Prêt')
        var_95 = np.nanpercentile(valeurs_finales, 5)
        ax_dist.axvline(x=var_95, color='orange', 
                       linestyle='--', label='VaR 5%')
        
        # Ajouter des annotations
        ax_dist.set_title(f'{nom_actif} - Distribution des Valeurs Finales du Portefeuille', fontsize=14)
        ax_dist.set_xlabel('Valeur Finale du Portefeuille (€)', fontsize=12)
        ax_dist.set_ylabel('Nombre', fontsize=12)
        
        # Ajouter l'annotation des statistiques
        valeur_moyenne = np.nanmean(valeurs_finales)
        valeur_mediane = np.nanmedian(valeurs_finales)
        texte_stats = (
            f"Moyenne : {valeur_moyenne:,.0f} €\n"
            f"Médiane : {valeur_mediane:,.0f} €\n"
            f"VaR 5% : {var_95:,.0f} €\n"
            f"Réussite : {np.mean(~np.isnan(valeurs_finales))*100:.1f}%"
        )
        ax_dist.text(0.95, 0.95, texte_stats, 
                    transform=ax_dist.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8),
                    ha='right', va='top')
        
        # Ajouter la légende
        ax_dist.legend()
    else:
        ax_dist.text(0.5, 0.5, "Pas de simulations réussies", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14)

# Ajouter un tableau des caractéristiques des classes d'actifs en bas
ax_asset_table = plt.subplot(gs[7, :])
ax_asset_table.axis('tight')
ax_asset_table.axis('off')

# Préparer les données des classes d'actifs
donnees_actifs = []
for cle, actif in sim.actifs.items():
    donnees_actifs.append([
        actif.nom,
        f"{actif.rendement_annuel*100:.1f}%",
        f"{actif.volatilite_annuelle*100:.1f}%",
        f"{actif.rendement_revenu_annuel*100:.1f}%",
        f"{actif.taux_imposition_revenu*100:.1f}%",
        f"{actif.frais_annuels*100:.2f}%",
        f"{actif.liquidite*100:.1f}%",
        f"{actif.taux_vacance*100:.1f}%" if actif.taux_vacance > 0 else "N/A",
        f"{actif.cout_entretien*100:.1f}%" if actif.cout_entretien > 0 else "N/A"
    ])

# Créer le tableau des caractéristiques des actifs
etiquettes_colonnes_actifs = [
    'Classe d\'Actif', 'Rendement Annuel', 'Volatilité', 'Rendement Revenu', 
    'Taux d\'Imposition', 'Frais Annuels', 'Liquidité', 'Taux de Vacance', 'Coût d\'Entretien'
]
tableau_actifs = ax_asset_table.table(
    cellText=donnees_actifs,
    colLabels=etiquettes_colonnes_actifs,
    cellLoc='center',
    loc='center',
    colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tableau_actifs.auto_set_font_size(False)
tableau_actifs.set_fontsize(10)
tableau_actifs.scale(1, 1.5)
for (i, j), cell in tableau_actifs.get_celld().items():
    if i == 0:  # Ligne d'en-tête
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4c72b0')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{repertoire_sortie}/rapport_complet_investissement.png", dpi=300, bbox_inches='tight')
print(f"\nRapport complet enregistré dans {repertoire_sortie}/rapport_complet_investissement.png")

# Enregistrer également en PDF pour une meilleure qualité
plt.savefig(f"{repertoire_sortie}/rapport_complet_investissement.pdf", bbox_inches='tight')
print(f"Rapport complet également enregistré en PDF dans {repertoire_sortie}/rapport_complet_investissement.pdf")

# Fermer la figure
plt.close()

# Afficher les résultats clés
print("\nRésultats Clés de la Simulation d'Investissement :")
print(f"Montant du prêt : {sim.montant_pret:,.2f} €")
print(f"Montant différé après {sim.mois_differe} mois (1,5 ans) : {sim.montant_differe:,.2f} €")
print(f"Paiement mensuel pendant {sim.mois_remboursement} mois (8,5 ans) : {sim.paiement_mensuel:,.2f} €")
print("\nTaux de Réussite :")
for nom_actif, taux_reussite in zip(resultats_df.index, resultats_df['taux_reussite']):
    print(f"  {nom_actif}: {taux_reussite:.2f}%")

print("\nValeurs Finales Moyennes :")
for nom_actif, valeur_finale in zip(resultats_df.index, resultats_df['moyenne_valeur_finale']):
    print(f"  {nom_actif}: {valeur_finale:,.2f} €")

print("\nRendements Annualisés Moyens :")
for nom_actif, rendement_annuel in zip(resultats_df.index, resultats_df['rendement_annualise_moyen']):
    print(f"  {nom_actif}: {rendement_annuel:.2f}%")
