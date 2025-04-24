import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from investment_sim import SimulationInvestissement, ClasseActif
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

# Initialiser la simulation de base
sim = SimulationInvestissement(mois_remboursement=102, n_simulations=10000)  # 8,5 ans, 10000 simulations

# Modifier les actifs disponibles
# 1. Supprimer la cryptomonnaie
if 'crypto' in sim.actifs:
    del sim.actifs['crypto']

# 2. Créer plusieurs répartitions de portefeuille mixte
# Conserver le portefeuille mixte original (60% actions, 40% obligations)
sim.actifs['portefeuille_mixte_60_40'] = sim.actifs.pop('portefeuille_mixte')
sim.actifs['portefeuille_mixte_60_40'].nom = 'Portefeuille Mixte (60/40)'

# Ajouter un portefeuille 80% actions, 20% obligations
sim.actifs['portefeuille_mixte_80_20'] = ClasseActif(
    nom="Portefeuille Mixte (80/20)",
    rendement_annuel=sim.actifs['etf_actions'].rendement_annuel * 0.8 + sim.actifs['obligations_etat'].rendement_annuel * 0.2,
    volatilite_annuelle=sim.actifs['etf_actions'].volatilite_annuelle * 0.8 + sim.actifs['obligations_etat'].volatilite_annuelle * 0.2,
    rendement_revenu_annuel=sim.actifs['etf_actions'].rendement_revenu_annuel * 0.8 + sim.actifs['obligations_etat'].rendement_revenu_annuel * 0.2,
    taux_imposition_revenu=sim.actifs['etf_actions'].taux_imposition_revenu * 0.8 + sim.actifs['obligations_etat'].taux_imposition_revenu * 0.2,
    frais_annuels=sim.actifs['etf_actions'].frais_annuels * 0.8 + sim.actifs['obligations_etat'].frais_annuels * 0.2,
    liquidite=sim.actifs['etf_actions'].liquidite * 0.8 + sim.actifs['obligations_etat'].liquidite * 0.2
)

# Ajouter un portefeuille 40% actions, 60% obligations
sim.actifs['portefeuille_mixte_40_60'] = ClasseActif(
    nom="Portefeuille Mixte (40/60)",
    rendement_annuel=sim.actifs['etf_actions'].rendement_annuel * 0.4 + sim.actifs['obligations_etat'].rendement_annuel * 0.6,
    volatilite_annuelle=sim.actifs['etf_actions'].volatilite_annuelle * 0.4 + sim.actifs['obligations_etat'].volatilite_annuelle * 0.6,
    rendement_revenu_annuel=sim.actifs['etf_actions'].rendement_revenu_annuel * 0.4 + sim.actifs['obligations_etat'].rendement_revenu_annuel * 0.6,
    taux_imposition_revenu=sim.actifs['etf_actions'].taux_imposition_revenu * 0.4 + sim.actifs['obligations_etat'].taux_imposition_revenu * 0.6,
    frais_annuels=sim.actifs['etf_actions'].frais_annuels * 0.4 + sim.actifs['obligations_etat'].frais_annuels * 0.6,
    liquidite=sim.actifs['etf_actions'].liquidite * 0.4 + sim.actifs['obligations_etat'].liquidite * 0.6
)

# Ajouter un portefeuille équilibré avec immobilier (40% actions, 30% obligations, 30% immobilier)
sim.actifs['portefeuille_equilibre'] = ClasseActif(
    nom="Portefeuille Équilibré",
    rendement_annuel=sim.actifs['etf_actions'].rendement_annuel * 0.4 + 
                    sim.actifs['obligations_etat'].rendement_annuel * 0.3 + 
                    sim.actifs['immobilier'].rendement_annuel * 0.3,
    volatilite_annuelle=sim.actifs['etf_actions'].volatilite_annuelle * 0.4 + 
                        sim.actifs['obligations_etat'].volatilite_annuelle * 0.3 + 
                        sim.actifs['immobilier'].volatilite_annuelle * 0.3,
    rendement_revenu_annuel=sim.actifs['etf_actions'].rendement_revenu_annuel * 0.4 + 
                            sim.actifs['obligations_etat'].rendement_revenu_annuel * 0.3 + 
                            sim.actifs['immobilier'].rendement_revenu_annuel * 0.3,
    taux_imposition_revenu=sim.actifs['etf_actions'].taux_imposition_revenu * 0.4 + 
                            sim.actifs['obligations_etat'].taux_imposition_revenu * 0.3 + 
                            sim.actifs['immobilier'].taux_imposition_revenu * 0.3,
    frais_annuels=sim.actifs['etf_actions'].frais_annuels * 0.4 + 
                sim.actifs['obligations_etat'].frais_annuels * 0.3 + 
                sim.actifs['immobilier'].frais_annuels * 0.3,
    liquidite=sim.actifs['etf_actions'].liquidite * 0.4 + 
            sim.actifs['obligations_etat'].liquidite * 0.3 + 
            sim.actifs['immobilier'].liquidite * 0.3,
    taux_vacance=sim.actifs['immobilier'].taux_vacance * 0.3,
    cout_entretien=sim.actifs['immobilier'].cout_entretien * 0.3
)

# Afficher les informations sur le prêt
print(f"Montant initial du prêt : {sim.montant_pret:,.2f} €")
print(f"Montant différé après {sim.mois_differe} mois : {sim.montant_differe:,.2f} €")
print(f"Paiement mensuel pendant la période de remboursement : {sim.paiement_mensuel:,.2f} €")
print(f"Total des paiements sur la durée du prêt : {sim.paiement_mensuel * sim.mois_remboursement:,.2f} €")
print(f"Coût total du prêt (intérêts) : {(sim.paiement_mensuel * sim.mois_remboursement) - sim.montant_pret:,.2f} €")

# Exécuter la simulation
liste_resultats = []
donnees_portefeuille = {}
donnees_impots = {}
donnees_frais_transaction = {}
donnees_mensuelles = {}

# CORRECTION: Modifier la méthode de simulation pour ne pas déduire les paiements du prêt du portefeuille
# Nous allons simuler la croissance du portefeuille indépendamment des remboursements du prêt
def simuler_portefeuille_corrige(sim, actif):
    np.random.seed(42)  # Pour la reproductibilité
    
    # Initialiser les tableaux
    valeurs_portefeuille = np.zeros((sim.n_simulations, sim.total_mois + 1))
    valeurs_portefeuille[:, 0] = sim.montant_pret
    
    # Suivre les impôts cumulés payés
    impots_payes = np.zeros((sim.n_simulations, sim.total_mois + 1))
    
    # Suivre les frais de transaction
    frais_transaction = np.zeros((sim.n_simulations, sim.total_mois + 1))
    
    for t in range(sim.total_mois):
        # Générer des rendements aléatoires
        rendements = np.random.normal(
            actif.rendement_mensuel,
            actif.volatilite_mensuelle,
            sim.n_simulations
        )
        
        # Calculer le revenu (avant impôt)
        rendement_revenu_mensuel = actif.rendement_revenu_annuel / 12
        
        # Pour l'immobilier, appliquer le taux de vacance
        if actif.nom == 'Immobilier':
            # Générer des événements de vacance
            evenements_vacance = np.random.random(sim.n_simulations) < actif.taux_vacance
            # Ajuster le revenu pour les vacances
            revenu = valeurs_portefeuille[:, t] * rendement_revenu_mensuel * (~evenements_vacance)
            
            # Appliquer les coûts d'entretien
            entretien = valeurs_portefeuille[:, t] * (actif.cout_entretien / 12)
            valeurs_portefeuille[:, t] -= entretien
        else:
            revenu = valeurs_portefeuille[:, t] * rendement_revenu_mensuel
        
        # Appliquer les impôts sur le revenu
        impot_revenu = revenu * actif.taux_imposition_revenu
        revenu_apres_impot = revenu - impot_revenu
        impots_payes[:, t+1] = impots_payes[:, t] + impot_revenu
        
        # Appliquer les frais mensuels
        frais_mensuels = valeurs_portefeuille[:, t] * (actif.frais_annuels / 12)
        
        # Calculer la croissance
        croissance = valeurs_portefeuille[:, t] * rendements
        
        # Mettre à jour les valeurs du portefeuille
        valeurs_portefeuille[:, t + 1] = (
            valeurs_portefeuille[:, t] +
            croissance +
            revenu_apres_impot -
            frais_mensuels
        )
        
        # CORRECTION: Ne pas déduire les paiements du prêt du portefeuille
        # Le prêt est remboursé séparément des investissements
        
        # Vérifier les conditions de défaut (valeur tombe en dessous d'un certain seuil)
        default_mask = valeurs_portefeuille[:, t + 1] < (sim.montant_pret * 0.1)  # 10% du montant initial
        valeurs_portefeuille[default_mask, t + 1:] = np.nan
    
    return valeurs_portefeuille, impots_payes, frais_transaction

# Exécuter les simulations pour chaque classe d'actif avec la méthode corrigée
for cle_actif, actif in sim.actifs.items():
    print(f"\nSimulation de {actif.nom}...")
    valeurs_portefeuille, impots_payes, frais_transaction = simuler_portefeuille_corrige(sim, actif)
    
    # Utiliser la méthode d'analyse existante
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

# Calculer le coût total des intérêts
cout_total_pret = (sim.paiement_mensuel * sim.mois_remboursement)
interets_payes = cout_total_pret - sim.montant_pret

# Calculer des métriques supplémentaires pour clarifier le gain réel
for nom_actif in resultats_df.index:
    # Gain net moyen (valeur finale - montant prêt - intérêts payés)
    resultats_df.loc[nom_actif, 'gain_net_moyen'] = resultats_df.loc[nom_actif, 'moyenne_valeur_finale'] - sim.montant_pret - interets_payes
    
    # Gain net médian
    resultats_df.loc[nom_actif, 'gain_net_median'] = resultats_df.loc[nom_actif, 'mediane_valeur_finale'] - sim.montant_pret - interets_payes
    
    # Rendement net annualisé (basé sur le gain net)
    resultats_df.loc[nom_actif, 'rendement_net_annualise'] = (
        ((resultats_df.loc[nom_actif, 'moyenne_valeur_finale'] / sim.montant_pret) ** (1/10) - 1) * 100
    )
    
    # Ratio gain/coût (combien on gagne pour chaque euro d'intérêt payé)
    resultats_df.loc[nom_actif, 'ratio_gain_cout'] = resultats_df.loc[nom_actif, 'gain_net_moyen'] / interets_payes if interets_payes > 0 else 0

# Créer un tableau de bord visuel complet
plt.figure(figsize=(20, 24))
gs = GridSpec(5, 2, figure=plt.gcf(), height_ratios=[0.8, 1, 1, 1, 1])

# Titre et paramètres
plt.suptitle("TABLEAU DE BORD - ANALYSE DES OPTIONS D'INVESTISSEMENT (CORRIGÉ)", fontsize=24, y=0.995)
texte_param = (
    f"Paramètres : Prêt {sim.montant_pret:,.0f} € à {sim.taux_interet_annuel*100:.2f}% | "
    f"Différé : {sim.mois_differe} mois (1,5 ans) | "
    f"Remboursement : {sim.mois_remboursement} mois (8,5 ans) | "
    f"Total : 10 ans | "
    f"Coût total des intérêts : {interets_payes:,.0f} €"
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
        f"{resultats_df.loc[nom_actif, 'taux_reussite']:.1f}%",
        f"{resultats_df.loc[nom_actif, 'gain_net_moyen']:,.0f} €",
        f"{resultats_df.loc[nom_actif, 'gain_net_median']:,.0f} €",
        f"{resultats_df.loc[nom_actif, 'rendement_net_annualise']:.2f}%",
        f"{resultats_df.loc[nom_actif, 'ratio_gain_cout']:.2f}",
        f"{resultats_df.loc[nom_actif, 'impots_moyens_payes']:,.0f} €",
        f"{resultats_df.loc[nom_actif, 'frais_transaction_moyens']:,.0f} €"
    ])

metric_labels = [
    'Classe d\'Actif', 'Taux de Réussite', 'Gain Net Moyen', 'Gain Net Médian',
    'Rendement Net Annualisé', 'Ratio Gain/Coût', 'Impôts Moyens', 'Frais de Transaction'
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
gains_nets = resultats_df['gain_net_moyen'].sort_values(ascending=False)
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
rendements_nets = resultats_df['rendement_net_annualise'].sort_values(ascending=False)
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
for nom_actif in donnees_portefeuille.keys():
    valeurs_finales = donnees_portefeuille[nom_actif][:, -1]
    valeurs_valides = valeurs_finales[~np.isnan(valeurs_finales)]
    if len(valeurs_valides) > 0:
        sns.kdeplot(valeurs_valides, label=nom_actif, fill=True, alpha=0.3)

# Ajouter des lignes verticales pour les points de référence
ax_dist.axvline(x=sim.montant_pret, color='r', linestyle='--', label='Montant Initial du Prêt')
ax_dist.axvline(x=sim.montant_pret + interets_payes, color='orange', linestyle='--', 
               label='Montant Prêt + Intérêts')

ax_dist.set_title('Distribution des Valeurs Finales par Classe d\'Actif', fontsize=16)
ax_dist.set_xlabel('Valeur Finale (€)', fontsize=14)
ax_dist.set_ylabel('Densité', fontsize=14)
ax_dist.legend()
ax_dist.grid(True, alpha=0.3)

# Ajouter une annotation pour expliquer le gain net
ax_dist.annotate('Gain Net = Valeur Finale - (Prêt + Intérêts)', 
                xy=(sim.montant_pret + interets_payes, 0.000005),
                xytext=(sim.montant_pret + interets_payes + 50000, 0.000008),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                ha='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))

# 5. Évolution temporelle des valeurs moyennes
ax_time = plt.subplot(gs[3, :])
mois = np.arange(sim.total_mois + 1)
for nom_actif in donnees_mensuelles.keys():
    ax_time.plot(mois, donnees_mensuelles[nom_actif]['moyenne'], label=nom_actif, linewidth=2)

# Ajouter le solde du prêt
ax_time.plot(mois, sim.solde_pret, 'r--', label='Solde du Prêt', linewidth=2)
ax_time.axvline(x=sim.mois_differe, color='k', linestyle=':', label='Fin du Différé')
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
    "EXPLICATION DES MÉTRIQUES ET CORRECTIONS\n\n"
    "• Taux de Réussite : Pourcentage des simulations où la valeur finale du portefeuille est supérieure au solde du prêt\n\n"
    "• Gain Net Moyen : Valeur finale moyenne - Montant du prêt - Intérêts payés\n"
    f"  (Intérêts totaux payés sur la durée du prêt : {interets_payes:,.0f} €)\n\n"
    "• Rendement Net Annualisé : Taux de rendement annuel équivalent basé sur la croissance du portefeuille\n\n"
    "• Ratio Gain/Coût : Combien d'euros de gain net pour chaque euro d'intérêt payé\n\n"
    "CORRECTION IMPORTANTE : Dans cette version corrigée, nous simulons la croissance du portefeuille indépendamment\n"
    "des remboursements du prêt. En effet, dans la réalité, vous remboursez le prêt avec vos revenus externes et non\n"
    "en prélevant sur votre portefeuille d'investissement. Cela permet de voir le véritable potentiel de croissance\n"
    "des investissements par rapport au coût du prêt.\n\n"
    "REMARQUE SUR LES RÉSULTATS :\n"
    "Les rendements positifs confirment que l'investissement génère plus que les intérêts, créant un bénéfice net.\n"
    "Le ratio gain/coût est particulièrement utile : un ratio > 1 signifie que vous gagnez plus que ce que vous coûtent les intérêts."
)

ax_explanation.text(0.5, 0.5, explanation_text, ha='center', va='center', fontsize=14,
                  bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=1'))

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3)
plt.savefig(f"{repertoire_sortie}/tableau_bord_corrige.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{repertoire_sortie}/tableau_bord_corrige.pdf", bbox_inches='tight')
print(f"\nTableau de bord corrigé enregistré dans {repertoire_sortie}/tableau_bord_corrige.png et .pdf")

# Afficher le chemin vers le fichier généré
print(f"\nVous pouvez consulter le tableau de bord corrigé ici: {os.path.abspath(f'{repertoire_sortie}/tableau_bord_corrige.png')}")
