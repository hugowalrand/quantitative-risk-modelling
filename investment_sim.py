import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

@dataclass
class ClasseActif:
    nom: str
    rendement_annuel: float
    volatilite_annuelle: float
    rendement_revenu_annuel: float
    taux_imposition_revenu: float
    frais_annuels: float
    rendement_mensuel: float = None
    volatilite_mensuelle: float = None
    liquidite: float = 1.0  # Échelle 0-1, 1 étant parfaitement liquide
    taux_vacance: float = 0.0  # Pour l'immobilier
    cout_entretien: float = 0.0  # Pour l'immobilier, en fraction de la valeur du bien

    def __post_init__(self):
        self.rendement_mensuel = self.rendement_annuel / 12
        self.volatilite_mensuelle = self.volatilite_annuelle / np.sqrt(12)

class SimulationInvestissement:
    def __init__(self, 
                 montant_pret: float = 200_000,
                 taux_interet_annuel: float = 0.0099,
                 mois_differe: int = 18,
                 mois_remboursement: int = 102,
                 inflation_annuelle: float = 0.02,
                 n_simulations: int = 10_000,
                 seuil_defaut: float = 0.2,  # La valeur du portefeuille tombe en dessous de 20% du prêt restant
                 repertoire_sortie: str = "output"):
        
        self.montant_pret = montant_pret
        self.taux_interet_annuel = taux_interet_annuel
        self.taux_interet_mensuel = taux_interet_annuel / 12
        self.mois_differe = mois_differe
        self.mois_remboursement = mois_remboursement
        self.total_mois = mois_differe + mois_remboursement
        self.inflation_mensuelle = (1 + inflation_annuelle) ** (1/12) - 1
        self.n_simulations = n_simulations
        self.seuil_defaut = seuil_defaut
        self.repertoire_sortie = repertoire_sortie
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(repertoire_sortie):
            os.makedirs(repertoire_sortie)
        
        # Définir les classes d'actifs
        self.actifs = {
            'etf_actions': ClasseActif(
                nom='ETF d\'Actions',
                rendement_annuel=0.07,
                volatilite_annuelle=0.15,
                rendement_revenu_annuel=0.02,
                taux_imposition_revenu=0.25,
                frais_annuels=0.001,
                liquidite=0.95
            ),
            'obligations_etat': ClasseActif(
                nom='Obligations d\'État',
                rendement_annuel=0.025,
                volatilite_annuelle=0.05,
                rendement_revenu_annuel=0.025,
                taux_imposition_revenu=0.25,
                frais_annuels=0.002,
                liquidite=0.9
            ),
            'immobilier': ClasseActif(
                nom='Immobilier',
                rendement_annuel=0.07,
                volatilite_annuelle=0.10,
                rendement_revenu_annuel=0.04,
                taux_imposition_revenu=0.35,
                frais_annuels=0.01,
                liquidite=0.3,
                taux_vacance=0.1,  # 10% de chance de vacance par mois
                cout_entretien=0.01  # 1% de coût d'entretien annuel
            ),
            'crypto': ClasseActif(
                nom='Cryptomonnaie',
                rendement_annuel=0.15,
                volatilite_annuelle=0.60,
                rendement_revenu_annuel=0.0,
                taux_imposition_revenu=0.25,
                frais_annuels=0.005,
                liquidite=0.7
            ),
            'portefeuille_mixte': ClasseActif(
                nom='Portefeuille Mixte',
                rendement_annuel=0.055,
                volatilite_annuelle=0.10,
                rendement_revenu_annuel=0.025,
                taux_imposition_revenu=0.25,
                frais_annuels=0.003,
                liquidite=0.93
            )
        }
        
        # Calculer les paramètres du prêt
        self.montant_differe = self._calculer_montant_differe()
        self.paiement_mensuel = self._calculer_paiement_mensuel()
        
        # Suivre le solde du prêt restant au fil du temps
        self.solde_pret = np.zeros(self.total_mois + 1)
        self.solde_pret[0] = self.montant_pret
        for t in range(1, self.mois_differe + 1):
            self.solde_pret[t] = self.solde_pret[t-1] * (1 + self.taux_interet_mensuel)
        
        for t in range(self.mois_differe + 1, self.total_mois + 1):
            paiement_interet = self.solde_pret[t-1] * self.taux_interet_mensuel
            paiement_principal = self.paiement_mensuel - paiement_interet
            self.solde_pret[t] = max(0, self.solde_pret[t-1] - paiement_principal)
        
    def _calculer_montant_differe(self) -> float:
        """Calculer le montant du prêt après la période de différé."""
        return self.montant_pret * (1 + self.taux_interet_mensuel) ** self.mois_differe
    
    def _calculer_paiement_mensuel(self) -> float:
        """Calculer le paiement mensuel fixe pour la période de remboursement."""
        r = self.taux_interet_mensuel
        n = self.mois_remboursement
        return self.montant_differe * (r * (1 + r)**n) / ((1 + r)**n - 1)
    
    def simuler_portefeuille(self, actif: ClasseActif) -> np.ndarray:
        """Simuler l'évolution du portefeuille pour une classe d'actif donnée."""
        np.random.seed(42)  # Pour la reproductibilité
        
        # Initialiser les tableaux
        valeurs_portefeuille = np.zeros((self.n_simulations, self.total_mois + 1))
        valeurs_portefeuille[:, 0] = self.montant_pret
        
        # Suivre les impôts cumulés payés
        impots_payes = np.zeros((self.n_simulations, self.total_mois + 1))
        
        # Suivre les frais de transaction
        frais_transaction = np.zeros((self.n_simulations, self.total_mois + 1))
        
        for t in range(self.total_mois):
            # Générer des rendements aléatoires
            rendements = np.random.normal(
                actif.rendement_mensuel,
                actif.volatilite_mensuelle,
                self.n_simulations
            )
            
            # Calculer le revenu (avant impôt)
            rendement_revenu_mensuel = actif.rendement_revenu_annuel / 12
            
            # Pour l'immobilier, appliquer le taux de vacance
            if actif.nom == 'Immobilier':
                # Générer des événements de vacance (10% de chance par mois)
                evenements_vacance = np.random.random(self.n_simulations) < actif.taux_vacance
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
            
            # Appliquer le remboursement du prêt après la période de différé
            if t >= self.mois_differe:
                # Calculer les contraintes de liquidité
                fonds_disponibles = valeurs_portefeuille[:, t + 1] * actif.liquidite
                
                # Vérifier si nous avons suffisamment de fonds liquides pour le paiement
                deficit = np.maximum(0, self.paiement_mensuel - fonds_disponibles)
                
                # S'il y a un déficit, nous devons liquider plus d'actifs avec une pénalité
                if actif.nom != 'Immobilier':  # L'immobilier utilise les revenus locatifs
                    penalite_liquidation = 0.02  # Pénalité de 2% pour liquidation forcée
                    frais_transaction[:, t+1] = deficit * penalite_liquidation
                    valeurs_portefeuille[:, t + 1] -= (self.paiement_mensuel + frais_transaction[:, t+1])
                else:
                    # Pour l'immobilier, utiliser d'abord les revenus locatifs, puis liquider si nécessaire
                    valeurs_portefeuille[:, t + 1] -= self.paiement_mensuel
                
                # Vérifier les conditions de défaut
                default_mask = valeurs_portefeuille[:, t + 1] < (self.solde_pret[t + 1] * self.seuil_defaut)
                valeurs_portefeuille[default_mask, t + 1:] = np.nan
        
        return valeurs_portefeuille, impots_payes, frais_transaction
    
    def analyser_resultats(self, valeurs_portefeuille: np.ndarray, impots_payes: np.ndarray, 
                          frais_transaction: np.ndarray, nom_actif: str) -> Dict:
        """Analyser les résultats de la simulation."""
        # Calculer les valeurs finales
        valeurs_finales = valeurs_portefeuille[:, -1]
        
        # Calculer le taux de réussite (portefeuille > 0 à la fin)
        taux_reussite = np.mean(~np.isnan(valeurs_finales)) * 100
        
        # Calculer les statistiques sur les valeurs finales (en ignorant les NaN)
        valeurs_finales_valides = valeurs_finales[~np.isnan(valeurs_finales)]
        
        if len(valeurs_finales_valides) > 0:
            moyenne_valeur_finale = np.mean(valeurs_finales_valides)
            mediane_valeur_finale = np.median(valeurs_finales_valides)
            var_95 = np.percentile(valeurs_finales_valides, 5)
            valeur_max = np.max(valeurs_finales_valides)
            valeur_min = np.min(valeurs_finales_valides)
            
            # Calculer le profit/perte moyen
            profit_perte_moyen = moyenne_valeur_finale - self.montant_pret
            
            # Calculer le rendement annualisé moyen
            rendement_annualise_moyen = ((moyenne_valeur_finale / self.montant_pret) ** (1 / (self.total_mois / 12)) - 1) * 100
            
            # Calculer les impôts moyens payés
            impots_moyens_payes = np.mean(impots_payes[~np.isnan(valeurs_finales), -1])
            
            # Calculer les frais de transaction moyens
            frais_transaction_moyens = np.mean(frais_transaction[~np.isnan(valeurs_finales), -1])
        else:
            moyenne_valeur_finale = 0
            mediane_valeur_finale = 0
            var_95 = 0
            valeur_max = 0
            valeur_min = 0
            profit_perte_moyen = -self.montant_pret
            rendement_annualise_moyen = -100
            impots_moyens_payes = 0
            frais_transaction_moyens = 0
        
        return {
            'nom_actif': nom_actif,
            'taux_reussite': taux_reussite,
            'moyenne_valeur_finale': moyenne_valeur_finale,
            'mediane_valeur_finale': mediane_valeur_finale,
            'var_95': var_95,
            'valeur_max': valeur_max,
            'valeur_min': valeur_min,
            'profit_perte_moyen': profit_perte_moyen,
            'rendement_annualise_moyen': rendement_annualise_moyen,
            'impots_moyens_payes': impots_moyens_payes,
            'frais_transaction_moyens': frais_transaction_moyens
        }
    
    def executer_simulation(self):
        """Exécuter la simulation pour toutes les classes d'actifs et analyser les résultats."""
        resultats_liste = []
        
        for cle_actif, actif in self.actifs.items():
            print(f"\nSimulation de {actif.nom}...")
            valeurs_portefeuille, impots_payes, frais_transaction = self.simuler_portefeuille(actif)
            resultat = self.analyser_resultats(valeurs_portefeuille, impots_payes, frais_transaction, actif.nom)
            resultats_liste.append(resultat)
            
            # Tracer les chemins de simulation
            self.tracer_simulation(valeurs_portefeuille, actif.nom)
        
        # Créer un DataFrame de résultats
        resultats_df = pd.DataFrame(resultats_liste)
        resultats_df.set_index('nom_actif', inplace=True)
        
        # Afficher le tableau récapitulatif
        print("\nRésumé des résultats:")
        print(resultats_df[['taux_reussite', 'moyenne_valeur_finale', 'rendement_annualise_moyen']].round(2))
        
        # Enregistrer les résultats dans un fichier CSV
        resultats_df.to_csv(f"{self.repertoire_sortie}/resume_resultats.csv")
        
        # Tracer la comparaison des classes d'actifs
        self.tracer_comparaison_actifs(resultats_df)
        
        # Effectuer une analyse de sensibilité
        self.analyse_sensibilite_etf_actions()
        self.analyse_sensibilite_immobilier()
        
        return resultats_df
    
    def tracer_simulation(self, valeurs_portefeuille: np.ndarray, nom_actif: str):
        """Tracer les chemins de simulation et la distribution des valeurs finales."""
        plt.figure(figsize=(15, 10))
        
        # Tracer les chemins de simulation
        plt.subplot(2, 1, 1)
        x = np.arange(self.total_mois + 1)
        
        # Tracer le solde du prêt
        plt.plot(x, self.solde_pret, 'r-', linewidth=2, label='Solde du Prêt')
        
        # Tracer les chemins de simulation (seulement 100 pour la clarté)
        n_chemins = 100
        for i in range(min(n_chemins, self.n_simulations)):
            plt.plot(x, valeurs_portefeuille[i], alpha=0.1)
        
        # Ajouter une ligne verticale pour la fin de la période de différé
        plt.axvline(x=self.mois_differe, color='r', linestyle='--', 
                   label='Fin de la Période de Différé')
        
        plt.title(f'Simulation {nom_actif} - Chemins de Simulation')
        plt.xlabel('Mois')
        plt.ylabel('Valeur du Portefeuille (€)')
        plt.legend()
        
        # Tracer la distribution des valeurs finales
        plt.subplot(2, 1, 2)
        valeurs_finales = valeurs_portefeuille[:, -1]
        valeurs_valides = valeurs_finales[~np.isnan(valeurs_finales)]
        
        if len(valeurs_valides) > 0:
            sns.histplot(valeurs_valides, kde=True)
            plt.axvline(x=self.montant_pret, color='r', linestyle='--', 
                       label='Montant Initial du Prêt')
            
            # Ajouter VaR à 5%
            var_95 = np.percentile(valeurs_valides, 5)
            plt.axvline(x=var_95, color='orange', linestyle='--', 
                       label='VaR 5%')
            
            plt.title(f'Distribution des Valeurs Finales du Portefeuille - {nom_actif}')
            plt.xlabel('Valeur Finale du Portefeuille (€)')
            plt.ylabel('Fréquence')
            plt.legend()
        else:
            plt.text(0.5, 0.5, "Pas de simulations réussies", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{self.repertoire_sortie}/{nom_actif.replace(' ', '_')}_simulation.png")
        plt.close()
    
    def tracer_comparaison_actifs(self, resultats_df: pd.DataFrame):
        """Tracer une comparaison des différentes classes d'actifs."""
        plt.figure(figsize=(15, 10))
        
        # Tracer les taux de réussite
        plt.subplot(2, 1, 1)
        sns.barplot(x=resultats_df.index, y='taux_reussite', data=resultats_df)
        plt.title('Taux de Réussite par Classe d\'Actif')
        plt.xlabel('Classe d\'Actif')
        plt.ylabel('Taux de Réussite (%)')
        plt.xticks(rotation=45)
        
        # Tracer les valeurs finales moyennes
        plt.subplot(2, 1, 2)
        sns.barplot(x=resultats_df.index, y='moyenne_valeur_finale', data=resultats_df)
        plt.title('Valeur Finale Moyenne par Classe d\'Actif')
        plt.xlabel('Classe d\'Actif')
        plt.ylabel('Valeur Finale Moyenne (€)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.repertoire_sortie}/comparaison_actifs.png")
        plt.close()
    
    def analyse_sensibilite_etf_actions(self, plage_rendement=np.arange(0.02, 0.12, 0.01)):
        """Effectuer une analyse de sensibilité sur le rendement annuel des ETF d'actions."""
        taux_reussite = []
        valeurs_finales_moyennes = []
        
        for rendement in plage_rendement:
            # Créer un actif ETF d'actions avec un rendement modifié
            actif_modifie = ClasseActif(
                nom='ETF d\'Actions',
                rendement_annuel=rendement,
                volatilite_annuelle=0.15,
                rendement_revenu_annuel=0.02,
                taux_imposition_revenu=0.25,
                frais_annuels=0.001,
                liquidite=0.95
            )
            
            # Simuler le portefeuille
            valeurs_portefeuille, _, _ = self.simuler_portefeuille(actif_modifie)
            
            # Calculer les statistiques
            valeurs_finales = valeurs_portefeuille[:, -1]
            taux_reussite.append(np.mean(~np.isnan(valeurs_finales)) * 100)
            
            valeurs_finales_valides = valeurs_finales[~np.isnan(valeurs_finales)]
            if len(valeurs_finales_valides) > 0:
                valeurs_finales_moyennes.append(np.mean(valeurs_finales_valides))
            else:
                valeurs_finales_moyennes.append(0)
        
        # Tracer les résultats
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(plage_rendement * 100, taux_reussite, 'o-')
        plt.title('Sensibilité du Taux de Réussite au Rendement Annuel')
        plt.xlabel('Rendement Annuel (%)')
        plt.ylabel('Taux de Réussite (%)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(plage_rendement * 100, valeurs_finales_moyennes, 'o-')
        plt.title('Sensibilité de la Valeur Finale Moyenne au Rendement Annuel')
        plt.xlabel('Rendement Annuel (%)')
        plt.ylabel('Valeur Finale Moyenne (€)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.repertoire_sortie}/sensibilite_etf_actions_rendement_annuel.png")
        plt.close()
    
    def analyse_sensibilite_immobilier(self, plage_taux_vacance=np.arange(0, 0.3, 0.02)):
        """Effectuer une analyse de sensibilité sur le taux de vacance de l'immobilier."""
        taux_reussite = []
        valeurs_finales_moyennes = []
        
        for taux_vacance in plage_taux_vacance:
            # Créer un actif immobilier avec un taux de vacance modifié
            actif_modifie = ClasseActif(
                nom='Immobilier',
                rendement_annuel=0.07,
                volatilite_annuelle=0.10,
                rendement_revenu_annuel=0.04,
                taux_imposition_revenu=0.35,
                frais_annuels=0.01,
                liquidite=0.3,
                taux_vacance=taux_vacance,
                cout_entretien=0.01
            )
            
            # Simuler le portefeuille
            valeurs_portefeuille, _, _ = self.simuler_portefeuille(actif_modifie)
            
            # Calculer les statistiques
            valeurs_finales = valeurs_portefeuille[:, -1]
            taux_reussite.append(np.mean(~np.isnan(valeurs_finales)) * 100)
            
            valeurs_finales_valides = valeurs_finales[~np.isnan(valeurs_finales)]
            if len(valeurs_finales_valides) > 0:
                valeurs_finales_moyennes.append(np.mean(valeurs_finales_valides))
            else:
                valeurs_finales_moyennes.append(0)
        
        # Tracer les résultats
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(plage_taux_vacance * 100, taux_reussite, 'o-')
        plt.title('Sensibilité du Taux de Réussite au Taux de Vacance')
        plt.xlabel('Taux de Vacance (%)')
        plt.ylabel('Taux de Réussite (%)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(plage_taux_vacance * 100, valeurs_finales_moyennes, 'o-')
        plt.title('Sensibilité de la Valeur Finale Moyenne au Taux de Vacance')
        plt.xlabel('Taux de Vacance (%)')
        plt.ylabel('Valeur Finale Moyenne (€)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.repertoire_sortie}/sensibilite_immobilier_taux_vacance.png")
        plt.close()


if __name__ == "__main__":
    # Initialiser et exécuter la simulation
    sim = SimulationInvestissement()
    print(f"\nMontant initial du prêt : {sim.montant_pret:,.2f} €")
    print(f"Montant différé après {sim.mois_differe} mois : {sim.montant_differe:,.2f} €")
    print(f"Paiement mensuel pendant la période de remboursement : {sim.paiement_mensuel:,.2f} €")
    
    resultats = sim.executer_simulation()
