# Quantitative Risk Modelling


## 🧠 Objectif du projet

Ce projet vise à modéliser et analyser les risques financiers associés à différents types d'investissements dans un cadre réaliste.  
Il applique des techniques de **simulation de Monte Carlo**, **modélisation stochastique** et **analyse de portefeuille** pour évaluer la performance d'actifs sur une période longue avec effets de levier.



## 🛠️ Fonctionnalités principales

- **Modélisation de plusieurs classes d'actifs** :
  - ETF d'actions mondiales
  - Obligations d'État
  - Immobilier locatif
  - Cryptomonnaies
  - Portefeuille diversifié (60% actions, 40% obligations)
- **Prise en compte de paramètres réalistes** :
  - Frais de transaction
  - Impacts fiscaux
  - Revenus locatifs / dividendes
  - Ajustements liés à l'inflation
  - Probabilités de défaut
- **Simulation avancée** :
  - 10 000 itérations de Monte Carlo
  - Analyse statistique détaillée (moyenne, VaR, percentiles)
  - Visualisations dynamiques des résultats (matplotlib)
- **Génération automatique de rapports PDF personnalisés**

---

## 📈 Technologies utilisées

- Python 3.10
- NumPy, Pandas, Matplotlib
- Scikit-learn (analyses statistiques complémentaires)
- FPDF (génération de rapports automatiques)

---

## 🚀 Comment utiliser le projet

1. **Clonez le dépôt** :
```bash
git clone https://github.com/hugowalrand/quantitative-risk-modelling.git
```
2. **Installez les dépendances** :
```bash
pip install -r requirements.txt
```
3. **Lancez une simulation** :
```bash
python investment_sim.py
```

> Le script génèrera des analyses de performance, des visualisations de risques, ainsi qu’un rapport d’investissement complet en PDF.

---

## ⚙️ Paramètres par défaut

- Montant du prêt : **200 000 €**
- Taux d’intérêt annuel : **0,99 %**
- Période de différé : **18 mois**
- Durée totale : **10 ans**
- Inflation annuelle : **2 %**
- Itérations Monte Carlo : **10 000**

---

## 📑 Structure du projet

| Fichier | Description |
|--------|-------------|
| `investment_sim.py` | Simulation principale de l'évolution des investissements |
| `generate_detailed_report.py` | Création automatique d'un rapport PDF complet |
| `visualisations.py` | Outils de visualisation dynamique |
| `analyse_approfondie.py` | Analyse quantitative détaillée |
| `fiscalite.py` | Modèle fiscal simplifié |
| `actifs_enrichis.py` | Données enrichies sur les classes d'actifs |

---

## 🎯 Perspective d'amélioration

- Intégration de modèles de séries temporelles (ARIMA, GARCH)
- Évaluation du risque systémique via la corrélation croisée des classes d’actifs
- Introduction de stress testing sur scénarios macroéconomiques



> 📩 N’hésitez pas à me contacter si vous souhaitez échanger sur les méthodologies employées ou sur l'application de ce projet à d'autres domaines du risk management.
