# Simulation d'Investissement

Ce projet Python simule différents scénarios d'investissement avec un prêt de 200 000 € sur une période de 10 ans (18 mois de différé + 8,5 ans de remboursement).

## Fonctionnalités

- Modélise plusieurs classes d'actifs :
  - ETF d'actions
  - Obligations d'État
  - Immobilier locatif
  - Cryptomonnaie
  - Portefeuille mixte (60% actions, 40% obligations)
- Intègre des caractéristiques réalistes du marché :
  - Frais de transaction
  - Implications fiscales
  - Rendements des revenus
  - Ajustement de l'inflation
  - Conditions de défaut
- Simulation de Monte Carlo avec paramètres configurables
- Analyse complète et visualisation des résultats

## Installation

1. Clonez ce dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Exécutez la simulation :
```bash
python src/investment_sim.py
```

La simulation produira :
- Paramètres initiaux du prêt
- Métriques de performance pour chaque classe d'actifs
- Visualisation des trajectoires de simulation et des distributions de valeurs finales

## Paramètres

- Montant du prêt : 200 000 €
- Taux d'intérêt annuel : 0,99%
- Période de différé : 18 mois (1,5 ans)
- Période de remboursement : 102 mois (8,5 ans)
- Taux d'inflation annuel : 2%
- Itérations de simulation par défaut : 10 000
