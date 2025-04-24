"""
Définition des classes d'actifs enrichies pour la simulation d'investissement.
Ce module contient des classes d'actifs plus diversifiées et réalistes pour mars 2025.
"""

def get_actifs_enrichis():
    """
    Retourne un dictionnaire des classes d'actifs enrichies avec leurs caractéristiques.
    Inclut des actifs traditionnels et alternatifs adaptés à différents profils de risque.
    """
    actifs = {
        # Actifs originaux (avec paramètres légèrement ajustés pour mars 2025)
        'ETF d\'Actions': {
            'nom': 'ETF d\'Actions',
            'rendement_annuel': 0.07,
            'volatilite_annuelle': 0.16,  # Légèrement augmentée pour refléter la volatilité actuelle
            'rendement_revenu_annuel': 0.018,  # Dividendes légèrement réduits
            'taux_imposition_revenu': 0.30,  # PFU à 30%
            'frais_annuels': 0.0015,  # ETF à faibles coûts
            'liquidite': 0.95,
            'description': 'ETF diversifiés suivant les indices boursiers mondiaux (MSCI World, S&P 500)',
            'profil_risque': 'Modéré à élevé',
            'horizon_recommande': '7+ ans',
            'couleur': '#1f77b4'
        },
        
        'Obligations d\'État': {
            'nom': 'Obligations d\'État',
            'rendement_annuel': 0.03,  # Taux plus élevés en 2025
            'volatilite_annuelle': 0.06,  # Volatilité accrue due aux fluctuations des taux
            'rendement_revenu_annuel': 0.03,  # Coupons
            'taux_imposition_revenu': 0.30,  # PFU à 30%
            'frais_annuels': 0.002,
            'liquidite': 0.9,
            'description': 'Obligations souveraines européennes (France, Allemagne) à moyen et long terme',
            'profil_risque': 'Faible à modéré',
            'horizon_recommande': '3+ ans',
            'couleur': '#ff7f0e'
        },
        
        'Immobilier': {
            'nom': 'Immobilier',
            'rendement_annuel': 0.05,  # Rendement total (locatif + appréciation)
            'volatilite_annuelle': 0.10,
            'rendement_revenu_annuel': 0.035,  # Rendement locatif net
            'taux_imposition_revenu': 0.37,  # Fiscalité immobilière (IR + prélèvements sociaux)
            'frais_annuels': 0.015,  # Charges, entretien, assurance
            'liquidite': 0.3,
            'taux_vacance': 0.08,  # 8% de vacance locative
            'cout_entretien': 0.012,  # 1.2% de coûts d'entretien
            'description': 'Investissement dans un bien immobilier locatif en zone urbaine dynamique',
            'profil_risque': 'Modéré',
            'horizon_recommande': '10+ ans',
            'couleur': '#2ca02c'
        },
        
        # Portefeuilles mixtes (conservés et ajustés)
        'Portefeuille Mixte (60/40)': {
            'nom': 'Portefeuille Mixte (60/40)',
            'rendement_annuel': 0.07*0.6 + 0.03*0.4,  # 60% actions, 40% obligations
            'volatilite_annuelle': 0.16*0.6 + 0.06*0.4,
            'rendement_revenu_annuel': 0.018*0.6 + 0.03*0.4,
            'taux_imposition_revenu': 0.30,
            'frais_annuels': 0.0015*0.6 + 0.002*0.4,
            'liquidite': 0.95*0.6 + 0.9*0.4,
            'description': 'Allocation classique avec 60% en ETF d\'actions et 40% en obligations',
            'profil_risque': 'Modéré',
            'horizon_recommande': '5+ ans',
            'couleur': '#d62728'
        },
        
        'Portefeuille Mixte (80/20)': {
            'nom': 'Portefeuille Mixte (80/20)',
            'rendement_annuel': 0.07*0.8 + 0.03*0.2,  # 80% actions, 20% obligations
            'volatilite_annuelle': 0.16*0.8 + 0.06*0.2,
            'rendement_revenu_annuel': 0.018*0.8 + 0.03*0.2,
            'taux_imposition_revenu': 0.30,
            'frais_annuels': 0.0015*0.8 + 0.002*0.2,
            'liquidite': 0.95*0.8 + 0.9*0.2,
            'description': 'Allocation dynamique avec 80% en ETF d\'actions et 20% en obligations',
            'profil_risque': 'Modéré à élevé',
            'horizon_recommande': '7+ ans',
            'couleur': '#9467bd'
        },
        
        'Portefeuille Mixte (40/60)': {
            'nom': 'Portefeuille Mixte (40/60)',
            'rendement_annuel': 0.07*0.4 + 0.03*0.6,  # 40% actions, 60% obligations
            'volatilite_annuelle': 0.16*0.4 + 0.06*0.6,
            'rendement_revenu_annuel': 0.018*0.4 + 0.03*0.6,
            'taux_imposition_revenu': 0.30,
            'frais_annuels': 0.0015*0.4 + 0.002*0.6,
            'liquidite': 0.95*0.4 + 0.9*0.6,
            'description': 'Allocation prudente avec 40% en ETF d\'actions et 60% en obligations',
            'profil_risque': 'Faible à modéré',
            'horizon_recommande': '3+ ans',
            'couleur': '#8c564b'
        },
        
        'Portefeuille Équilibré': {
            'nom': 'Portefeuille Équilibré',
            'rendement_annuel': 0.07*0.4 + 0.03*0.3 + 0.05*0.3,  # 40% actions, 30% obligations, 30% immobilier
            'volatilite_annuelle': 0.16*0.4 + 0.06*0.3 + 0.10*0.3,
            'rendement_revenu_annuel': 0.018*0.4 + 0.03*0.3 + 0.035*0.3,
            'taux_imposition_revenu': 0.30*0.7 + 0.37*0.3,  # Moyenne pondérée
            'frais_annuels': 0.0015*0.4 + 0.002*0.3 + 0.015*0.3,
            'liquidite': 0.95*0.4 + 0.9*0.3 + 0.3*0.3,
            'taux_vacance': 0.08*0.3,  # Pour la partie immobilière
            'cout_entretien': 0.012*0.3,  # Pour la partie immobilière
            'description': 'Portefeuille diversifié avec 40% actions, 30% obligations et 30% immobilier',
            'profil_risque': 'Modéré',
            'horizon_recommande': '7+ ans',
            'couleur': '#e377c2'
        },
        
        # Nouveaux actifs pertinents pour 2025
        'ETF d\'Actions à Faible Volatilité': {
            'nom': 'ETF d\'Actions à Faible Volatilité',
            'rendement_annuel': 0.055,  # Rendement légèrement inférieur aux ETF classiques
            'volatilite_annuelle': 0.11,  # Volatilité réduite
            'rendement_revenu_annuel': 0.022,  # Dividendes légèrement supérieurs
            'taux_imposition_revenu': 0.30,
            'frais_annuels': 0.0025,  # Frais légèrement plus élevés
            'liquidite': 0.95,
            'description': 'ETF sélectionnant des actions à faible volatilité historique',
            'profil_risque': 'Modéré',
            'horizon_recommande': '5+ ans',
            'couleur': '#7f7f7f'
        },
        
        'ETF Obligations d\'Entreprises': {
            'nom': 'ETF Obligations d\'Entreprises',
            'rendement_annuel': 0.04,  # Rendement supérieur aux obligations d'État
            'volatilite_annuelle': 0.08,  # Volatilité plus élevée
            'rendement_revenu_annuel': 0.04,  # Coupons
            'taux_imposition_revenu': 0.30,
            'frais_annuels': 0.0025,
            'liquidite': 0.85,
            'description': 'ETF d\'obligations d\'entreprises Investment Grade (notation BBB- ou supérieure)',
            'profil_risque': 'Modéré',
            'horizon_recommande': '3+ ans',
            'couleur': '#bcbd22'
        },
        
        'ETF Obligations High Yield': {
            'nom': 'ETF Obligations High Yield',
            'rendement_annuel': 0.055,  # Rendement plus élevé
            'volatilite_annuelle': 0.12,  # Volatilité plus élevée
            'rendement_revenu_annuel': 0.055,  # Coupons élevés
            'taux_imposition_revenu': 0.30,
            'frais_annuels': 0.004,
            'liquidite': 0.8,
            'description': 'ETF d\'obligations d\'entreprises à haut rendement (notation inférieure à BBB-)',
            'profil_risque': 'Modéré à élevé',
            'horizon_recommande': '5+ ans',
            'couleur': '#17becf'
        },
        
        'SCPI': {
            'nom': 'SCPI',
            'rendement_annuel': 0.045,  # Rendement total
            'volatilite_annuelle': 0.06,  # Volatilité limitée
            'rendement_revenu_annuel': 0.042,  # Rendement distribué
            'taux_imposition_revenu': 0.30,  # PFU ou barème IR
            'frais_annuels': 0.012,  # Frais de gestion
            'liquidite': 0.5,  # Liquidité moyenne
            'description': 'Sociétés Civiles de Placement Immobilier investissant dans l\'immobilier commercial',
            'profil_risque': 'Faible à modéré',
            'horizon_recommande': '8+ ans',
            'couleur': '#aec7e8'
        },
        
        'ETF Smart Beta': {
            'nom': 'ETF Smart Beta',
            'rendement_annuel': 0.065,  # Entre ETF classiques et actifs
            'volatilite_annuelle': 0.15,  # Proche des ETF classiques
            'rendement_revenu_annuel': 0.02,
            'taux_imposition_revenu': 0.30,
            'frais_annuels': 0.003,  # Frais plus élevés
            'liquidite': 0.93,
            'description': 'ETF utilisant des stratégies factorielles (valeur, qualité, momentum)',
            'profil_risque': 'Modéré à élevé',
            'horizon_recommande': '7+ ans',
            'couleur': '#ffbb78'
        },
        
        'Livret A': {
            'nom': 'Livret A',
            'rendement_annuel': 0.02,  # Taux du Livret A en 2025
            'volatilite_annuelle': 0.0,  # Pas de volatilité
            'rendement_revenu_annuel': 0.02,
            'taux_imposition_revenu': 0.0,  # Exonéré d'impôts
            'frais_annuels': 0.0,  # Pas de frais
            'liquidite': 1.0,  # Parfaitement liquide
            'description': 'Placement réglementé sans risque, exonéré d\'impôts (plafond 22 950€)',
            'profil_risque': 'Très faible',
            'horizon_recommande': 'Court terme',
            'couleur': '#98df8a'
        },
        
        'Assurance-Vie Fonds Euro': {
            'nom': 'Assurance-Vie Fonds Euro',
            'rendement_annuel': 0.025,  # Rendement moyen des fonds euros en 2025
            'volatilite_annuelle': 0.005,  # Très faible volatilité
            'rendement_revenu_annuel': 0.025,
            'taux_imposition_revenu': 0.175,  # Fiscalité avantageuse après 8 ans
            'frais_annuels': 0.007,  # Frais de gestion
            'liquidite': 0.8,  # Bonne liquidité
            'description': 'Fonds euros d\'assurance-vie avec capital garanti et fiscalité avantageuse',
            'profil_risque': 'Faible',
            'horizon_recommande': '4+ ans',
            'couleur': '#ff9896'
        },
        
        'PEA ETF Européens': {
            'nom': 'PEA ETF Européens',
            'rendement_annuel': 0.065,  # Légèrement inférieur aux ETF mondiaux
            'volatilite_annuelle': 0.17,  # Légèrement plus volatile
            'rendement_revenu_annuel': 0.025,  # Dividendes réinvestis
            'taux_imposition_revenu': 0.175,  # Fiscalité PEA après 5 ans
            'frais_annuels': 0.002,
            'liquidite': 0.9,
            'description': 'ETF européens dans une enveloppe PEA fiscalement avantageuse',
            'profil_risque': 'Modéré à élevé',
            'horizon_recommande': '5+ ans',
            'couleur': '#c5b0d5'
        }
    }
    
    return actifs
