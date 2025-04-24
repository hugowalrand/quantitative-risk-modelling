"""
Module de fiscalité pour la simulation d'investissement.
Contient les fonctions et paramètres pour calculer l'impact fiscal sur différents types d'investissements.
"""

def get_regimes_fiscaux():
    """
    Retourne un dictionnaire des différents régimes fiscaux applicables en France en 2025.
    """
    regimes = {
        # Flat Tax (PFU) - Prélèvement Forfaitaire Unique
        'PFU': {
            'nom': 'Prélèvement Forfaitaire Unique (PFU)',
            'taux_global': 0.30,  # 12.8% IR + 17.2% prélèvements sociaux
            'description': 'Régime fiscal par défaut pour les revenus de capitaux mobiliers et plus-values mobilières',
            'applicable_a': ['dividendes', 'interets', 'plus_values_mobilieres']
        },
        
        # Impôt sur le Revenu (barème progressif) + Prélèvements sociaux
        'IR': {
            'nom': 'Impôt sur le Revenu (barème progressif)',
            'tranches': [
                (0, 0.0),
                (10_777, 0.11),
                (27_478, 0.30),
                (78_570, 0.41),
                (168_994, 0.45)
            ],
            'prelevements_sociaux': 0.172,  # 17.2% de prélèvements sociaux
            'description': 'Barème progressif de l\'impôt sur le revenu + prélèvements sociaux',
            'applicable_a': ['revenus_fonciers', 'option_dividendes', 'option_interets']
        },
        
        # Régime micro-foncier
        'MICRO_FONCIER': {
            'nom': 'Régime Micro-Foncier',
            'abattement': 0.30,  # 30% d'abattement forfaitaire
            'plafond': 15_000,  # Applicable si revenus fonciers < 15 000€
            'description': 'Régime simplifié pour les petits revenus fonciers avec abattement forfaitaire',
            'applicable_a': ['revenus_fonciers']
        },
        
        # Régime réel foncier
        'REEL_FONCIER': {
            'nom': 'Régime Réel Foncier',
            'description': 'Imposition sur le revenu net foncier (revenus - charges déductibles)',
            'applicable_a': ['revenus_fonciers']
        },
        
        # PEA (Plan d'Épargne en Actions)
        'PEA': {
            'nom': 'Plan d\'Épargne en Actions',
            'taux_apres_5_ans': 0.172,  # Seulement prélèvements sociaux après 5 ans
            'taux_avant_5_ans': 0.30,  # PFU avant 5 ans
            'plafond': 150_000,  # Plafond de versement
            'description': 'Enveloppe fiscale avantageuse pour les actions européennes',
            'applicable_a': ['actions_europeennes']
        },
        
        # Assurance-vie
        'ASSURANCE_VIE': {
            'nom': 'Assurance-Vie',
            'taux_apres_8_ans': {
                'abattement': 4_600,  # 9 200€ pour un couple
                'taux_apres_abattement': 0.172 + 0.075  # 17.2% PS + 7.5% IR
            },
            'taux_entre_4_et_8_ans': 0.172 + 0.125,  # 17.2% PS + 12.5% IR
            'taux_avant_4_ans': 0.30,  # PFU
            'description': 'Contrat d\'assurance-vie avec fiscalité avantageuse après 8 ans',
            'applicable_a': ['assurance_vie']
        },
        
        # Livret A et autres livrets réglementés
        'LIVRETS_REGLEMENTES': {
            'nom': 'Livrets Réglementés',
            'taux': 0.0,  # Exonération totale
            'plafonds': {
                'Livret A': 22_950,
                'LDDS': 12_000,
                'LEP': 10_000
            },
            'description': 'Livrets d\'épargne réglementés totalement exonérés d\'impôts',
            'applicable_a': ['livret_a', 'ldds', 'lep']
        }
    }
    
    return regimes

def calculer_impot_revenu(revenu_imposable, tranches):
    """
    Calcule l'impôt sur le revenu selon le barème progressif.
    
    Args:
        revenu_imposable: Montant du revenu imposable
        tranches: Liste de tuples (seuil, taux)
    
    Returns:
        Montant de l'impôt sur le revenu
    """
    impot = 0
    tranche_precedente = 0
    
    for seuil, taux in tranches:
        if revenu_imposable > seuil:
            impot += (min(revenu_imposable, seuil) - tranche_precedente) * taux
            tranche_precedente = seuil
        else:
            break
    
    return impot

def appliquer_fiscalite(montant, type_revenu, regime_fiscal, duree_detention=0, revenu_fiscal_reference=50_000):
    """
    Applique la fiscalité appropriée selon le type de revenu et le régime fiscal.
    
    Args:
        montant: Montant brut du revenu ou de la plus-value
        type_revenu: Type de revenu ('dividendes', 'interets', 'plus_values_mobilieres', etc.)
        regime_fiscal: Régime fiscal applicable
        duree_detention: Durée de détention en années (pour les régimes avec avantages liés à la durée)
        revenu_fiscal_reference: Revenu fiscal de référence (pour les calculs IR)
    
    Returns:
        Montant net après fiscalité
    """
    regimes = get_regimes_fiscaux()
    
    # PFU (Flat Tax)
    if regime_fiscal == 'PFU' and type_revenu in regimes['PFU']['applicable_a']:
        return montant * (1 - regimes['PFU']['taux_global'])
    
    # Impôt sur le Revenu (barème progressif)
    elif regime_fiscal == 'IR' and type_revenu in regimes['IR']['applicable_a']:
        # Simplification: on suppose que ce revenu s'ajoute au revenu fiscal de référence
        impot_ir = calculer_impot_revenu(revenu_fiscal_reference + montant, regimes['IR']['tranches']) - \
                  calculer_impot_revenu(revenu_fiscal_reference, regimes['IR']['tranches'])
        prelevements_sociaux = montant * regimes['IR']['prelevements_sociaux']
        return montant - impot_ir - prelevements_sociaux
    
    # Micro-foncier
    elif regime_fiscal == 'MICRO_FONCIER' and type_revenu in regimes['MICRO_FONCIER']['applicable_a']:
        if montant <= regimes['MICRO_FONCIER']['plafond']:
            revenu_imposable = montant * (1 - regimes['MICRO_FONCIER']['abattement'])
            # Calcul simplifié avec un taux moyen d'imposition
            taux_moyen_imposition = 0.30  # Estimation pour un revenu moyen
            impot = revenu_imposable * taux_moyen_imposition
            return montant - impot
        else:
            # Si au-dessus du plafond, on bascule sur le régime réel
            return appliquer_fiscalite(montant, type_revenu, 'REEL_FONCIER', duree_detention, revenu_fiscal_reference)
    
    # Régime réel foncier (simplification)
    elif regime_fiscal == 'REEL_FONCIER' and type_revenu in regimes['REEL_FONCIER']['applicable_a']:
        # On suppose des charges déductibles de 30% (simplification)
        charges_deductibles = montant * 0.30
        revenu_imposable = montant - charges_deductibles
        # Calcul simplifié avec un taux moyen d'imposition
        taux_moyen_imposition = 0.30  # Estimation pour un revenu moyen
        impot = revenu_imposable * taux_moyen_imposition
        return montant - impot
    
    # PEA
    elif regime_fiscal == 'PEA' and type_revenu in regimes['PEA']['applicable_a']:
        if duree_detention >= 5:
            return montant * (1 - regimes['PEA']['taux_apres_5_ans'])
        else:
            return montant * (1 - regimes['PEA']['taux_avant_5_ans'])
    
    # Assurance-vie
    elif regime_fiscal == 'ASSURANCE_VIE' and type_revenu in regimes['ASSURANCE_VIE']['applicable_a']:
        if duree_detention >= 8:
            # Simplification: on ne tient pas compte de l'abattement pour cet exemple
            return montant * (1 - regimes['ASSURANCE_VIE']['taux_apres_8_ans']['taux_apres_abattement'])
        elif duree_detention >= 4:
            return montant * (1 - regimes['ASSURANCE_VIE']['taux_entre_4_et_8_ans'])
        else:
            return montant * (1 - regimes['ASSURANCE_VIE']['taux_avant_4_ans'])
    
    # Livrets réglementés
    elif regime_fiscal == 'LIVRETS_REGLEMENTES' and type_revenu in regimes['LIVRETS_REGLEMENTES']['applicable_a']:
        return montant  # Exonération totale
    
    # Par défaut, on applique le PFU
    else:
        return montant * (1 - regimes['PFU']['taux_global'])

def get_fiscalite_par_actif():
    """
    Retourne la fiscalité applicable à chaque type d'actif.
    """
    fiscalite = {
        'ETF d\'Actions': {
            'regime_dividendes': 'PFU',
            'regime_plus_values': 'PFU',
            'taux_effectif_global': 0.30,
            'commentaire': 'Dividendes et plus-values soumis au PFU de 30%'
        },
        'Obligations d\'État': {
            'regime_interets': 'PFU',
            'regime_plus_values': 'PFU',
            'taux_effectif_global': 0.30,
            'commentaire': 'Intérêts et plus-values soumis au PFU de 30%'
        },
        'Immobilier': {
            'regime_revenus': 'REEL_FONCIER',  # ou 'MICRO_FONCIER' selon les revenus
            'regime_plus_values': 'IR',  # Avec abattements pour durée de détention
            'taux_effectif_global': 0.37,  # Estimation moyenne
            'commentaire': 'Revenus fonciers imposés au régime réel ou micro-foncier, plus-values avec abattements pour durée de détention'
        },
        'SCPI': {
            'regime_revenus': 'IR',
            'regime_plus_values': 'PFU',
            'taux_effectif_global': 0.33,  # Estimation moyenne
            'commentaire': 'Revenus imposés comme des revenus fonciers, plus-values comme des plus-values mobilières'
        },
        'PEA ETF Européens': {
            'regime_dividendes': 'PEA',
            'regime_plus_values': 'PEA',
            'taux_effectif_global': 0.172,  # Après 5 ans
            'commentaire': 'Exonération d\'impôt sur le revenu après 5 ans, uniquement prélèvements sociaux (17,2%)'
        },
        'Assurance-Vie Fonds Euro': {
            'regime': 'ASSURANCE_VIE',
            'taux_effectif_global': 0.247,  # Après 8 ans (17,2% PS + 7,5% IR)
            'commentaire': 'Fiscalité avantageuse après 8 ans avec abattement annuel de 4 600€ (9 200€ pour un couple)'
        },
        'Livret A': {
            'regime': 'LIVRETS_REGLEMENTES',
            'taux_effectif_global': 0.0,
            'commentaire': 'Exonération totale d\'impôts et prélèvements sociaux'
        }
    }
    
    # Compléter pour les autres actifs avec des valeurs par défaut
    actifs_par_defaut = [
        'ETF d\'Actions à Faible Volatilité', 'ETF Obligations d\'Entreprises', 
        'ETF Obligations High Yield', 'ETF Smart Beta', 'Portefeuille Mixte (60/40)', 
        'Portefeuille Mixte (80/20)', 'Portefeuille Mixte (40/60)', 'Portefeuille Équilibré'
    ]
    
    for actif in actifs_par_defaut:
        if actif not in fiscalite:
            fiscalite[actif] = {
                'regime_dividendes': 'PFU',
                'regime_plus_values': 'PFU',
                'taux_effectif_global': 0.30,
                'commentaire': 'Dividendes et plus-values soumis au PFU de 30% (régime par défaut)'
            }
    
    return fiscalite
