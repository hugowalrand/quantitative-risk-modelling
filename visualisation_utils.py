"""
Fonctions utilitaires pour les visualisations du tableau de bord d'investissement
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from matplotlib.table import Table
import os
from datetime import datetime

def configurer_style():
    """Configure le style global pour les visualisations."""
    # Définir le style global
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Palette de couleurs pour différentes classes d'actifs
    colors = {
        'ETF d\'Actions': '#1f77b4',  # Bleu
        'Obligations d\'État': '#ff7f0e',  # Orange
        'Immobilier': '#2ca02c',  # Vert
        'Portefeuille Mixte (60/40)': '#d62728',  # Rouge
        'Portefeuille Mixte (80/20)': '#9467bd',  # Violet
        'Portefeuille Mixte (40/60)': '#8c564b',  # Marron
        'Portefeuille Équilibré': '#e377c2',  # Rose
        'ETF d\'Actions à Faible Volatilité': '#7f7f7f',  # Gris
        'ETF Obligations d\'Entreprises': '#bcbd22',  # Jaune-vert
        'ETF Obligations High Yield': '#17becf',  # Bleu clair
        'SCPI': '#aec7e8',  # Bleu pâle
        'ETF Smart Beta': '#ffbb78',  # Orange pâle
        'PEA ETF Européens': '#98df8a'  # Vert pâle
    }
    
    return colors

def creer_tableau_metriques(ax, resultats_df, metriques, titre="Tableau des métriques clés"):
    """
    Crée un tableau élégant des métriques clés.
    
    Args:
        ax: Axes matplotlib
        resultats_df: DataFrame des résultats
        metriques: Liste de tuples (nom_colonne, format, titre_colonne, description)
        titre: Titre du tableau
    """
    ax.axis('off')
    
    # Préparer les données pour le tableau
    data = []
    for index in resultats_df.index:
        # Tronquer les noms trop longs pour améliorer la lisibilité
        nom_court = index[:15] + '...' if len(index) > 15 else index
        row = [nom_court]  # Nom de l'actif
        for col, fmt, _, _ in metriques:
            if col in resultats_df.columns:
                val = resultats_df.loc[index, col]
                if fmt == 'pct':
                    cell = f"{val:.1f}%"
                elif fmt == 'eur':
                    cell = f"{val:,.0f} €"
                elif fmt == 'ratio':
                    cell = f"{val:.2f}"
                else:
                    cell = f"{val}"
                row.append(cell)
            else:
                row.append("N/A")
        data.append(row)
    
    # En-têtes de colonnes avec descriptions très concises
    headers = ['Actif']
    for _, _, titre, desc in metriques:
        header = f"{titre}\n{desc}"
        headers.append(header)
    
    # Créer le tableau
    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    
    # Formater le tableau
    table.auto_set_font_size(False)
    table.set_fontsize(7)  # Taille de police réduite pour tout faire tenir
    table.scale(1, 1.2)    # Ajuster l'échelle pour une meilleure lisibilité
    
    # Ajuster la largeur des colonnes
    table.auto_set_column_width([0] + list(range(1, len(headers))))
    
    # Styliser l'en-tête
    for j, header in enumerate(headers):
        cell = table[(0, j)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white', size=7)
        
        # Ajuster la hauteur des cellules d'en-tête pour les descriptions
        if j > 0:  # Sauf pour la première colonne
            cell.set_height(0.15)
    
    # Styliser les cellules selon les valeurs
    for i in range(len(resultats_df.index)):
        # Colorier la première colonne (nom de l'actif)
        cell = table[(i+1, 0)]
        cell.set_facecolor('#f8f9fa')
        cell.set_text_props(weight='bold', size=7)
        
        # Colorier les autres colonnes selon les valeurs
        for j, (col, fmt, _, _) in enumerate(metriques, 1):
            if col in resultats_df.columns:
                cell = table[(i+1, j)]
                cell.set_text_props(size=7)  # Taille de police pour les données
                
                # Définir la couleur selon le type de métrique
                if col == 'volatilite_annuelle' or col == 'drawdown_max':
                    # Rouge pour les valeurs élevées (risque)
                    val = resultats_df.iloc[i][col]
                    min_val = resultats_df[col].min()
                    max_val = resultats_df[col].max()
                    norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                    r = min(0.9, 0.5 + norm_val * 0.4)
                    g = max(0.5, 0.9 - norm_val * 0.4)
                    b = 0.5
                    cell.set_facecolor((r, g, b))
                
                elif col == 'ratio_sharpe' or col == 'taux_reussite' or col == 'rendement_net_annualise':
                    # Vert pour les valeurs élevées (performance)
                    val = resultats_df.iloc[i][col]
                    min_val = resultats_df[col].min()
                    max_val = resultats_df[col].max()
                    norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                    r = max(0.5, 0.9 - norm_val * 0.4)
                    g = min(0.9, 0.5 + norm_val * 0.4)
                    b = 0.5
                    cell.set_facecolor((r, g, b))
                
                elif col == 'gain_net_apres_impots' or col == 'ratio_gain_cout':
                    # Bleu pour les valeurs élevées (gain)
                    val = resultats_df.iloc[i][col]
                    min_val = resultats_df[col].min()
                    max_val = resultats_df[col].max()
                    norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                    r = 0.5
                    g = min(0.9, 0.5 + norm_val * 0.3)
                    b = min(0.9, 0.5 + norm_val * 0.4)
                    cell.set_facecolor((r, g, b))
    
    # Ajouter une légende pour les couleurs
    ax.text(0.01, 0.01, "Couleurs: Vert = Performance élevée | Rouge = Risque élevé | Bleu = Gain élevé", 
            transform=ax.transAxes, fontsize=8, ha='left')
    
    # Ajouter le titre au-dessus du tableau
    ax.set_title(titre, fontsize=14, pad=10, loc='center')
    
    return table

def creer_graphique_barres(ax, data, x, y, hue=None, titre="", xlabel="", ylabel="", 
                          rotation=0, format_y='numeric', colors=None, annotate=True):
    """
    Crée un graphique à barres élégant.
    
    Args:
        ax: Axes matplotlib
        data: DataFrame des données
        x, y: Colonnes à utiliser pour x et y
        hue: Colonne à utiliser pour le regroupement (optionnel)
        titre, xlabel, ylabel: Textes pour le titre et les axes
        rotation: Rotation des étiquettes de l'axe x
        format_y: Format des étiquettes de l'axe y ('numeric', 'eur', 'pct')
        colors: Liste de couleurs pour les barres
        annotate: Si True, annotez les barres avec les valeurs
    """
    # Créer le graphique à barres (avec correction du FutureWarning)
    if hue is None and colors is not None:
        # Créer une colonne 'hue' temporaire identique à 'x' pour utiliser les couleurs
        data = data.copy()
        data['_hue'] = data[x]
        bars = sns.barplot(x=x, y=y, hue='_hue', data=data, ax=ax, palette=colors, legend=False)
    else:
        bars = sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax)
    
    # Configurer le titre et les étiquettes
    ax.set_title(titre, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Configurer le format de l'axe y
    if format_y == 'eur':
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
    elif format_y == 'pct':
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Rotation des étiquettes de l'axe x
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right' if rotation > 0 else 'center')
    
    # Grille et esthétique
    ax.grid(True, axis='y', alpha=0.3)
    
    # Annoter les barres avec les valeurs
    if annotate:
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            if format_y == 'eur':
                annotation = f"{height:,.0f} €"
            elif format_y == 'pct':
                annotation = f"{height:.1f}%"
            else:
                annotation = f"{height:.1f}"
            
            # Ajuster la position de l'annotation
            ax.annotate(annotation,
                      (p.get_x() + p.get_width() / 2., height),
                      ha='center', va='bottom', fontsize=9,
                      xytext=(0, 5), textcoords='offset points')
    
    return bars

def creer_graphique_ligne(ax, data, x, y, hue=None, titre="", xlabel="", ylabel="", 
                         rotation=0, format_y='numeric', palette=None):
    """
    Crée un graphique linéaire élégant.
    
    Args:
        ax: Axes matplotlib
        data: DataFrame des données
        x, y: Colonnes à utiliser pour x et y
        hue: Colonne à utiliser pour le regroupement
        titre, xlabel, ylabel: Textes pour le titre et les axes
        rotation: Rotation des étiquettes de l'axe x
        format_y: Format des étiquettes de l'axe y ('numeric', 'eur', 'pct')
        palette: Palette de couleurs à utiliser
    """
    # Créer le graphique linéaire
    lines = sns.lineplot(x=x, y=y, hue=hue, data=data, ax=ax, palette=palette)
    
    # Configurer le titre et les étiquettes
    ax.set_title(titre, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Configurer le format de l'axe y
    if format_y == 'eur':
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
    elif format_y == 'pct':
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Rotation des étiquettes de l'axe x
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right' if rotation > 0 else 'center')
    
    # Grille et esthétique
    ax.grid(True, alpha=0.3)
    
    return lines
