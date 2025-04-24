"""
Module de visualisations pour le tableau de bord d'investissement.
Contient des fonctions pour créer des graphiques élégants et lisibles.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick

# Configuration générale des visualisations
def configurer_style():
    """Configure un style élégant et moderne pour les visualisations."""
    # Palette de couleurs personnalisée
    colors = ["#2c3e50", "#e74c3c", "#3498db", "#2ecc71", "#f39c12", 
              "#9b59b6", "#1abc9c", "#34495e", "#d35400", "#c0392b"]
    
    # Configuration de seaborn
    sns.set(style="whitegrid", font_scale=1.1)
    sns.set_palette(colors)
    
    # Configuration de matplotlib
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#dddddd'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Format des nombres
    plt.rcParams['axes.formatter.use_locale'] = True
    
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
    table.set_fontsize(7)  # Réduire davantage la taille de la police
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
                          rotation=45, ha='right', format_y='eur', grid=True, 
                          annotate=True, colors=None):
    """
    Crée un graphique à barres élégant.
    
    Args:
        ax: Axes matplotlib
        data: DataFrame des données
        x: Colonne pour l'axe x
        y: Colonne pour l'axe y
        hue: Colonne pour la couleur (optionnel)
        titre: Titre du graphique
        xlabel: Label de l'axe x
        ylabel: Label de l'axe y
        rotation: Rotation des labels de l'axe x
        ha: Alignement horizontal des labels de l'axe x
        format_y: Format des valeurs de l'axe y ('eur', 'pct', 'none')
        grid: Afficher la grille
        annotate: Annoter les barres avec les valeurs
        colors: Liste de couleurs
    """
    if hue is None:
        if colors is None:
            bars = sns.barplot(x=x, y=y, data=data, ax=ax)
        else:
            bars = sns.barplot(x=x, y=y, data=data, ax=ax, palette=colors)
    else:
        if colors is None:
            bars = sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax)
        else:
            bars = sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax, palette=colors)
    
    # Configurer le titre et les labels
    ax.set_title(titre, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Rotation des labels de l'axe x
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha=ha)
    
    # Format de l'axe y
    if format_y == 'eur':
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
    elif format_y == 'pct':
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Grille
    ax.grid(grid, axis='y', alpha=0.3)
    
    # Annotations
    if annotate:
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            if not np.isnan(height):
                if format_y == 'eur':
                    text = f"{height:,.0f} €"
                elif format_y == 'pct':
                    text = f"{height:.1%}"
                else:
                    text = f"{height:.2f}"
                
                ax.annotate(text,
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='bottom', fontsize=9,
                           rotation=0)
    
    return bars

def creer_graphique_ligne(ax, data_dict, x, y_keys, labels, titre="", xlabel="", ylabel="",
                         colors=None, markers=None, styles=None, grid=True, legend=True,
                         format_y='eur'):
    """
    Crée un graphique en ligne élégant pour plusieurs séries.
    
    Args:
        ax: Axes matplotlib
        data_dict: Dictionnaire de DataFrames ou arrays
        x: Valeurs de l'axe x (communes à toutes les séries)
        y_keys: Liste des clés pour les valeurs y dans data_dict
        labels: Liste des labels pour la légende
        titre: Titre du graphique
        xlabel: Label de l'axe x
        ylabel: Label de l'axe y
        colors: Liste de couleurs
        markers: Liste de marqueurs
        styles: Liste de styles de ligne
        grid: Afficher la grille
        legend: Afficher la légende
        format_y: Format des valeurs de l'axe y ('eur', 'pct', 'none')
    """
    if markers is None:
        markers = [None] * len(y_keys)
    
    if styles is None:
        styles = ['-'] * len(y_keys)
    
    for i, (y_key, label) in enumerate(zip(y_keys, labels)):
        color = colors[i % len(colors)] if colors is not None else None
        marker = markers[i % len(markers)]
        style = styles[i % len(styles)]
        
        ax.plot(x, data_dict[y_key], label=label, color=color, marker=marker, linestyle=style, linewidth=2)
    
    # Configurer le titre et les labels
    ax.set_title(titre, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Format de l'axe y
    if format_y == 'eur':
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
    elif format_y == 'pct':
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Grille
    ax.grid(grid, alpha=0.3)
    
    # Légende
    if legend:
        ax.legend(loc='best')
    
    return ax

def creer_graphique_distribution(ax, data_dict, keys, labels, titre="", xlabel="", ylabel="",
                                colors=None, fill=True, alpha=0.3, grid=True, legend=True,
                                references=None, format_x='eur'):
    """
    Crée un graphique de distribution (KDE) élégant pour plusieurs séries.
    
    Args:
        ax: Axes matplotlib
        data_dict: Dictionnaire de DataFrames ou arrays
        keys: Liste des clés pour les valeurs dans data_dict
        labels: Liste des labels pour la légende
        titre: Titre du graphique
        xlabel: Label de l'axe x
        ylabel: Label de l'axe y
        colors: Liste de couleurs
        fill: Remplir sous la courbe
        alpha: Transparence du remplissage
        grid: Afficher la grille
        legend: Afficher la légende
        references: Liste de tuples (valeur, label, couleur, style) pour les lignes de référence
        format_x: Format des valeurs de l'axe x ('eur', 'pct', 'none')
    """
    for i, (key, label) in enumerate(zip(keys, labels)):
        color = colors[i % len(colors)] if colors is not None else None
        sns.kdeplot(data_dict[key], ax=ax, label=label, color=color, fill=fill, alpha=alpha)
    
    # Configurer le titre et les labels
    ax.set_title(titre, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Format de l'axe x
    if format_x == 'eur':
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
    elif format_x == 'pct':
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Grille
    ax.grid(grid, alpha=0.3)
    
    # Lignes de référence
    if references is not None:
        for val, label, color, style in references:
            ax.axvline(x=val, color=color, linestyle=style, label=label)
    
    # Légende
    if legend:
        ax.legend(loc='best')
    
    return ax

def creer_graphique_scatter(ax, data, x, y, hue=None, size=None, titre="", xlabel="", ylabel="",
                           colors=None, grid=True, legend=True, annotate=True, format_x='pct', 
                           format_y='pct'):
    """
    Crée un graphique de dispersion élégant.
    
    Args:
        ax: Axes matplotlib
        data: DataFrame des données
        x: Colonne pour l'axe x
        y: Colonne pour l'axe y
        hue: Colonne pour la couleur (optionnel)
        size: Colonne pour la taille des points (optionnel)
        titre: Titre du graphique
        xlabel: Label de l'axe x
        ylabel: Label de l'axe y
        colors: Liste de couleurs
        grid: Afficher la grille
        legend: Afficher la légende
        annotate: Annoter les points avec les index
        format_x: Format des valeurs de l'axe x ('eur', 'pct', 'none')
        format_y: Format des valeurs de l'axe y ('eur', 'pct', 'none')
    """
    scatter_kws = {'s': 100, 'alpha': 0.7}
    
    if hue is None and size is None:
        scatter = sns.scatterplot(x=x, y=y, data=data, ax=ax, **scatter_kws)
    elif hue is not None and size is None:
        scatter = sns.scatterplot(x=x, y=y, hue=hue, data=data, ax=ax, palette=colors, **scatter_kws)
    elif hue is None and size is not None:
        scatter = sns.scatterplot(x=x, y=y, size=size, data=data, ax=ax, **scatter_kws)
    else:
        scatter = sns.scatterplot(x=x, y=y, hue=hue, size=size, data=data, ax=ax, palette=colors, **scatter_kws)
    
    # Configurer le titre et les labels
    ax.set_title(titre, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Format des axes
    if format_x == 'eur':
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
    elif format_x == 'pct':
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    if format_y == 'eur':
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
    elif format_y == 'pct':
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Grille
    ax.grid(grid, alpha=0.3)
    
    # Annotations
    if annotate:
        for i, txt in enumerate(data.index):
            ax.annotate(txt, (data[x].iloc[i], data[y].iloc[i]), 
                       fontsize=9, ha='left', va='bottom',
                       xytext=(5, 5), textcoords='offset points')
    
    # Légende
    if legend and (hue is not None or size is not None):
        ax.legend(loc='best')
    
    return scatter

def creer_heatmap(ax, data, titre="", cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5,
                 cbar=True, xticklabels=True, yticklabels=True):
    """
    Crée une heatmap élégante.
    
    Args:
        ax: Axes matplotlib
        data: DataFrame des données
        titre: Titre de la heatmap
        cmap: Palette de couleurs
        annot: Annoter les cellules avec les valeurs
        fmt: Format des annotations
        linewidths: Largeur des lignes entre les cellules
        cbar: Afficher la barre de couleurs
        xticklabels: Labels de l'axe x (True, False, ou liste)
        yticklabels: Labels de l'axe y (True, False, ou liste)
    """
    heatmap = sns.heatmap(data, ax=ax, cmap=cmap, annot=annot, fmt=fmt, 
                         linewidths=linewidths, cbar=cbar, 
                         xticklabels=xticklabels, yticklabels=yticklabels)
    
    # Configurer le titre
    ax.set_title(titre, fontsize=14, pad=15)
    
    return heatmap

def creer_boxplot(ax, data, x=None, y=None, hue=None, titre="", xlabel="", ylabel="",
                 colors=None, grid=True, format_y='eur'):
    """
    Crée un boxplot élégant.
    
    Args:
        ax: Axes matplotlib
        data: DataFrame des données
        x: Colonne pour l'axe x
        y: Colonne pour l'axe y
        hue: Colonne pour la couleur (optionnel)
        titre: Titre du graphique
        xlabel: Label de l'axe x
        ylabel: Label de l'axe y
        colors: Liste de couleurs
        grid: Afficher la grille
        format_y: Format des valeurs de l'axe y ('eur', 'pct', 'none')
    """
    if hue is None:
        if colors is None:
            boxplot = sns.boxplot(x=x, y=y, data=data, ax=ax)
        else:
            boxplot = sns.boxplot(x=x, y=y, data=data, ax=ax, palette=colors)
    else:
        if colors is None:
            boxplot = sns.boxplot(x=x, y=y, hue=hue, data=data, ax=ax)
        else:
            boxplot = sns.boxplot(x=x, y=y, hue=hue, data=data, ax=ax, palette=colors)
    
    # Configurer le titre et les labels
    ax.set_title(titre, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Format de l'axe y
    if format_y == 'eur':
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f} €")
    elif format_y == 'pct':
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Grille
    ax.grid(grid, axis='y', alpha=0.3)
    
    return boxplot

def creer_texte_explicatif(ax, texte, titre=None, couleur_fond="#f8f9fa", couleur_bordure="#dddddd"):
    """
    Crée un encadré de texte explicatif élégant.
    
    Args:
        ax: Axes matplotlib
        texte: Texte à afficher
        titre: Titre de l'encadré (optionnel)
        couleur_fond: Couleur de fond de l'encadré
        couleur_bordure: Couleur de la bordure de l'encadré
    """
    ax.axis('off')
    
    if titre:
        texte = f"{titre}\n\n{texte}"
    
    ax.text(0.5, 0.5, texte, ha='center', va='center', fontsize=11,
           transform=ax.transAxes, wrap=True,
           bbox=dict(facecolor=couleur_fond, edgecolor=couleur_bordure, 
                    boxstyle='round,pad=1', alpha=0.9))
    
    return ax
