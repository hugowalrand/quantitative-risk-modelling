"""
Script simple pour afficher le tableau de bord d'investissement
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Chemin vers l'image du tableau de bord
chemin_image = "../output/tableau_bord_elegant.png"

# Vérifier que l'image existe
if not os.path.exists(chemin_image):
    print(f"Erreur: L'image {chemin_image} n'existe pas.")
    exit(1)

# Charger et afficher l'image
img = mpimg.imread(chemin_image)
plt.figure(figsize=(16, 20))  # Taille ajustée pour l'affichage
plt.imshow(img)
plt.axis('off')  # Masquer les axes
plt.tight_layout()
plt.show()

print(f"Affichage de l'image: {os.path.abspath(chemin_image)}")
print("Fermez la fenêtre pour quitter.")
