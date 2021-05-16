# librairies générales
import pandas as pd
import numpy as np

# librairie affichage
import matplotlib.pyplot as plt
import seaborn as sns

# fonction utilisée pour l'affichage de la matrice de confusion
def plot_confusion_matrix(confusionmatrix, classes):
    sns.set(color_codes=True)
    plt.figure(1, figsize=(8, 5))
 
    plt.title("Matrice de confusion")
 
    sns.set(font_scale=1.4)
    ax = sns.heatmap(confusionmatrix, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'},fmt='g')
 
    ax.set_xticklabels(classes,rotation=45)
    ax.set_yticklabels(classes,rotation=0)
 
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.show()
    plt.close()  