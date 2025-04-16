# Objectif :Créer un système de recommandation de films similaires à partir d’un film donné, en utilisant la méthode des k plus proches voisins (KNN) après normalisation et réduction de dimensionnalité avec PCA.

# Importer les bibliotheques :
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Chargement et préparation des données

## Charger le dataset
df_art_essai = pd.read_csv('df_dummies_arts_essaie_ML_def.csv')

# 2. Prétraitement des données numériques

## Sélection des colonnes numériques
X_art_essai = df_art_essai.select_dtypes(include=['number'])

## Normalisation des données
scaler_art_essai = StandardScaler()
X_scaled_art_essai = scaler_art_essai.fit_transform(X_art_essai)

## Réduction de la dimensionnalité avec PCA après avoir essayé plusieurs components je trouve que 40 me donne les meilleurs resultats
pca_art_essai = PCA(n_components=446)
X_pca_art_essai = pca_art_essai.fit_transform(X_scaled_art_essai)

# 3. Modèle de Nearest Neighbors (k=5): pour identifier les films les plus proches dans l’espace des composantes principales

## Initialisation du modèle KNN, l'attribus 'algorithm auto' pour qu'il selectionne le meilleur algo automatiquement
modelNN_art_essai = NearestNeighbors(n_neighbors=5, algorithm='auto')
modelNN_art_essai.fit(X_pca_art_essai)


# 4. Fonction de recommandation qui Retourne les 4 films les plus similaires à un film donné.

def film_art_essai(titre):
    if titre in df_art_essai['French_title'].values:
        index_titre = df_art_essai[df_art_essai['French_title'] == titre].index[0]
        distances, indices = modelNN_art_essai.kneighbors([X_pca_art_essai[df_art_essai.index.get_loc(index_titre)]])
        list_vide = []
        for idx in indices[0]:
            list_vide.append(df_art_essai['French_title'].iloc[idx])
        return list_vide[-4:]
    else:
        print("Ce film n'est pas dans notre DataFrame.")
        return []