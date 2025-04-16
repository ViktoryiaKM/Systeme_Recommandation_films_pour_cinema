# Objectif :Créer un système de recommandation de films similaires à partir d’un film donné, en utilisant la méthode des k plus proches voisins (KNN) après normalisation et réduction de dimensionnalité avec PCA.

# Importer les bibliotheques :
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Chargement et préparation des données

## Charger le dataset
df_blockbuster = pd.read_csv('df_dummies_blockbusters_ML2.csv')

## Nettoyage : renommer une colonne et corriger un titre (pour que ça match mieux par rapport aux autres DF)
df_blockbuster.rename(columns={"Title_french": "French_title"}, inplace=True)
df_blockbuster.drop(columns = 'Unnamed: 0',inplace = True)
# j'ai renommé le nom du film car il était faux et ne correspondait pas au titre original
df_blockbuster.loc[df_blockbuster['French_title'] == "Maman", 'French_title'] = "Maman j'ai raté l'avion"

# 2. Prétraitement des données numériques

## Sélection des colonnes numériques
X_blockbuster = df_blockbuster.select_dtypes(include=['number'])

## Normalisation des données
scaler_blockbuster = StandardScaler()
X_scaled_blockbuster = scaler_blockbuster.fit_transform(X_blockbuster)

## Réduction de la dimensionnalité avec PCA
pca_blockbuster = PCA(n_components=258)
X_pca_blockbuster = pca_blockbuster.fit_transform(X_scaled_blockbuster)

# 3. Modèle de Nearest Neighbors (k=5): pour identifier les films les plus proches dans l’espace des composantes principales

## Initialisation du modèle KNN, l'attribus 'algorithm auto' pour qu'il selectionne le meilleur algo automatiquement
modelNN_blockbuster = NearestNeighbors(n_neighbors=5, algorithm='auto')
modelNN_blockbuster.fit(X_pca_blockbuster)


# # Recherche du film par mot clé
# titre_1 = input('Saisissez un mot clé: ').lower()
# titre_2 = df_blockbuster[df_blockbuster['French_title'].str.lower().str.contains(titre_1)]

# if not titre_2.empty:
#     # Afficher les résultats de la recherche
#     dict_titre = {index: row['French_title'] for index, row in titre_2.iterrows()}
#     print(f'{dict_titre}')
    
#     # Sélection du film souhaité
#     var_1 = int(input("Insérez le numéro du film souhaité: "))
    
#     if var_1 in df_blockbuster.index:
#         # Utiliser les données transformées par PCA pour trouver l'index
#         distances, indices = modelNN_blockbuster.kneighbors([X_pca_blockbuster[df_blockbuster.index.get_loc(var_1)]])
        
#         print(f"Films similaires à '{df_blockbuster['French_title'].loc[var_1]}':")
#         for idx in indices[0]:
#             print(df_blockbuster['French_title'].iloc[idx])
#     else:
#         print(f"L'index {var_1} n'est pas valide.")
# else:
#     print(f"Aucun film trouvé avec le mot clé '{titre_1}'.")
    
    
 # 4. Fonction de recommandation qui Retourne les 4 films les plus similaires à un film donné.
    
def film_blockbuster(titre):
    if titre in df_blockbuster['French_title'].values:
        index_titre = df_blockbuster[df_blockbuster['French_title'] == titre].index[0]
        distances, indices = modelNN_blockbuster.kneighbors([X_pca_blockbuster[df_blockbuster.index.get_loc(index_titre)]])
        list_vide = []
        for idx in indices[0]:
            list_vide.append(df_blockbuster['French_title'].iloc[idx])
        return list_vide[-4:]
    else:
        print("Ce film n'est pas dans notre DataFrame.")
        return []