import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df_art_essai = pd.read_csv('df_dummies_arts_essaie_ML_def.csv')

# Sélection des colonnes numériques
X_art_essai = df_art_essai.select_dtypes(include=['number'])

# Normalisation des données
scaler_art_essai = StandardScaler()
X_scaled_art_essai = scaler_art_essai.fit_transform(X_art_essai)

# Réduction de la dimensionnalité avec PCA après avoir essayé plusieurs components je trouve que 40 me donne les meilleurs resultats
pca_art_essai = PCA(n_components=446)
X_pca_art_essai = pca_art_essai.fit_transform(X_scaled_art_essai)

# Modèle de Nearest Neighbors
modelNN_art_essai = NearestNeighbors(n_neighbors=5, algorithm='auto')
modelNN_art_essai.fit(X_pca_art_essai)


# Recherche du film par mot clé
# titre_1 = input('Saisissez un mot clé: ').lower()
# titre_2 = df_art_essai[df_art_essai['primaryTitle'].str.lower().str.contains(titre_1)]

# if not titre_2.empty:
#     # Afficher les résultats de la recherche
#     dict_titre = dict(titre_2['primaryTitle'])
#     print(f'{dict_titre}')
    
#     # Sélection du film souhaité
#     var_1 = int(input("Insérez le numéro du film souhaité: "))
    
#     if var_1 in df_art_essai.index:
#         # Utiliser les données transformées par PCA pour trouver l'index
#         distances, indices = modelNN_art_essai.kneighbors([X_pca_art_essai[df_art_essai.index.get_loc(var_1)]])
        
#         print(f"Films similaires à '{df_art_essai['primaryTitle'].iloc[var_1]}':")
#         for idx in indices[0]:
#             print(df_art_essai['primaryTitle'].iloc[idx])
#     else:
#         print(f"L'index {var_1} n'est pas valide.")
# else:
#     print(f"Aucun film trouvé avec le mot clé '{titre_1}'.")







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