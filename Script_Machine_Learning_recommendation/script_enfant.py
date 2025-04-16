import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df_dummies_enfants_for_ML_def= pd.read_csv("df_dummies_enfants_for_ML_def.csv")


# Selection des colonnes numériques
X_enfants = df_dummies_enfants_for_ML_def.select_dtypes(include=['number'])


# Normalisation des données
scaler_enfants = StandardScaler()
X_scaled_enfants = scaler_enfants.fit_transform(X_enfants)

# reduction de la dimensionnalité avec PCA
pca = PCA(n_components=132)
pca.fit(X_scaled_enfants)
X_pca_enfants = pca.transform(X_scaled_enfants)

# Modèle de Nearest Neighbors
modelNN_enfant = NearestNeighbors(n_neighbors=5, algorithm='auto')
modelNN_enfant.fit(X_pca_enfants)


def film_enfant(titre):
    if titre in df_dummies_enfants_for_ML_def['French_title'].values:
        index_titre = df_dummies_enfants_for_ML_def[df_dummies_enfants_for_ML_def['French_title'] == titre].index[0]
        distances, indices = modelNN_enfant.kneighbors([X_pca_enfants[df_dummies_enfants_for_ML_def.index.get_loc(index_titre)]])
        list_vide = []
        for idx in indices[0]:
            list_vide.append(df_dummies_enfants_for_ML_def['French_title'].iloc[idx])
        return list_vide[-4:]
    else:
        print("Ce film n'est pas dans notre DataFrame.")
        return []
