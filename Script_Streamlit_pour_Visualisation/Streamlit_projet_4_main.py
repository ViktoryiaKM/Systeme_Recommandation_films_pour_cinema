# Ce script Streamlit permet de recommander des films en fonction d’un film sélectionné par l’utilisateur, selon 3 univers différents :

# - Succès intemporels (blockbusters)
# - Famille / enfant
# - Arts et essai

# Il utilise des scripts de Machine Learning personnalisés pour chaque catégorie, et affiche les affiches + résumés des films recommandés, avec un design soigné en CSS.

#pip install streamlit
#pip install --upgrade deepl
#pip install -U scikit-learn

#  Import des bibliothèques
import streamlit as st
import pandas as pd
import deepl

# Import des modules personnalisés
import script_enfant
import script_art_essai
import script_blockbuster
import global_variable as gv

# Chargement des datasets
df_film_enfant = pd.read_csv("df_titres_image_enfants.csv")
df_film_blockbuster = pd.read_csv("df_titres_blockbusters.csv")
df_film_art_et_essaie = pd.read_csv("df_titres_image_arts_essai.csv")

# Fusion des trois catégories de films (on ignore la première colonne)
df_film = pd.concat([
    df_film_art_et_essaie.iloc[:,1:],
    df_film_blockbuster.iloc[:,1:],
    df_film_enfant.iloc[:,1:]])

# Lien vers les images d'affiche
site_image="https://image.tmdb.org/t/p/w780"

# Clé d’API pour la traduction Deepl
auth_key = "28c7bfdf-5e26-4497-ba77-e223074bddf2:fx"
translator = deepl.Translator(auth_key)

# Personnalisation de l'interface (CSS)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');   /* Importer une police de style vintage */

    h2#aca79716{
        color: #D2B48C; !important                                                                    /* Couleur beige foncée */;
        text-align : center; !important;
        font-family: 'Playfair Display', serif; !important                                            /* Police vintage */;
        font-size: 34px;  !important                                                                  /* Taille du texte */ ;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background: linear-gradient(to bottom, #000000, #434343);
        color: #D2B48C; !important;
        text-align : center; !important;
        font-family: 'Playfair Display', serif; !important                                            /* Police vintage */
    }
    .st-emotion-cache-13ln4jf {
        max-width: none !important;                                                                   /* Remove max-width limitation */
    }
    .film-description {
        background-color: #333333;
        padding: 15px;
        border-radius: 10px;
        text-align: justify;
        color: white;
        height: 100px;                                                                                /* Limiter la hauteur pour éviter les débordements */
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .film-description:hover {
        height: auto;                                                                                 /* Déplier la description au survol */
        overflow: visible;
        white-space: normal;
    }
    .stTextInput > div > div > input {
        color: #D2B48C;
        background-color: #333333;
        border-radius: 5px;
        padding: 10px;
    }
    .stSelectbox > div > div > select {
        color: white;
        background-color: #333333;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold !important
    }
    .stButton > button {
        background-color: #D2B48C;
        color: #333333;
        border: none;
        border-radius: 8px;
        padding: 15px 20px;
        margin-top: 10px;
        font-weight: bold !important
    }
    .st-emotion-cache-183lzff.exotz4b0 {
        font-size: 2.5vh;
        font-family: 'Playfair Display', serif; !important
        font-weight: bold !important
    }
    button.st-emotion-cache-7ym5gk.ef3psqc12 {
        width: 11vw;
        margin-top: 5vh;
    }
    .stButton > button:hover {
        background-color: ##D2B48C;
        font-weight: bold !important; 
    }
    .st-emotion-cache-y4bq5x {
        display: block; !important
    }
    .st-emotion-cache-j6qv4b.e1nzilvr5>p {
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Affichage de l’image de couverture et du titre principal

st.image("cinema_creuse_ajuste.jpg", use_column_width=True)
st.header(
    "Explorez les films avec notre système de recommandation intelligent"
)

st.text("Selectionner un theme selon vos préférences:")

# Boutons de sélection par thème
butcol1, butcol2, butcol3 = st.columns(3)

with butcol1:
    if st.button("Succes intemporels"):
        gv.df_active=df_film_blockbuster["French_title"]
        gv.fonction_ml="blockbuster"

with butcol2:
    if st.button("Famille/enfant"):
        gv.df_active=df_film_enfant["French_title"]
        gv.fonction_ml="enfant"

with butcol3:
    if st.button("Arts et essaies"):
        gv.df_active=df_film_art_et_essaie["French_title"]
        gv.fonction_ml="art_et_essaie"

# Sélecteur de film: Parametrage barre de recherche
film_select= st.selectbox("recherche film",gv.df_active,index=None,placeholder="Sélectionnez votre film.")

# Affichage des recommandations
if film_select!=None:
    if gv.fonction_ml=="enfant":
        film1,film2,film3,film4=script_enfant.film_enfant(film_select)
    elif gv.fonction_ml=="blockbuster":
        film1,film2,film3,film4=script_blockbuster.film_blockbuster(film_select)
    elif gv.fonction_ml=="art_et_essaie":
        film1,film2,film3,film4=script_art_essai.film_art_essai(film_select)
    st.text("Nos recommendations pour les films similaires:")

    filmcol1, filmcol2, filmcol3, filmcol4 = st.columns(4)

    # Fonction pour afficher chaque film recommandé avec affiche et description:

    def afficher_film(film_title):
        backdrop_path = df_film[df_film["French_title"] == film_title]["backdrop_path"].values[0]
        if backdrop_path == "Affiche absente":
            st.image("IMG_defaut.jpg", use_column_width=True)
        else:
            st.image(site_image + backdrop_path)
        st.markdown(f"**{film_title}**")
        overview = df_film[df_film["French_title"] == film_title]["overview"].values[0]
        if overview == "Resume absent":
            st.write("Synopsis absent")
        else:
            st.markdown(f"<div class='film-description'>{translator.translate_text(overview, target_lang='FR')}</div>", unsafe_allow_html=True)
    
     # Affichage des 4 films recommandés:

    with filmcol1:
        afficher_film(film1)
    with filmcol2:
        afficher_film(film2)
    with filmcol3:
        afficher_film(film3)
    with filmcol4:
        afficher_film(film4)
        