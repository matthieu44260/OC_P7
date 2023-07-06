import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import shap

shap.initjs()

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title = "Dashboard-implémentez un modèle de scoring",
    page_icon = "✅",
    layout = "wide"
)

st.title(':blue[Demande de prêt]')

# Chargement du fichier clients
@st.cache_data
def load_df_test():
    data_test = pd.read_csv('./donnees_test.csv')
    return data_test

data_test = load_df_test()
noms_col = data_test.columns

@st.cache_data
def load_df_train():
    data_train = pd.read_csv('./donnees_train_sample.csv')
    return data_train

data_train = load_df_train()

# Chargement du modèle
model = load('best_model.joblib')

# Chargement des valeurs Shap
@st.cache_data
def load_shap_values(data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data)
    return shap_values

shap_values_test = load_shap_values(data_test)
shap_values_train = load_shap_values(data_train.drop('TARGET',axis=1))


# Affichage de la feature importance globale
@st.cache_data
def load_feat_imp():
    importance = model.feature_importances_
    indices = np.argsort(importance)
    indices = indices[-10:]
    color_list =  sns.color_palette("dark", len(noms_col)) 
    fig, ax = plt.subplots()
    ax.barh(range(len(indices)), importance[indices], color= [color_list[indices[i]] for i in range(10)],
         align='center')  
    ax.set_yticks(range(len(indices)), [(noms_col[j] + ' : ' + str(round(importance[j],3))) for j in indices],
           fontweight="normal", fontsize=16) 
    for i, ticklabel in enumerate(plt.gca().get_yticklabels()):
        ticklabel.set_color(color_list[indices[i]])  
    return fig

feat_imp = load_feat_imp()

# Chargement des définitions des features
@st.cache_data
def explications():
    df_expli = pd.read_csv('./HomeCredit_columns_description.csv', encoding = 'latin-1')
    return df_expli.sort_values('Row')

df_expli = explications()


def affiche_moy_clients():
    fig4, ax = plt.subplots()
    shap.plots.bar(shap_values_train)
    fig4

# Affichage de la prédiction
def affiche_texte(score, proba):
    if score == 0:
        texte = f':green[Client sans risque à {proba}%]'
        st.markdown("<span style='font-size: 40px;'>{}</span>".format(texte), unsafe_allow_html=True)
    elif score == 1:
        texte = f':red[Client à risque à {100-proba}%]'
        st.markdown("<span style='font-size: 40px;'>{}</span>".format(texte), unsafe_allow_html=True)

group_0 = data_train[data_train['TARGET'] == 0]
group_1 = data_train[data_train['TARGET'] == 1]

# Affichage du nuage de points
def nuage_pts(feat1, feat2):
    plt.scatter(group_0[feat1], group_0[feat2], marker = '.', color='blue', label='Clients sans risque')
    plt.scatter(group_1[feat1], group_1[feat2], marker = '.', color='red', label='Clients à risque')
    plt.scatter(data_client[feat1], data_client[feat2], marker = 'x', color = 'yellow', label = f'Client {id_client}')
    plt.legend(loc = (-0.5,0.8))
    plt.xlabel(feat1)
    plt.ylabel(feat2)

# Affichage de la distribution d'une feature
def afficher_distributions(feat):
    sns.histplot(group_0[feat], color='blue', label='Clients sans risque', kde = True)
    sns.histplot(group_1[feat], color='red', label='clients à risque', kde = True)
    plt.scatter(data_client[feat], 100, color = 'yellow', marker = 'o', label = f'Client {id_client}')
    plt.legend()
    
# Noms des features
liste_features = df_expli['Row'].unique()

# Liste des identifiants clients
liste_id = data_test['SK_ID_CURR'].to_list()

# Fonctionnalités de la sidebar
with st.sidebar:
    st.markdown("**Tester l'éligibilité**")
    id_client = st.selectbox('Identifiant client : ', options = liste_id)
    predict_btn = st.checkbox('Evaluation')              
    data_client = data_test.loc[data_test['SK_ID_CURR']==id_client]
    index = data_client.index
    st.markdown("""---""")
    locale = st.checkbox("Afficher l'influence des caractéristiques du client")
    moy_clients = st.checkbox('Afficher la moyenne des clients')
    info_client = st.checkbox('Informations du client')
    globale = st.checkbox("Afficher l'influence gobale des caractéristiques")
    afficher_nuage = st.checkbox('Afficher le nuage de points des clients')
    distribution_feat = st.checkbox('Distribution des variables')
    st.markdown("""---""")
    afficher_descriptions = st.checkbox('Descriptions des caractéristiques')
    

if predict_btn:
    score = model.predict(data_client)
    proba = round(model.predict_proba(data_client)[0][0]*100)
    affiche_texte(score, proba)

fig_3, fig_4 = st.columns(2)
        
if locale :
    with fig_3:
        st.markdown('**Influence des caractéristiques du client**')
        fig3, ax = plt.subplots()
        shap.plots.bar(shap_values_test[index])
        fig3
        
if moy_clients:
    with fig_4:
        st.markdown('**Moyenne des clients**')
        affiche_moy_clients()

fig_1, fig_2 = st.columns(2)
  
if info_client:
    with fig_1:
        st.markdown("**Informations du client :**")
        data_client
        
if globale :
    with fig_2:
        st.markdown("**Influence globale des caractéristiques**")
        feat_imp

if afficher_descriptions:
    feature = st.sidebar.selectbox('Sélectionner la variable', options = liste_features)
    texte = df_expli.loc[df_expli['Row']==feature, 'Description'].values[0]
    st.sidebar.markdown(texte)

fig_5, fig_6 = st.columns(2)

if afficher_nuage:
    with fig_5:
        feat_1 = st.selectbox('Première variable :', options = noms_col)
        feat_2 = st.selectbox('Deuxième variable :', options = noms_col)
    with fig_6:
        st.markdown('**Clients selon le risque**')
        fig6, ax = plt.subplots() 
        nuage_pts(feat_1, feat_2)
        fig6
    
fig_7, fig_8 = st.columns(2)

if distribution_feat:
    with fig_7:
        distrib_feat = st.selectbox('Variable :', options = noms_col)
        st.markdown(f'**Valeur de la variable pour le client : {data_client[distrib_feat].values[0]}**')
    with fig_8:
        st.markdown(f'**Distribution de la variable {distrib_feat}**')
        fig8, ax = plt.subplots()
        afficher_distributions(distrib_feat)
        fig8
        
        