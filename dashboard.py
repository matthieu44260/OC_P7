# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:53:02 2023

@author: matth
"""

import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import json
import requests
import plotly.graph_objects as go

shap.initjs()

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title = "Dashboard-implémentez un modèle de scoring",
    page_icon = "✅",
    layout = "wide"
)

st.title(':blue[Demande de prêt]')


# Affichage de la jauge de prédiction
def jauge(value):
    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = value,
    mode = "gauge+number+delta",
    title = {'text': "Speed"},
    delta = {'reference': 60},
    gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, 50], 'color': "lightgray"},
                 {'range': [50, 60], 'color': "gray"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 60}}))
    st.plotly_chart(fig)

# Affichage du nuage de points
def nuage_pts(content_1, content_2, feat_1, feat_2):
    plt.scatter(content_1['group_0'], content_2['group_0'], marker = '.', color='blue', label='Clients sans risque')
    plt.scatter(content_1['group_1'], content_2['group_1'], marker = '.', color='red', label='Clients à risque')
    plt.scatter(content_1['client'], content_2['client'], marker = 'x', color = 'yellow', label = f'Client {id_client}')
    plt.legend(loc = (-0.5,0.8))
    plt.xlabel(feat_1)
    plt.ylabel(feat_2)

# Affichage de la distribution d'une feature
def afficher_distributions(content):
    sns.histplot(content['group_0'], color='blue', label='Clients sans risque', kde = True)
    sns.histplot(content['group_1'], color='red', label='clients à risque', kde = True)
    plt.scatter(content['client'],np.max(content['group_0']), color = 'yellow', marker = 'x', s = 100, label = f'Client {id_client}')
    plt.legend()
 

# Premier appel de l'API pour récupérer la liste des identifiants
API_url = "http://127.0.0.1:5000/credit"
response = requests.get(API_url)
if response.status_code == 200: 
    liste_id = response.json()
else:
    'Erreur lors de la requête à l\'API'


# Fonctionnalités de la sidebar
with st.sidebar:
    st.markdown("**Tester l'éligibilité**")
    id_client = st.selectbox('Identifiant client : ', options = sorted(liste_id))
    predict_btn = st.checkbox('Evaluation')              
    st.markdown("""---""")
    locale = st.checkbox("Afficher l'influence des caractéristiques du client")
    moy_clients = st.checkbox('Afficher la moyenne des clients')
    info_client = st.checkbox('Informations du client')
    globale = st.checkbox("Afficher l'influence gobale des caractéristiques")
    afficher_nuage = st.checkbox('Afficher le nuage de points des clients')
    distribution_feat = st.checkbox('Distribution des variables')
    st.markdown("""---""")
    afficher_descriptions = st.checkbox('Descriptions des caractéristiques')


# Affichage de la prédiction
if predict_btn:
    API_url = f"http://127.0.0.1:5000/credit/{id_client}"
    response = requests.get(API_url)
    if response.status_code == 200:
        resultats = response.json() 
        proba = resultats['proba']
        if proba >= 60:  # Seuil à 60%
            texte = f':green[Client sans risque à {proba}%]'
        elif proba > 50:
            texte = f':red[Client à risque à {proba}%]'
        else :
            texte = f':red[Client à risque à {100-proba}%]'
        st.markdown("<span style='font-size: 40px;'>{}</span>".format(texte), unsafe_allow_html=True)
        jauge(proba)
    else:
        'Erreur'

fig_1, fig_2 = st.columns(2)

# Affichage des données du client
if info_client:
    API_url = f"http://127.0.0.1:5000/credit/{id_client}/data"
    response = requests.get(API_url)
    with fig_1:
        st.markdown("**Informations du client :**")
        resultats = response.json()
        data_client = pd.read_json(resultats['data'])
        data_client

# Affichage de la feature importance globale        
if globale :
    API_url = "http://127.0.0.1:5000/credit/globale"
    response = requests.get(API_url)
    with fig_2:
        content = json.loads(response.content)
        shap_val_glob_0 = content['shap_values_0']
        shap_val_glob_1 = content['shap_values_1']
        liste_features = content['liste_features']
        shap_globales = np.array([shap_val_glob_0, shap_val_glob_1])
        st.markdown("**Influence globale des caractéristiques**")
        fig = shap.summary_plot(shap_globales, features = liste_features, plot_type='bar')
        st.pyplot(fig)


fig_3, fig_4 = st.columns(2)

# Affichage de la feature importance locale
if locale :
    API_url = f"http://127.0.0.1:5000/credit/locale/{id_client}"
    response = requests.get(API_url)
    response.status_code
    with fig_3:
        res = json.loads(response.content)
        shap_val_local = res['shap_val']
        base_value = res['shap_base']
        feat_values = res['shap_data']
        feat_names = res['feature_names']
        explan = shap.Explanation(np.reshape(np.array(shap_val_local, dtype='float'), (1, -1)),
                                   base_value,
                                   data=np.reshape(np.array(feat_values, dtype='float'), (1, -1)),
                                   feature_names=feat_names)
        st.markdown('**Influence des caractéristiques du client**')
        fig = shap.waterfall_plot(explan[0])
        st.pyplot(fig)      

# Affichage de la feature importance sur la moyenne des clients sans risque
if moy_clients:
    API_url = "http://127.0.0.1:5000/credit/moyenne"
    response = requests.get(API_url)
    with fig_4:
        content = json.loads(response.content)
        shap_val_glob_0 = content['shap_values_0']
        shap_val_glob_1 = content['shap_values_1']
        liste_features = content['liste_features']
        shap_globales = np.array([shap_val_glob_0, shap_val_glob_1])
        st.markdown('**Moyenne des clients sans risque**')
        fig = shap.summary_plot(shap_globales, features = liste_features, plot_type='bar')
        st.pyplot(fig)

# Affichage des explications des features
if afficher_descriptions:
    API_url = "http://127.0.0.1:5000/credit/descriptions"
    response = requests.get(API_url)
    liste_features = response.json()
    feature = st.sidebar.selectbox('Sélectionner la variable', options = sorted(liste_features))
    if feature :
        API_url = f"http://127.0.0.1:5000/credit/descriptions/{feature}"
        response = requests.get(API_url)
        texte = response.json()
        st.sidebar.markdown(texte)

fig_5, fig_6 = st.columns(2)

# Affiche le graaphique bi-varié entre 2 features
if afficher_nuage:
    with fig_5:
        API_url = "http://127.0.0.1:5000/credit/liste_feature"
        response = requests.get(API_url)
        noms_col = response.json()
        feat_1 = st.selectbox('Première variable :', options = noms_col)
        API_url = f"http://127.0.0.1:5000/credit/nuage/{feat_1}/{id_client}"
        response = requests.get(API_url)
        content_1 = json.loads(response.content)
                
        feat_2 = st.selectbox('Deuxième variable :', options = noms_col)
        API_url = f"http://127.0.0.1:5000/credit/nuage/{feat_2}/{id_client}"
        response = requests.get(API_url)
        content_2 = json.loads(response.content)
        
    with fig_6:
        st.markdown('**Clients selon le risque**')
        fig6, ax = plt.subplots() 
        nuage_pts(content_1, content_2, feat_1, feat_2)
        fig6
    
fig_7, fig_8 = st.columns(2)

if distribution_feat:
    API_url = "http://127.0.0.1:5000/credit/liste_feature"
    response = requests.get(API_url)
    noms_col = response.json()
    with fig_7:
        feature = st.selectbox('Variable :', options = noms_col)
        API_url = f"http://127.0.0.1:5000/credit/nuage/{feature}/{id_client}"
        response = requests.get(API_url)
        content = json.loads(response.content)
        donnee_client = content['client']
        st.markdown(f'**Valeur de la variable pour le client : {donnee_client}**')
    with fig_8:
        st.markdown(f'**Distribution de la variable {feature}**')
        fig8, ax = plt.subplots()
        afficher_distributions(content)
        fig8
 