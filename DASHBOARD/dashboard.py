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



def jauge(value):
    """Affichage de la jauge de prédiction"""
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

def nuage_pts(content_1, content_2, feat_1, feat_2):
    """Affichage du nuage de points"""
    plt.scatter(content_1['group_0'], content_2['group_0'], marker = '.', color='blue', label='Clients sans risque')
    plt.scatter(content_1['group_1'], content_2['group_1'], marker = '.', color='red', label='Clients à risque')
    plt.scatter(content_1['client'], content_2['client'], marker = 'x', color = 'yellow', label = f'Client {id_client}')
    plt.legend(loc = (-0.5,0.8))
    plt.xlabel(feat_1)
    plt.ylabel(feat_2)

def afficher_distributions(content):
    """Affichage de la distribution d'une feature"""
    sns.histplot(content['group_0'], color='blue', label='Clients sans risque', kde = True)
    sns.histplot(content['group_1'], color='red', label='clients à risque', kde = True)
    plt.scatter(content['client'],1, color = 'yellow', marker = 'x', s = 100, label = f'Client {id_client}')
    plt.legend()

def assembler_shap_values(res):
    shap_val_local = res['shap_val']
    base_value = res['shap_base']
    feat_values = res['shap_data']
    explan = shap.Explanation(np.array(shap_val_local, dtype='float'),
                              np.array(base_value, dtype = 'float'),
                              data = np.array(feat_values, dtype='float'),
                              feature_names = liste_features)
    return explan

# Premier appel de l'API pour récupérer la liste des identifiants
API_url = "http://127.0.0.1:5000/credit"
response = requests.get(API_url)
if response.status_code == 200: 
    resultats = response.json()
    liste_id = resultats['liste_id']
    liste_features = resultats['liste_features']
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
        proba = response.json()
        if proba >= 60:  # Seuil à 60%
            texte = f':green[Client sans risque à {proba}%]'
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
        data_client = pd.read_json(resultats)
        data_client

# Affichage de la feature importance globale        
if globale :
    API_url = "http://127.0.0.1:5000/credit/globale"
    response = requests.get(API_url)
    with fig_2:
        res = json.loads(response.content)
        explan = assembler_shap_values(res)
        st.markdown("**Influence globale des caractéristiques**")
        fig = shap.plots.bar(explan)
        st.pyplot(fig)


fig_3, fig_4 = st.columns(2)

# Affichage de la feature importance locale
if locale :
    API_url = f"http://127.0.0.1:5000/credit/locale/{id_client}"
    response = requests.get(API_url)
    with fig_3:
        res = json.loads(response.content)
        explan = assembler_shap_values(res)
        st.markdown('**Influence des caractéristiques du client**')
        fig = shap.waterfall_plot(explan[0])
        st.pyplot(fig)      

# Affichage de la feature importance sur la moyenne des clients sans risque
if moy_clients:
    API_url = "http://127.0.0.1:5000/credit/moyenne"
    response = requests.get(API_url)
    with fig_4:
        shap_values = response.json()
        st.markdown('**Moyenne des clients sans risque**')
        fig = shap.bar_plot(np.array(shap_values), feature_names = liste_features, max_display = 10)
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

# Affiche le graphique bi-varié entre 2 features
if afficher_nuage:
    with fig_5:
        feat_1 = st.selectbox('Première variable :', options = sorted(liste_features))
        API_url = f"http://127.0.0.1:5000/credit/nuage/{feat_1}/{id_client}"
        response = requests.get(API_url)
        content_1 = json.loads(response.content)
                
        feat_2 = st.selectbox('Deuxième variable :', options = sorted(liste_features))
        API_url = f"http://127.0.0.1:5000/credit/nuage/{feat_2}/{id_client}"
        response = requests.get(API_url)
        content_2 = json.loads(response.content)
        
    with fig_6:
        st.markdown('**Clients selon le risque**')
        fig6, ax = plt.subplots() 
        nuage_pts(content_1, content_2, feat_1, feat_2)
        fig6
    
fig_7, fig_8 = st.columns(2)

# Affiche la distribution d'une feature selon la target des clients
if distribution_feat:
    with fig_7:
        feature = st.selectbox('Variable :', options = sorted(liste_features))
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
 