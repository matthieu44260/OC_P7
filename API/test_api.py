# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:22:45 2023

@author: matth
"""

import requests
import pandas as pd
from joblib import load
import json
import shap


# Chargement des données
data_test = pd.read_csv('./donnees_test_essai.csv')
liste_id = data_test['SK_ID_CURR'].to_list()
liste_features = data_test.columns.tolist()

# Chargement des données du client exemple
id_client = 221167
data_client = data_test.loc[data_test['SK_ID_CURR'] == id_client]

# Chargement du modèle
model = load('model.joblib')


def test_liste_identifiants():
    """Teste si la fonction renvoie bien la liste des identifiants clients
    et des features"""
    url = "https://oc-p7-app-1bfc88840522.herokuapp.com/credit"
    response = requests.get(url)
    resultats = response.json()
    assert resultats['liste_id'] == liste_id
    assert resultats['liste_features'] == liste_features


def test_credit_client():
    """Teste si la fonction renvoie la bonne probabilité"""
    url = "https://oc-p7-app-1bfc88840522.herokuapp.com/credit/221167"
    response = requests.get(url)
    proba_api = response.json()

    proba_model = model.predict_proba(data_client)
    proba_model = round(proba_model[0][0]*100)

    assert proba_api == proba_model

   
def test_valeurs_shap():
    """Teste si la fonction renvoie les valeurs shap du client"""
    url = "https://oc-p7-app-1bfc88840522.herokuapp.com/credit/locale/221167"
    response = requests.get(url)
    res = json.loads(response.content)
    shap_val_api = res['shap_val']
    
    explainer = shap.TreeExplainer(model)
    shap_values_test = explainer(data_client)
    
    assert shap_val_api == shap_values_test.values.tolist()

    
if __name__ == "__main__":
    test_liste_identifiants()
    test_credit_client()
    test_valeurs_shap()
    print("Everything passed")