# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:08:41 2023

@author: matth
"""

import pandas as pd
from joblib import load
import shap
from flask import Flask, jsonify


shap.initjs()

app = Flask(__name__)
app.config["DEBUG"] = True

# Chargement des fichiers
data_test = pd.read_csv('./donnees_test_essai.csv')
data_train = pd.read_csv('./donnees_train_essai.csv')
noms_col = data_test.columns

# Séparation des groupes
group_0 = data_train[data_train['TARGET'] == 0]
group_1 = data_train[data_train['TARGET'] == 1]

# Chargement du modèle
model = load('model.joblib')

# Chargement des valeurs Shap
explainer = shap.TreeExplainer(model)
shap_values_test = explainer(data_test)

# Chargement des définitions des features
df_expli = pd.read_csv('./Columns_description.csv', encoding = 'latin-1')
   
# Noms des features
liste_features = data_test.columns.tolist()

# Liste des identifiants clients
liste_id = data_test['SK_ID_CURR'].to_list()


@app.route('/credit', methods = ['GET'])
def liste_identifiants():
    """Renvoie la liste des identifiants clients et des features"""
    return {'liste_id': liste_id,
            'liste_features': liste_features}

@app.route('/credit/<id_client>', methods = ['GET'])
def credit_client(id_client):
    """Renvoie la probabilité que le client soit sans risque"""
    if int(id_client) in liste_id :
        data_client = data_test.loc[data_test['SK_ID_CURR']== int(id_client)]
        proba = model.predict_proba(data_client) # Calcul de la probabilité d'obtenir 0
        proba_0 = round(proba[0][0]*100)  
        return jsonify(proba_0)
    else :
        return 'Identifiant inconnu'

@app.route('/credit/<id_client>/data', methods = ['GET'])
def donnees_client(id_client):
    """Renvoie les données du client"""
    data_client = data_test.loc[data_test['SK_ID_CURR']== int(id_client)]
    resultats = data_client.to_json(orient='records')
    return jsonify(resultats)

@app.route('/credit/globale', methods = ['GET'])
def globale():
    """Influence globale des features"""
    shap_val = shap_values_test.values.tolist()
    shap_base = shap_values_test.base_values.tolist()
    shap_data = shap_values_test.data.tolist()
    return {
        'shap_val': shap_val,
        'shap_base': shap_base,
        'shap_data': shap_data
            }
    
@app.route('/credit/locale/<int:id_client>', methods = ['GET'])
def valeurs_shap(id_client):
    """Valeurs shap des données du client"""
    data_client = data_test.loc[data_test['SK_ID_CURR']== int(id_client)]
    index = data_client.index
    shap_values = shap_values_test[index]
    shap_val = shap_values.values.tolist()
    shap_base = shap_values.base_values.tolist()
    shap_data = shap_values.data.tolist()
    return {
        'shap_val': shap_val,
        'shap_base': shap_base,
        'shap_data': shap_data
            }
    
@app.route('/credit/moyenne', methods = ['GET'])
def moyenne():
    """Valeurs shap du data_train_group_0 (clients sans risque)"""
    shap_val = explainer(group_0.drop('TARGET',axis=1))
    shap_val_values = shap_val.values.mean(axis=0).tolist()
    return jsonify(shap_val_values)
           
@app.route('/credit/descriptions', methods = ['GET'])
def descriptions():
    """Liste des features disponibles qui ont une définition"""
    return list(set(sorted(df_expli['Row'].tolist())))

@app.route('/credit/descriptions/<feature>', methods = ['GET'])
def textes(feature):
    """Explications des features"""
    texte = df_expli.loc[df_expli['Row'] == feature, 'Description'].tolist()
    return jsonify(texte)

@app.route('/credit/nuage/<feature>/<id_client>', methods = ['GET'])
def nuage(feature, id_client):
    """Renvoie les valeurs des features pour les graphiques"""
    data_client = data_test.loc[data_test['SK_ID_CURR']== int(id_client)]
    return {'group_0': group_0[feature].tolist(),
            'group_1': group_1[feature].tolist(),
            'client': data_client[feature].tolist()
             }

if __name__ == '__main__':
    app.run(debug=True)