import streamlit as st
import pandas as pd
import json
import flask
import numpy as np
import pickle
import shap
import streamlit_shap
from streamlit_shap import st_shap
import streamlit.components.v1 as components
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import requests
from PIL import Image

# Chargement de tous les objets nécéssaire : explainers, images, listes etc

liste_features_ordre = pickle.load(open('liste_features_ordre', 'rb'))
filename_model = 'RFR_opti.pkl'
model = pickle.load(open(filename_model,'rb'))
moyennes = pickle.load(open('moyennes', 'rb'))
liste_variables_impactantes = pickle.load(open('liste_variable_impact', 'rb'))
explainer = pickle.load(open('explainer_2', 'rb'))
image = Image.open('shap.png')                        
liste_features = pickle.load(open('liste_features.pkl', 'rb'))
documentation = pickle.load(open('documentation', 'rb'))

# Définition des URLs à appeler pour les requêtes
url='http://127.0.0.1:5000/api'
url_traitement='http://127.0.0.1:5000/traitement'
headers = {'Content-Type' : 'application/json'}
                        
                        

                        
st.title('Dashboard de gestion du candidat à la contraction de prêt')
st.subheader("Nous allons effectuer une prédiction d'un éventuel défaut de paiement du prêt, ainsi qu'une interprétabilité des choix effectués concernant le potentiel contractant")





# Un FileBrowser pour pouvoir déposer facilement les données client
# On demande en input un .json encodé précisemment

data = st.file_uploader("Charger le json des données du client", type = 'json')
print('upload success')


                        
if data is not None :
    
    data = data.read().decode('utf-8')
    data = json.loads(data)           # Récupèration de la donnée
                        
    response = requests.post(url, json=data, headers= headers)  #Requête API pour récupèrer la prédiction
    
    prediction = json.loads(response.text)
    response_2 = requests.post(url_traitement, json=data, headers= headers) # Requête API pour récupèrer un dataframe (une ligne)                                                                                        #transformé pour notre client
    traitement = json.loads(response_2.text)
    
    
    df_traitement = pd.DataFrame(traitement, index = [0])
    df_traitement = df_traitement[liste_features_ordre]  #On remet les colonnes dans le "bon" ordre
    
    df = df_traitement[liste_variables_impactantes]
    df = df.append(moyennes.iloc[0]).reset_index()   # Création du dataframe qui nous permettra de comparer notre individu à la moyenne des                                                        individus ayant obtenu leurs prêts
    df = df.drop(['index'], axis = 1)

    
    
    # Définition de cette comparaison puis création d'un dictionnaire contenant ses résultats, que nous afficherons sous forme tabulaire
    liste_resultats_comparaison =[]
    dictionnaire_comparaison ={}
    if df.iloc[0][0] <= df.iloc[1][0]:
        liste_resultats_comparaison.append('mauvais')
    else: 
        liste_resultats_comparaison.append('bon')
    
    if df.iloc[0][1] <= df.iloc[1][1]:
        liste_resultats_comparaison.append('mauvais')
    else: 
        liste_resultats_comparaison.append('bon')
    
    if df.iloc[0][2] <= df.iloc[1][2]:
        liste_resultats_comparaison.append('mauvais')
    else: 
        liste_resultats_comparaison.append('bon')
    
    if df.iloc[0][3] <= df.iloc[1][3]:
        liste_resultats_comparaison.append('bon')
    else: 
        liste_resultats_comparaison.append('mauvais')
    
    if df.iloc[0][4] <= df.iloc[1][4]:
        liste_resultats_comparaison.append('bon')
    else: 
        liste_resultats_comparaison.append('mauvais')
            
    if df.iloc[0][5] <= df.iloc[1][5]:
        liste_resultats_comparaison.append('mauvais')
    else: 
        liste_resultats_comparaison.append('bon')
            
    if df.iloc[0][6] <= df.iloc[1][6]:
        liste_resultats_comparaison.append('bon')
    else: 
        liste_resultats_comparaison.append('mauvais')
    
    if df.iloc[0][7] <= df.iloc[1][7]:
        liste_resultats_comparaison.append('bon')
    else: 
        liste_resultats_comparaison.append('mauvais')
    
    
    dictionnaire_comparaison = {'EXT_SOURCE_2' : liste_resultats_comparaison[0], 'EXT_SOURCE_3' : liste_resultats_comparaison[1], 'Age_Client' : liste_resultats_comparaison[2], 'DAYS_EMPLOYED' : liste_resultats_comparaison[3], 'AMT_GOODS_PRICE' : liste_resultats_comparaison[4], 'EXT_SOURCE_1' : liste_resultats_comparaison[5], 'DAYS_LAST_PHONE_CHANGE' : liste_resultats_comparaison[6], 'MOY_PREV_DAYS_DECISION' : liste_resultats_comparaison[7]}
    df_comparaison = pd.DataFrame(dictionnaire_comparaison, index =[0])

#Affichage de la documentation    
with st.sidebar:
    st.subheader('Documentation')
    st.dataframe(documentation)
    
    
    
# Partie affichage de la prédiction sur Streamlit
st.subheader('Prédiction pour notre individu')
if data is not None:
    if prediction[0] == 0:
            st.write('Bien joué vous avez le prêt')
    else :
            st.write('Désolé, vous n/avez pas le prêt')

        
# Partie interprétabilité globale du modèle, chargée sous forme .png                        
st.subheader('Interprétabilité globale du modèle')

st.image(image)

# Interprétabilité de la prédiction de l'individu, pour chacune des classes.
st.subheader('Interprétabilité locale du modèle')
if data is not None :
    
    shap_values = explainer.shap_values(df_traitement.loc[[0]])
    
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], df_traitement.loc[[0]]))
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], df_traitement.iloc[[0]]))
    



#Affichage des résultats de comparaison avec la moyenne des 0 sur le jeu d'entraînement
st.subheader('Comparaison avec la moyenne des valeurs obtenues par les individus sans défaut de paiement')
if data is not None :
    st.dataframe(df)
    st.dataframe(df_comparaison)
