import flask
from flask import Flask, jsonify, request, render_template
import requests
import pandas as pd
import numpy as np
import sklearn
import json
import pickle
import http.client

http.client._MAXHEADERS = 1000

#Initialisation de l'API
app = Flask(__name__)
app.config["DEBUG"] = True # donne des messages d'erreur

#Chargement de tous les objets dont nous aurons besoin
filename = 'RFR_opti.pkl'
#filename_log = 'logreg_opti.pkl'

filename_scaler = 'min_max_scaler_full_projet7_7.pkl'

filename_encoding = 'ohe_full_projet7_7.pkl'
filename_cols_num = 'liste_float'
filename_cols_obj = 'liste_objets'
filename_imputer_object = 'imputer_object_projet7_7.pkl'
filename_imputer_float = 'imputer_float_projet7_7.pkl'
filename_float_cols = 'liste_float'
filename_object_cols = 'liste_objets'



print("Chargement du modèle...")
model = pickle.load(open(filename,'rb'))
print("Modèle chargé")

print("Chargement de l'encodage..")
encoder = pickle.load(open(filename_encoding, 'rb'))

print("Encodage chargé")

print("Chargement du scaler")
scaler = pickle.load(open(filename_scaler, 'rb'))
print("Scaler chargé")

print("Chargement des imputers")
imputer_object = pickle.load(open(filename_imputer_object, 'rb'))
imputer_float = pickle.load(open(filename_imputer_float, 'rb'))
print("Imputers chargés")
float_cols = pickle.load(open(filename_float_cols, 'rb'))
object_cols = pickle.load(open(filename_object_cols, 'rb'))
print("Listes chargées")

@app.route('/', methods=['GET'])
def home():
	return "<h1> Doit-on accorder le prêt ?</h1><p>This site is a Prototype API.</p>"

  


# END-point qui renverra un dictionnaire contenant les données traitées de l'individu, pour des raisons de praticité
@app.route('/traitement',methods=['POST'])
def traitement():
    data = request.get_json(force=True)
    df = pd.read_json(json.dumps(data), orient = 'index').T
    
    df[float_cols] = imputer_float.transform(df[float_cols])
    df[object_cols] = imputer_object.transform(df[object_cols])
    
    
    df_scaled = pd.DataFrame(scaler.transform(df[float_cols]),
                                      index = df[float_cols].index,
                                          columns = df[float_cols].columns)
    codes = encoder.transform(df[object_cols]).toarray()
    feature_names = encoder.get_feature_names()
    
    df_encoded = pd.DataFrame(codes,columns=feature_names).astype(int)
    
    df_full_stack = pd.concat([df_scaled, 
               df_encoded], axis=1)
    df_transformed_dict = df_full_stack.to_dict(orient='records')[0]
  
    
    
    return df_transformed_dict    
    
    
#END-point lié à la prédiction, on recoit un .json qui sera traité et envoyé en input pour le modèle qui prédira la classe.
@app.route('/api',methods=['POST'])
def predict2():
    data = request.get_json(force=True)   #Récupération de la donnée
    df = pd.read_json(json.dumps(data), orient = 'index').T
    
    df[float_cols] = imputer_float.transform(df[float_cols])  #Imputation des colonnes catégorielles et quantitatives par mode et médiane
    df[object_cols] = imputer_object.transform(df[object_cols])
    
    
    df_scaled = pd.DataFrame(scaler.transform(df[float_cols]),
                                      index = df[float_cols].index,  # Scaling des données numériques
                                          columns = df[float_cols].columns)
    codes = encoder.transform(df[object_cols]).toarray()
    feature_names = encoder.get_feature_names()                         # Encodage des données catégorielles
    
    df_encoded = pd.DataFrame(codes,columns=feature_names).astype(int)
    
    df_full_stack = pd.concat([df_scaled, 
               df_encoded], axis=1)
  
  
    pred = model.predict(df_full_stack) # Jeu de données conforme pour l'input modèle
    
    return pred.tolist() 





# Un moyen simple de savoir quand il y a des erreurs
@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp


# Execute l'application
if __name__ == '__main__':
    app.run(debug= True)

