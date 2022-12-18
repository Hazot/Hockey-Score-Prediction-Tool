"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import xgboost
import requests
import pandas as pd
from comet_ml import API


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
# MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
matching_model = {
    'xgboost-best-all-features': 'XGBoost_best_all_features.pkl',
    'xgboost-all-features': 'XGBoost_all_features.pkl'
}

app = Flask(__name__)


def get_api_key():
    return os.environ.get("COMET_API_KEY", None)


def lower_model_path(path):
    for file in os.listdir(path):
        os.rename(path + file, path + file.lower())


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    format = "%(asctime)s;%(levelname)s;%(message)s"
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=format)
    app.logger.info("Before first request - Start")

    # TODO: any other initialization before the first request (e.g. load default model)
    default_model = {
        'workspace': 'ift-6758-projet-quipe-13',
        'model': 'xgboost-best-all-features',
        'version': '1.0.0'
    }
    global api
    api = API(get_api_key())
    
    default_model['filename'] = matching_model[default_model['model']]
    global model
    model_path = 'models/' + default_model['filename']
    api.download_registry_model(default_model['workspace'], default_model['model'], default_model['version'], output_path="models/")
    xgb = xgboost.XGBClassifier()
    xgb.load_model(model_path)
    model = xgb
    # requests.post('http://0.0.0.0:5000/download_registry_model', json=default_model)
    app.logger.info("Before first request - End")


@app.route("/hello")
def hello():
    app.logger.info('Accessed page /hello - Hello!')
    return "Hello!"


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    app.logger.info('Accessed page /logs')
    # TODO: read the log file specified and return the data
    with open(LOG_FILE) as f:
        data = f.read().splitlines()

    response = data
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required)
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info('Comet ML API parameters are the following:')
    app.logger.info(json)
    
    json['filename'] = matching_model[json['model']]

    model_path = 'models/' + json['filename']
    
    if Path(model_path).exists():
        # TODO: if yes, load that model and write to the log about the model change.  
        # eg: app.logger.info(<LOG STRING>)
        xgb = xgboost.XGBClassifier()
        xgb.load_model(model_path)
        model = xgb
        app.logger.info("Model updated without download from: " + model_path)
        response = 'Updated from local folder.'

    else:
        # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
        # about the model change. If it fails, write to the log about the failure and keep the 
        # currently loaded model
        
        # Download a Registry Model:
        try:
            api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="models/")
            xgb = xgboost.XGBClassifier()
            xgb.load_model(model_path)
            model = xgb
            app.logger.info("Model successfully downloaded to: " + model_path)
            response = 'Updated from CometML api.'
        except Exception as e:
            response = 'Unexpected error while downloading the model.'
            app.logger.info(e)
            

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here
    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info("Prediction start")
    app.logger.info(json)
    df = pd.read_json(json, orient='split')
    pred = model.predict_proba(df)
    df_pred = pd.DataFrame(pred)
    response = df_pred.to_json(orient='split')
    app.logger.info(response)
    app.logger.info("Prediction end")
    return jsonify(response)  # response must be json serializable!
