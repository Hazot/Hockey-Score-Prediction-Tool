"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:5000 app:app

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
from flask_caching import Cache
import xgboost
import pandas as pd
from comet_ml import API


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

# Matching the name of the model to the name of the downloaded file
matching_model = {
    'xgboost-best-select-features': 'XGBoost_best_select_features.pkl',
    'xgboost-best-all-features': 'XGBoost_best_all_features.pkl',
    'xgboost-base-all-features': 'XGBoost_base_all_features.pkl'
}

# Flask Cache Configs
config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

# Configure the flask app and the cache to save data accross requests
app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app=app)

def get_api_key():
    """Retrieves the CometML API key from the current environment.

    Returns:
        string: cometML api key in string format
    """
    return os.environ.get("COMET_API_KEY", None)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # Basic logging configuration
    format = "%(asctime)s;%(levelname)s;%(message)s"
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=format)
    app.logger.info("Before first request - Start")

    # Load default model before handling the very first request
    default_model = {
        'workspace': 'ift-6758-projet-quipe-13',
        'model': 'xgboost-best-all-features',
        'version': '1.0.0'
    }
    try:
        # Try to load the model
        api = API(get_api_key())
        cache.set('api', api)
        default_model['filename'] = matching_model[default_model['model']]
        model_path = 'models/' + default_model['filename']
        api.download_registry_model(default_model['workspace'], default_model['model'], default_model['version'], output_path="models/")
        xgb = xgboost.XGBClassifier()
        xgb.load_model(model_path)
        cache.set('model', xgb)
        app.logger.info(f"Default model properly loaded: {default_model}.")
    except Exception as e:
        # If an error occurs, catch it and shows it in the logs.
        app.logger.info("Error while trying to load the default model.")
        app.logger.error(e)
    app.logger.info("Before first request - End")


@app.route("/hello")
def hello():
    """Simple requests that can be use to test the requests on the server. Returns 'Hello!'."""
    app.logger.info('Accessed page /hello - TESTING')
    return "Hello!\n\nThe purpose of this page is to know if requests work."


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    app.logger.info('Accessed page /logs')
    
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
    json = request.get_json()
    app.logger.info("Accessed page /download_registry_model")
    app.logger.info(f'Comet ML API parameters are the following: {json}')
    
    json['filename'] = matching_model[json['model']]

    model_path = 'models/' + json['filename']
    
    if Path(model_path).exists():
        # Load the model from local files if it exists.
        xgb = xgboost.XGBClassifier()
        xgb.load_model(model_path)
        cache.set('model', xgb)
        app.logger.info("Model locally updated (without download) from: " + model_path)
        response = 'Updated from local folder: ' + model_path
    else:
        try:
            # Else, try downloading the model from cometML and then loading it.
            api = cache.get('api')
            api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="models/")
            xgb = xgboost.XGBClassifier()
            xgb.load_model(model_path)
            cache.set('model', xgb)
            app.logger.info("Model successfully loaded from CometML API")
            app.logger.info("Model downloaded to: " + model_path)
            response = 'Updated model from CometML api and downloaded locally.'
        except Exception as e:
            # Catch the exception
            app.logger.error(e)
            response = 'Unexpected error while downloading the model: ' + str(e)
            
    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    app.logger.info("Accessed page /predict")
    app.logger.info("Start of prediction request.")
    r = request.get_json()
    df = pd.json_normalize(r, list(r.keys())[0])
    app.logger.debug('Input DataFrame shape:' + str(df.shape))
    try:
        preds = cache.get('model').predict_proba(df)[:, 1]
        app.logger.debug('Output prediction DataFrame shape:' + str(preds.shape))
        response = preds.tolist()
    except Exception as e:
        app.logger.error(f"The following error occured during the prediction: {e}")
        # response = ['error, check logs.'] * df.shape[0]
        response = str(e)
    return jsonify(response)  # response must be json serializable!
