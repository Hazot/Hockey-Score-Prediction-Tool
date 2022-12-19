"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:5000 app:app

gunicorn can be installed via:

    $ pip install gunicorn

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

matching_model = {
    'xgboost-best-select-features': 'XGBoost_best_select_features.pkl',
    'xgboost-best-all-features': 'XGBoost_best_all_features.pkl',
    'xgboost-base-all-features': 'XGBoost_base_all_features.pkl'
}

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app=app)

def get_api_key():
    return os.environ.get("COMET_API_KEY", None)


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
    api = API(get_api_key())
    cache.set('api', api)
    default_model['filename'] = matching_model[default_model['model']]
    model_path = 'models/' + default_model['filename']
    api.download_registry_model(default_model['workspace'], default_model['model'], default_model['version'], output_path="models/")
    xgb = xgboost.XGBClassifier()
    xgb.load_model(model_path)
    cache.set('model', xgb)
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
        cache.set('model', xgb)
        app.logger.info("Model updated without download from: " + model_path)
        response = 'Updated from local folder.'

    else:
        # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
        # about the model change. If it fails, write to the log about the failure and keep the 
        # currently loaded model
        
        # Download a Registry Model:
        try:
            api = cache.get('api')
            api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="models/")
            xgb = xgboost.XGBClassifier()
            xgb.load_model(model_path)
            cache.set('model', xgb)
            app.logger.info("Model successfully downloaded to: " + model_path)
            response = 'Updated from CometML api.'
        except Exception as e:
            response = 'Unexpected error while downloading the model: ' + str(e)
            app.logger.error(e)
            

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
    app.logger.info("Prediction start")
    r = request.get_json()
    df = pd.json_normalize(r, list(r.keys())[0])
    app.logger.debug('Input DataFrame shape:' + str(df.shape))
    preds = cache.get('model').predict_proba(df)[:, 1]
    app.logger.debug('Prediction DataFrame shape:' + str(preds.shape))
    response = preds.tolist()

    app.logger.info("Prediction end")
    return jsonify(response)  # response must be json serializable!
