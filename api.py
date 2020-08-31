
import json

from flask import Flask

from MLEng_Exercise.exercise.mongo.mongodb import MongoDB
from exercise.utils import logger

log = logger.get_logger("main")

app = Flask(__name__)

@app.route("/hydrated_model_predictions", methods=['GET'])
def get_predicted_labels():
    """
    :return:
    """
    mongo_db = MongoDB(database_name="MLEng_Exercise_DB", collection_name="predicted_labels")
    predictions = [js["Job Seniority"] for js in list(mongo_db.retrieve_all())]
    request_in_json = json.dumps({"Predictions": predictions})
    return request_in_json


@app.route("/latest_online_predictions", methods=['GET'])
def get_online_predicted_labels():
    """
    :return:
    """
    mongo_db = MongoDB(database_name="MLEng_Exercise_DB", collection_name="online_predicted_labels")
    predictions = [js["Job Seniority"] for js in list(mongo_db.retrieve_all())]
    request_in_json = json.dumps({"Predictions": predictions})
    return request_in_json


@app.route("/loaded_model_predictions", methods=['GET'])
def get_predicted_labels_from_loaded_models():
    """
    :return:
    """
    mongo_db = MongoDB(database_name="MLEng_Exercise_DB", collection_name="loaded_model_predictions")
    predictions = [js["Job Seniority"] for js in list(mongo_db.retrieve_all())]
    request_in_json = json.dumps({"Predictions": predictions})
    return request_in_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)

