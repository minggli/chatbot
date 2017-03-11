from flask import Flask, jsonify, abort, make_response

from . import API_BASE_URL
from ..engine import leaflets

app = Flask(__name__)


@app.errorhandler(404)
def symptom_not_found(error):
    return make_response('symptom is not found, please choose from all symptoms.', 404)


@app.route(API_BASE_URL + '/symptoms', methods=['GET'])
def show_symptoms():
    """show all symptoms captured from scrapper"""
    resp = \
        make_response(jsonify({'symptoms': {key.lower(): leaflets[key] for key in leaflets}}), 200)
    return resp


@app.route(API_BASE_URL + '/symptoms/<string:symptom_name>', methods=['GET'])
def show_symptom(symptom_name):
    """show a single symptom based on input"""
    symptom_name = symptom_name.lower()
    modified_mapping = {key.lower(): leaflets[key] for key in leaflets}
    try:
        resp = make_response(jsonify({'symptoms': {symptom_name: modified_mapping[symptom_name]}}), 200)
    except KeyError:
        abort(404)
    return resp

if __name__ == '__main__':
    app.run(port=5001, debug=False)
