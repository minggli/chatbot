from flask import Flask, jsonify, make_response, url_for, redirect

from . import APP_CONFIG, API_BASE_URL
from app.engine import leaflets

app = Flask(__name__)
app.config.update(APP_CONFIG)


@app.route(API_BASE_URL + '/symptoms', methods=['GET'])
def show_symptoms():
    """show all symptoms captured from scrapper"""
    resp = \
        make_response(jsonify(
            {'symptoms': {key.lower(): leaflets[key] for key in leaflets}}
        ), 200)
    return resp


@app.route(API_BASE_URL + '/symptoms/<string:symptom_name>', methods=['GET'])
def show_symptom(symptom_name):
    """show a single symptom based on input"""
    symptom_name = symptom_name.lower()
    modified_mapping = {key.lower(): leaflets[key] for key in leaflets}
    try:
        resp = make_response(jsonify(
            {'symptoms': {symptom_name: modified_mapping[symptom_name]}}
        ), 200)
    except KeyError:
        return redirect(url_for('show_symptoms'))
    return resp

if __name__ == '__main__':
    app.run(port=5001)
