from flask import Flask, jsonify, abort, make_response

from . import mapping, API_BASE_URL

app = Flask(__name__)


@app.route(API_BASE_URL + '/symptoms', methods=['GET'])
def show_symptoms_json():
    """show all symptoms captured from wrapper"""
    return make_response(
        jsonify(
            {'symptoms': {str.lower(key): mapping[key] for key in mapping}}
        ), 200)


@app.route(API_BASE_URL + '/symptoms/<string:symptom_name>', methods=['GET'])
def show_symptom(symptom_name):

    symptom_name = str.lower(symptom_name)
    modified_mapping = {str.lower(key): mapping[key] for key in mapping}

    try:
        symptom = modified_mapping[symptom_name]
    except KeyError:
        abort(404)
    return make_response(jsonify({'symptom': symptom}), 200)

if __name__ == '__main__':
    app.run(port=5002, debug=False)
