#!venv/bin/python
from flask import Flask, jsonify, abort, make_response, request
import main

app = Flask(__name__)

@app.route('/chatbot/api/v1/symptoms', methods=['GET'])
def show_symptoms_json():
	return jsonify({'symptoms': {str.lower(key): main.mapping[key] for key in main.mapping}})


@app.route('/chatbot/api/v1/symptoms/<string:symptom_name>', methods=['GET'])
def show_symptom(symptom_name):

	symptom_name = str.lower(symptom_name)
	modified_mapping = {str.lower(key): main.mapping[key] for key in main.mapping}

	try:
		symptom = modified_mapping[symptom_name]
	except KeyError:
		abort(404)
	return jsonify({'symptom': symptom})


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Symptom Not found...use symptoms/<symptom_name>'}), 404)


@app.route('/chatbot/api/v1/ask', methods=['POST'])
def ask():

	def t(s=2):
		import time
		time.sleep(s)

	def query():
		pass
		# os.system('clear')
		# aggregate_text = list()
		# count = 0

		# question = {
		# 	'question': request.json.get('question', "")
		# }

		# output = main(query=question['question'])

		# responses = {
		# 	-1: 'How can I help you?',
		# 	-2: 'Can you tell me more about the symptoms?',
		# 	0: 'Based on what you told me, here is my diagnosis: {0}.'.format(output[0]),
		# 	1: 'Would you like to have NHS leaflet?',
		# 	2: 'here is the link: {0}'.format(mapping[output[0]]),
		# 	3: 'Based on what you told me, here are several possible reasons, including: \n\n{0}'.format(output[0]),
		# 	4: 'You can improve result by describing symptoms further.',
		# 	5: 'Sorry I don\'t have enough knowledge to help you, you can improve result by describing symptoms further.',
		# 	6: 'Ok, we don\'t seem to get anywhere. Let\'s start again...'
		# }
	return None

if __name__ == '__main__':
    app.run(debug=True)
