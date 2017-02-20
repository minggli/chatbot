from . import raw_data, NLP_PROCESSOR, mapping, labels, API_BASE_URL
from flask import Flask, jsonify, abort, make_response, request
from helpers import NLPProcessor
from nltk.tokenize import word_tokenize
from engine.naivebayes import NB_classifier, train_model

app = Flask(__name__)

nlp = NLPProcessor(attrs=NLP_PROCESSOR)
processed_data = nlp.process(raw_data)

Engine = train_model(processed_data, labels, n=100)

responses = list()
aggregate_text = list()
count = 0

@app.route(API_BASE_URL + '/ask', methods=['POST'])
def ask(clf=NB_classifier, engine=Engine, nlp=nlp, ambiguity_trials=3):
	"""this function needs completely refactor"""
	
	global responses
	global aggregate_text
	global count

	question = request.json['question']
	aggregate_text.append(question)

	output = clf(query=' '.join(aggregate_text), engine=engine, nlp=nlp)

	try:
		if responses[-1][1] == 0 and 'yes' in question.lower():
			output = responses[-1]
			responses = list()
			aggregate_text = list()

			respond_templates = {
				-1: '\n\nHow can I help you?',
				2: 'here is the link: {0}'.format(mapping[output[0]])
			}

			return make_response(respond_templates[2] + respond_templates[-1])

		elif responses[-1][1] == 0 and not 'yes' in question.lower():
			responses = list()
			aggregate_text = list()

			respond_templates = {
				-1: '\n\nHow can I help you?',
			}

			return make_response(respond_templates[-1])

	except IndexError:
		pass


	if output and output[1] == 0:
		# confident diagnosis
		responses.append(output)
		aggregate_text = list()
		count = 0

		respond_templates = {
			-1: '\n\nHow can I help you?',
			-2: 'Can you tell me more about the symptoms?',
			0: 'Based on what you told me, here is my best guess: {0}.'.format(output[0]),
			1: '\n\nWould you like to have NHS leaflet?',
			2: 'here is the link: {0}'.format(mapping[output[0]]),
			3: 'Based on what you told me, here are several possible reasons, including: \n\n{0}'.format(output[0]),
			4: '\n\nYou can improve result by describing symptoms further.',
			5: 'Sorry I don\'t have enough information to help you, you can improve result by describing symptoms further.',
			6: 'Ok, we don\'t seem to get anywhere. Let\'s start again...'
		}
		return make_response(respond_templates[0] + respond_templates[1])

	elif output and output[1] == 1:
		# multiple possibilities
		count += 1

		respond_templates = {
			-1: '\n\nHow can I help you?',
			-2: '\n\nCan you tell me more about the symptoms?',
			3: 'Based on what you told me, here are several possible reasons, including: \n\n{0}'.format(output[0]),
			4: '\n\nYou can improve result by describing symptoms further.',
			6: '\n\nOk, we don\'t seem to get a confident result. Let\'s start again...'
		}
		if count == ambiguity_trials:
			aggregate_text = list()
			count = 0
			return make_response(respond_templates[3] + respond_templates[6] + respond_templates[-1])
		else:
			return make_response(respond_templates[3] + respond_templates[4] + respond_templates[-2])

	else:
		# None
		count += 1

		respond_templates = {
			-1: '\n\nHow can I help you?',
			-2: '\n\nCan you tell me more about the symptoms?',
			5: 'Sorry I don\'t have enough information to help you, you can improve result by describing symptoms further.',
			6: 'Ok, we don\'t seem to get anywhere. Let\'s start again...'
		}
		if count == ambiguity_trials:
			aggregate_text = list()
			count = 0
			return make_response(respond_templates[6] + respond_templates[-1])
		else:
			return make_response(respond_templates[5] + respond_templates[-2])


if __name__ == '__main__':
	app.run(port=5000, debug=False)
