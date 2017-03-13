from flask import Flask, make_response, abort, request, session, g
from uuid import uuid4

from . import APP_CONFIG, API_BASE_URL
from ..engine import leaflets
from ..engine.naivebayes import Engine, naive_bayes_classifier
from ..conversation import Conversation

app = Flask(__name__)
app.config.update(APP_CONFIG)


@app.errorhandler(400)
def symptom_not_found(error):
    return make_response('endpoint accepts a json object e.g. {"question": "description of symptoms"}.', 400)


@app.before_request
def initiate_session():
    """set up session variables that persist per browser cookie for multiple requests."""
    session['sid'] = session.get('sid', 'Session ID: {0}\n'.format(str(uuid4())))
    session['count'] = session.get('count', 0)
    session['aggregate_texts'] = session.get('aggregate_texts', list())
    session['prev_outputs'] = session.get('prev_outputs', list())
    session['leaflet'] = session.get('leaflet', False)
    pass


@app.before_request
def initiate_controller():
    """instantiate conversation controller with global context per single request."""
    g.controller = Conversation(leaflets, 3)
    pass


@app.route(API_BASE_URL + '/ask', methods=['POST'])
def ask(clf=naive_bayes_classifier, engine=Engine):

    question = request.json.get('question', None)

    if not question:
        abort(400)

    controller = getattr(g, 'controller', None)
    controller.sess = session
    controller.curr_question = question
    controller.sess['aggregate_texts'].append(question)

    output = clf(query=' '.join(controller.sess['aggregate_texts']), engine=engine)
    resp = controller.converse(output)

    return make_response(session['sid'] + resp, 200)

if __name__ == '__main__':
    app.run(port=5000)
