"""
    ask

    endpoint that accepts user medical questions per context session
"""

from flask import Flask, make_response, abort, request, session, g
from uuid import uuid4

from chatbot.engine import leaflets
from chatbot.engine.naivebayes import engine, naive_bayes_classifier
from chatbot.conversation import Conversation
from chatbot.settings import APP_CONFIG, API_BASE_URL

app = Flask(__name__)
app.config.update(APP_CONFIG)


@app.errorhandler(400)
def symptom_not_found(error):
    return make_response('endpoint accepts a json object e.g. \
    {"question": "description of symptoms"}.', 400)


@app.before_request
def initiate_session():
    """set up session that persists for multiple requests per cookie."""
    session.setdefault('sid', 'Session ID: {0}\n'.format(str(uuid4())))
    session.setdefault('count', 0)
    session.setdefault('aggregate_texts', list())
    session.setdefault('prev_outputs', list())
    session.setdefault('leaflet', False)


@app.before_request
def initiate_controller():
    """instantiate conversation controller with global context."""
    g.controller = Conversation(leaflets, 3)


@app.route(API_BASE_URL + '/ask', methods=['POST'])
def ask(clf=naive_bayes_classifier, engine=engine):

    question = request.json.get('question', None)

    if not question:
        abort(400)

    controller = getattr(g, 'controller', None)
    controller.sess = session
    controller.curr_question = question
    controller.sess['aggregate_texts'].append(question)

    output = clf(query=controller.sess[
                 'aggregate_texts'], engine=engine)
    resp = controller.converse(output)

    return make_response(session['sid'] + resp, 200)


if __name__ == '__main__':
    app.run(port=5000)
