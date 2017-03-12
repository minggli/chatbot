from flask import Flask, make_response, request
from uuid import uuid4


class SessionController(object):

    def __init__(self, session, ambiguity_trials=3):

        self.sess = session

        self.sess['sid'] = str(uuid4())
        self.sess['count'] = 0
        self.sess['aggregate_texts'] = None

        self.question = None
        self.last_msg = None
        self.aggregate_questions = list()

        self.ambiguity_trials = ambiguity_trials

    def log(self, func):
        def wrapper(*args):
            self.count += 1
            self.last_msg = func
            return func(*args)
        return wrapper

    def aggregate(self):
        self.aggregate_questions.append(self.question)
        return ' '.join(self.aggregate_questions)

    @property
    def can_leaflet(self):
        if self.last_msg.__name__ == 'leaflet_prompt':
            return True
        else:
            return False

    def resp_leaflet(self):
        if self.can_leaflet and 'yes' in self.question.lower():
            return make_response(Messenger.leaflet, 200)
        elif self.can_leaflet and not 'yes' in self.question.lower():
            return make_response(Messenger.greeting, 200)

    def resp_no_result(self):
        return make_response(Messenger.no_result_prompt + Messenger.greeting, 200)


class Conversation(object):

    def __init__(self):

        self.output = None
        self.leaflets = None

    def greeting(self):
        return '\n\nHow can I help you?'

    @Controller.log
    def no_result_prompt(self):
        return  'Sorry I don\'t have enough information to help you, ' \
                'you can improve result by describing symptoms further.'

    @staticmethod
    def _present_multiple(output):
        return ';\n'.join([pair[0] + ' (~{:.0%})'.format(pair[1]) for pair in output])

    def several_results(self):
        return  'Based on what you told me, here are several possible reasons' \
                ', including: \n\n{0}'.format(present_multiple(self.output))

    def undecided_prompt(self):
        return  '\n\nYou can improve result by describing symptoms further.' \
                '\n\nCan you tell me more about the symptoms?'

    def undecided_reset(self):
        return '\n\nOk, we don\'t seem to get a confident result. Let\'s start again...'

    def single_result(self):
        return 'Based on what you told me, here is what I think: {0} (~{1:.0%})'.format(
                self.output[0], self.output[1])

    def leaflet_prompt(self):
        return '\n\nWould you like to have NHS leaflet?'

    def leaflet(self):
        return 'Here is the link: {0}'.format(self.leaflets[self.output[0]])

new = Controller()
msg = Messenger()
