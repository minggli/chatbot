"""
    Conversation

    rule-based retrieval conversation module with pre-determined responses.
"""


class Conversation(object):

    def __init__(self, leaflet, max_trials):

        self.leaflets = leaflet
        self.max_trials = max_trials

        self.curr_question = None
        self.output = None
        self.sess = None

    def converse(self, output):

        self.output = output
        self.sess['prev_outputs'].append(output)
        self.sess['count'] += 1

        if self.sess['leaflet']:
            if 'yes' in self.curr_question.lower():
                self.initiate_sess()
                return self._leaflet() + self._greeting()
            else:
                self.initiate_sess()
                return self._greeting()

        if not self.output:
            if self.sess['count'] < self.max_trials:
                return self._no_result_prompt()
            elif self.sess['count'] == self.max_trials:
                self.initiate_sess()
                return self._undecided_reset() + self._greeting()

        elif isinstance(self.output, list):
            if self.sess['count'] < self.max_trials:
                return self._several_possibilities() + self._undecided_prompt()
            elif self.sess['count'] == self.max_trials:
                self.initiate_sess()
                return self._several_possibilities() + self._undecided_reset() + self._greeting()

        elif isinstance(self.output, tuple):
            self.initiate_sess()
            self.sess['leaflet'] = True
            return self._single_result() + self._leaflet_prompt()

    def initiate_sess(self):
        self.sess['count'] = 0
        self.sess['leaflet'] = False
        self.sess['aggregate_texts'] = list()

    def _greeting(self):
        return '\n\nHow can I help you?'

    def _no_result_prompt(self):
        return '\nSorry I don\'t have enough symptom details, ' \
            'can you tell me more?'

    def _several_possibilities(self):
        return '\nBased on what you told me, here are several possible reasons' \
            ', including: \n\n{0}\n'.format(';\n'.join(
                [pair[0] + ' (~{:.0%})'.format(pair[1]) for pair in self.output]))

    def _undecided_prompt(self):
        return '\nYou can improve result by describing symptoms further.' \
            '\n\nCan you tell me more about the symptoms?'

    def _undecided_reset(self):
        return '\nOk, we don\'t seem to get a confident result. Let\'s start again...'

    def _single_result(self):
        return '\nBased on what you told me, here is what I think: {0} (~{1:.0%})'.format(
            self.output[0], self.output[1])

    def _leaflet_prompt(self):
        return '\n\nWould you like to have NHS leaflet?'

    def _leaflet(self):
        return '\nHere is the link: {0}'.format(self.leaflets[self.sess['prev_outputs'][-2][0]])
