from chatbot.settings import ENGINE

if ENGINE.upper() == 'TENSORFLOW':
    from chatbot.engine.lstm import inference as engine
elif ENGINE.upper() == 'NLTK':
    from chatbot.engine.naivebayes import classify as engine
