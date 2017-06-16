from chatbot import settings

if settings.ENGINE.upper() == 'TENSORFLOW':
    from chatbot.engine.lstm import inference
    engine = inference
elif settings.ENGINE.upper() == 'NLTK':
    from chatbot.engine.naivebayes import classify
    engine = classify
