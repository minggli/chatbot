from chatbot import settings

if settings.ENGINE.upper() == 'TENSORFLOW':
    from chatbot.engine.lstm import sess, inference
    engine, classifier = sess, inference
elif settings.ENGINE.upper() == 'NLTK':
    from chatbot.engine.naivebayes import engine, naive_bayes_classifier
    engine, classifier = engine, naive_bayes_classifier
