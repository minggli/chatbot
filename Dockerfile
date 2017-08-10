
FROM python:3-onbuild

EXPOSE 5000

RUN python -m spacy download en_core_web_md

CMD ["python", "-m", "chatbot.services.ask"]
