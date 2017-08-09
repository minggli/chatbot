
FROM python:3-onbuild

EXPOSE 5000 5001

CMD ['python3 -m', 'chatbot.services.ask']
