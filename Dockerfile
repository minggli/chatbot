
FROM python:3-onbuild

EXPOSE 5000

CMD ["python", "-m", "chatbot.services.ask"]
