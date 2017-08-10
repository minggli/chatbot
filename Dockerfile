
FROM python:3-onbuild

EXPOSE 5000 5001

CMD ["python", "-m", "chatbot.services.symptoms"]
CMD ["python", "-m", "chatbot.services.ask"]
