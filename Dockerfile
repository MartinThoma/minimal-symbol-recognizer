FROM python:3.8.1-slim-buster

# Copy projects code
COPY . /opt/app
WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN pip install .

# Start app

ENTRYPOINT ["minimal_symbol_recognizer"]
CMD ["run-server", "--model", "model4.h5", "--labels", "labels.csv"]
