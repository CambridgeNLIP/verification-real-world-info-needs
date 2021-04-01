FROM continuumio/miniconda3

RUN mkdir /fever
WORKDIR /fever

VOLUME /fever/data

RUN apt update
RUN apt install -y build-essential

ADD requirements.txt /fever/
RUN pip install -r requirements.txt
RUN python -m spacy download en

RUN mkdir -pv src
RUN mkdir -pv configs

ADD src src
ADD www www
ADD templates templates

ENV PYTHONPATH src
CMD ["waitress-serve", "--host=0.0.0.0","--port=8080", "--call", "annotation.annotation_service:boot"]