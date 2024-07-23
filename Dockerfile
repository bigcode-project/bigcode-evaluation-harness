#FROM ubuntu:22.04
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y python3 python3-pip

COPY . /app

WORKDIR /app
RUN mkdir -p /app/data
#RUN test -f /app/generations.json && rm /app/generations.json || true

RUN pip3 install .

CMD ["python3", "main.py"]
