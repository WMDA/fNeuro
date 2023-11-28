FROM python:3.10
RUN apt-get update && apt-get install -y git
WORKDIR /app
RUN git clone https://github.com/WMDA/fNeuro.git
WORKDIR /app/fNeuro
RUN python3 -m pip install .