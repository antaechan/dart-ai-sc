FROM python:3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /dart-ai-sc

COPY requirements.txt /dart-ai-sc/
RUN pip install -r requirements.txt

COPY src /dart-ai-sc/src

EXPOSE 7000
CMD ["tail", "-f", "/dev/null"]