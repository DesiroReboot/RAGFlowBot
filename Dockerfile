FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.org/simple \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    http_proxy= \
    https_proxy= \
    NO_PROXY= \
    no_proxy=

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy pip install -r requirements.txt

COPY src ./src
COPY config ./config
COPY .env.example ./.env.example
COPY README.md ./README.md

RUN mkdir -p /app/DB /app/logs /app/Eval/report /app/Eval/trace /app/Eval/HTML /app/kb

EXPOSE 8000

CMD ["python", "-m", "src.main"]
