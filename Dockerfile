FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
COPY .env ./

# Install the Project dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y jq

RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev

# Copy project
COPY . .

# Copy and activate script to start api and chainlit
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
