## getting python image
FROM python:3.8.1-alpine3.11
# Install additional packages
RUN apk add --no-cache build-base
## setting working directory
WORKDIR /app

## copying requirements.txt to working directory
COPY requirements.txt .

## Upgrade pip
RUN pip install --upgrade pip

RUN pip install --upgrade pip setuptools wheel

## installing requirements
RUN pip install -r requirements.txt


## copying all files to working directory
COPY . .

## exposing port 5000
EXPOSE 5000

## running app.py

CMD ["uvicorn", "main:app" "--reload"]

