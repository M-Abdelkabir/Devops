FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Le dataset est monté via volume ou copié
# Pour cet exemple on suppose qu'il est dans le dossier courant

CMD ["python", "train.py"]
