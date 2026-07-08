FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Veri klasörünü Docker imajının içine kalıcı olarak kopyalıyoruz
COPY data/ ./data/

COPY train.py .
CMD ["python", "train.py"]