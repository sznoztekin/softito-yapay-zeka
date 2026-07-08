FROM python:3.10-slim

WORKDIR /app

# Sorun çıkaran apt-get adımını kaldırıp doğrudan kütüphaneleri yüklüyoruz
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]