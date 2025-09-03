FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements и установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY app.py .

# Создание директории для файлов UML
RUN mkdir -p uml_files

# Установка переменных окружения
ENV PLANTUML_SERVER_URL=http://plantuml-server:8080
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 5000

# Запуск только Flask приложения
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]