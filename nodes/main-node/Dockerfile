# Указываем базовый образ
FROM python:3.10.11

# Устанавливаем текущую рабочую директорию
WORKDIR .

# Копируем все файлы из текущей директории внутрь контейнера
COPY . /data

# Обновление установщика pip
RUN pip install --upgrade pip

# Установка всех библиотек, указанных в requirements.txt
RUN pip install --no-cache-dir -r /data/requirements.txt