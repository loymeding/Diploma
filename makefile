# Переменная для пути к текущей директории
PROJECT_DIR := $(shell pwd)

# Цель для установки зависимостей из requirements.txt
install:
	pip install -r requirements.txt

# Цель для добавления переменной project_dir в файл .env
add_project_dir_to_env:
	@echo "project_dir=$(PROJECT_DIR)" >> .env

# Цель для загрузки данных с помощью скрипта download_data.py
download_data:
	python boto/download_data.py
