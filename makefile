# Переменная для пути к текущей директории
PROJECT_DIR := $(shell pwd)

# Цель для установки зависимостей из requirements.txt
install:
	pip install -r requirements.txt

# Цель для создания файла .env и добавления в него значений
create_env_file:
	@echo "aws_access_key_id=jjQG3AystFqYVxCWUghFRs" > .env
	@echo "aws_secret_access_key=bwEjD23yDj5kjSavtSB35MNBWnGFVSvBFVjofH6ckpjG" >> .env
	@echo "project_dir=$(PROJECT_DIR)" >> .env

# Цель для добавления переменной project_dir в файл .env
add_project_dir_to_env:
	@echo "project_dir=$(PROJECT_DIR)" >> .env

# Цель для загрузки данных с помощью скрипта download_data.py
download_data:
	python boto/download_data.py
