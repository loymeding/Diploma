# Диплом

# Как установить зависимости и скачать датасет
## Настройка с помощью makefile
```bash
make
```
## Без использования makefile
### 1. Установить зависимости
```bash
pip install -r requirements.txt
```
### 2. Изменить файл с переменными среды .env
Необходимо добавить в файл переменную project_dir с указанием пути к директории проекта

### 3. Скачать датасет
Для этого необходимо зпустить скрипт boto/download_data.py
```bash
python boto/download_data.py
```

# Как запустить проект

## Настройка конфига
Для начала необходимо зайти в config/config.py и изменить значения параметров под требующуюся задачу.
Изначально конфиг полностью настроен под классификацию на датасете Animals10.

## Ручной запуск
### 1. Для запуска проекта без Node-RED необходимо вручную запустить скрипт, отвечающий за создание конфига, содержащего перечень команд для работы пайплайна
Для этого необходимо запустить скрипт src/make_json.py, передав в него следующие параметры:
+ --config_path: путь к конфигу (по умолчанию config.config)
+ --dataloaders_path: путь к функции создания даталоадеров (по умолчанию src.components.dataloaders.dataloaders)
+ --model_path: путь к функции, инициализирующей модель (по умолчанию src.components.models.resnet)
+ --train_path: путь к функции обучения (по умолчанию src.components.train.train)
+ --predict_path: путь к функции предсказания (по умолчанию src.components.predict.predict)


Пример запуска скрипта для решения задачи классификации на датасете Animals10
```bash
python src/make_json.py --config_path config.config --dataloaders_path src.components.dataloaders.dataloaders --model_path src.components.models.resnet --train_path src.components.train.train --predict_path src.components.predict.predict
```
### 2. Запуск пайплайна
Далее, когда имеется файл commands.yaml необходимо запустить скрипт exec_commands.py
```bash
python exec_commands.py
```

### ВАЖНО!
Перед тем как запускать скрипт make_json.py необходимо удалять старый commands.yaml файл, потому что данный скрипт не перезаписывает файл, а добавляет указанные команды в конец файла. Поэтому если необходимо запустить пайплайн с тем же набором команд, то создавать новый commands.yaml не обязательно, но если необходимо изменить список команд, то необъодимо удалить старый файл.

## При работе с Node-RED
### 1. Импортировать в Node-RED flows из директории node-red/flows.json
### 2. Изменить содержание узлов
Необходимо открыть узлы "Команды" и "Запуск пайплайна", расположенные в импортированном потоке и изменить в них пути к функциям на абсолютные пути, соответствующие расположению проекта.

![](https://github.com/loymeding/Diploma/blob/main/images/python-pipeline-flow.png)

![](https://github.com/loymeding/Diploma/blob/main/images/exec-flow.png)

### 3. Запуск узла

![](https://github.com/loymeding/Diploma/blob/main/images/node-red-flow.png)

Развернуть узел и последовательно нажать на запуск всех узлов: Class Config, Animals10, ResNet, Train, Predict. Затем запустить узел "Запуск пайплайна".


## Структура проекта

```bash
.
├── config                      
│   ├── main.yaml                   # Main configuration file
│   ├── model                       # Configurations for training model
│   │   ├── model1.yaml             # First variation of parameters to train model
│   │   └── model2.yaml             # Second variation of parameters to train model
│   └── process                     # Configurations for processing data
│       ├── process1.yaml           # First variation of parameters to process data
│       └── process2.yaml           # Second variation of parameters to process data
├── data            
│   ├── final                       # data after training the model
│   ├── processed                   # data after processing
│   └── raw                         # raw data
├── docs                            # documentation for your project
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks
├── pyproject.toml                  # Configure black

├── README.md                       # describe your project
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module 
│   ├── process.py                  # process data before training model
│   └── train_model.py              # train model
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module 
    ├── test_process.py             # test functions for process.py
    └── test_train_model.py         # test functions for train_model.py
```
## Auto-generate API documentation

To auto-generate API document for your project, run:

```bash
make docs
```
