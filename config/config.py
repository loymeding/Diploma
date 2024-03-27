import torch
import torch.nn as nn
import os
from dotenv import load_dotenv


# Функция-метрика
def accuracy_fn(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy


'''
Параметры:
- project_dir(str): путь к директории проекта
- data_path(str): путь к датасету
- train_dir(str): путь к обучающей выборке
- test_dir(str): путь к тестовой выборке
- valid_dir(str): путь к валидационной выборке

- num_classes(int): количество классов для предсказаний
- batch_size(int): размер батча
- shuffle(bool): перемешивание выборки
- lr(float): шаг градиентного спуска
- num_epochs(int): количество эпох обучения
- device(str): cuda или cpu
- criterion: функция потерь
- optimizer: оптимизатор
'''


project_dir = os.getenv('project_dir')
data_path = project_dir + '/data/raw/animals10/'
train_dir = data_path + 'train'
test_dir = data_path + 'test'
valid_dir = data_path + 'valid'

num_classes = 10
batch_size = 32
shuffle = False
lr = 0.001
num_epochs = 0
device = 'cpu'
criterion = nn.CrossEntropyLoss()
optimizer = None

