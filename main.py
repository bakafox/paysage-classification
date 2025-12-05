import click
import torch
from torchvision import transforms
import kagglehub
from pathlib import Path
from datetime import datetime

from samples import read_samples, read_splits, create_splits
from test import model_test, show_metrics
from train import model_train
from visualize import visualize


@click.command()
@click.option('--mode', required=True, type=click.Choice([
    'train', 'test', 'visualize' # Тренировка, тестирование, визуализация датасета
]), help='Режим работы')
@click.option('--e', default=20, show_default=True, help='Количество эпох обучения')
@click.option('--bs', default=256, show_default=True, help='Размер батча')
@click.option('--ss', default=75, show_default=True, help='Размер стороны изображения')
@click.option('--path-labels', default='data/labels.txt', help='Путь до файла с названиями классов')
@click.option('--path-splits', default='data/', help='Путь до CSVшек с выборками train-eval-test')
@click.option('--path-checkpoint', default='models/', help='Путь до модели (при test), либо место сохранения оной (при train)')
@click.option('--dataset-name', default='nitishabharathi/scene-classification', help='Название датасета на KaggleHub')
@click.option('--dataset-splits', default='0.7-0.2-0.1', help='Как разделить датасет (три числа от 0 до 1 через дефис)')
def run(
    mode, e, bs, ss, # Epochs, BatchSize, SampleSize
    path_labels, path_splits, path_checkpoint, dataset_name, dataset_splits
):
    # Для начала, скачем датасет! Если оный уже был скачан,
    # KaggleHub сам определит это и ничего перекачивать не будет.
    path_dataset = Path(
        kagglehub.dataset_download(dataset_name)
    )
    path_images = path_dataset / 'train-scene classification' / 'train'
    path_csv = path_dataset / 'train-scene classification' / 'train.csv'
    # print(path_dataset)

    # Считаем входные метки, чтобы не делать этого по многу раз
    labels_f = open(path_labels, 'r', encoding='utf-8')
    labels_list = list(map(lambda x: x.replace('\n', ''), labels_f.readlines()))
    labels = {label_no: label_name for label_no, label_name in enumerate(labels_list)}

    # Как мы хочем выполнить препроцессинг изображений
    transform_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([ss, ss]), # Ориг. разрешение: [150, 150]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    print(f'Подготовка к {mode} завершена, обработка...')


    if mode == 'train':
        # Читаем CSV-шку со сплитами на train-eval-test,
        # либо создаём её с нуля, если таковой не существует
        try:
            [train_samples, eval_samples] = read_splits(['train', 'eval'], path_splits)
        except:
            print('Файл сплитов не найден! Создаю новые...')
            create_splits(path_csv, dataset_splits, path_splits)
            [train_samples, eval_samples] = read_splits(['train', 'eval'], path_splits)

        # Обучаем с нуля модель и сохраняем чек-поинты
        model = model_train(
            train_samples,
            eval_samples,
            path_images,
            labels,
            transform_list,
            e,
            bs
        )

        model_date = datetime.now().strftime("%y%m%d")
        model_name = f'{path_checkpoint}{model_date}-{e}-{bs}-{ss}.pt'

        torch.save(model, model_name)
        print('Обучение завершено!')


    elif mode == 'test':
        # Читаем CSV-шку со сплитами данных на train-eval-test,
        # либо создаём её с нуля, если таковой не существует
        try:
            [test_samples] = read_splits(['test'], path_splits)
        except:
            print('Файл сплитов не найден! Создаю новые...')
            create_splits(path_csv, dataset_splits, path_splits)
            [test_samples] = read_splits(['test'], path_splits)

        # Загружаем модель и прогоняем на тестовых данных
        model = torch.load(path_checkpoint, weights_only=False)

        true, pred = model_test(
            model,
            test_samples,
            path_images,
            transform_list,
            bs
        )
        show_metrics(true, pred, labels)
    

    elif mode == 'visualize':
        # В случае с визуализацией, нас сплиты не интересуют
        samples = read_samples(path_csv)

        visualize(
            samples,
            path_images,
            labels_list,
            transform_list
        )


if __name__ == '__main__':
    run()
