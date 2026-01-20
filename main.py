import logging
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from paysage_classification.dataset.downloader import check_dataset, download_dataset
from paysage_classification.dataset.samples import (
    create_splits,
    read_samples,
    read_splits,
)
from paysage_classification.inference import inference
from paysage_classification.test import model_test, show_metrics
from paysage_classification.train import model_train
from paysage_classification.visualize import visualize

log = logging.getLogger(__name__)


def model_name_format(path_checkpoint, e, bs, ss, best_epoch, best_acc) -> str:
    model_date = datetime.now().strftime('%y%m%d')
    return (
        f'{path_checkpoint}{model_date}_'
        f'{e}-{bs}-{ss}_'
        f'{best_epoch}-{best_acc:.4f}.pt'
    )


@hydra.main(version_base=None, config_path='', config_name='config')
def run(cfg: DictConfig):
    params = OmegaConf.to_container(cfg['params'])
    # print(params)

    path_dataset = (
        Path(params['path-data']) / 'nitishabharathi/scene-classification/versions/1'
    )
    path_images = path_dataset / 'train-scene classification' / 'train'
    path_csv = path_dataset / 'train-scene classification' / 'train.csv'

    # Для начала, скачем датасет! УИИИИИИИИИ
    if not check_dataset(params['dataset-size'], path_dataset):
        print('Датасет отсутствует или повреждён, скачивание с DVC...')
        download_dataset()

    # Считаем входные метки, чтобы не делать этого по многу раз
    labels_f = open(params['path-labels'], 'r', encoding='utf-8')
    labels_list = list(map(lambda x: x.replace('\n', ''), labels_f.readlines()))
    labels = {label_no: label_name for label_no, label_name in enumerate(labels_list)}

    # Как мы хочем выполнить препроцессинг изображений
    transform_list = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([params['ss'], params['ss']]),  # В датасете: [150, 150]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print(f'Подготовка к {params["mode"]} завершена, обработка...')

    if params['mode'] == 'train':
        # Обучаем с нуля модель и сохраняем чек-поинты
        try:
            [train_samples, eval_samples] = read_splits(
                ['train', 'eval'], params['path-data']
            )

        except Exception:
            print('Файл сплитов не найден! Создаю новые...')
            create_splits(path_csv, params['dataset-splits'], params['path-data'])
            [train_samples, eval_samples] = read_splits(
                ['train', 'eval'], params['path-data']
            )

        model, best_acc, best_epoch = model_train(
            train_samples,
            eval_samples,
            path_images,
            labels,
            transform_list,
            params['e'],
            params['bs'],
        )

        model_name = model_name_format(
            params['path-checkpoint'],
            params['e'],
            params['bs'],
            params['ss'],
            best_epoch,
            best_acc,
        )

        torch.save(model, model_name)
        print('Обучение завершено!')

    elif params['mode'] == 'test':
        # Загружаем модель и прогоняем на тестовых данных
        try:
            [test_samples] = read_splits(['test'], params['path-data'])
        except Exception:
            print('Файл сплитов не найден! Создаю новые...')
            create_splits(path_csv, params['dataset-splits'], params['path-data'])
            [test_samples] = read_splits(['test'], params['path-data'])

        model = torch.load(params['path-checkpoint'], weights_only=False)

        true, pred = model_test(
            model, test_samples, path_images, transform_list, params['bs']
        )
        show_metrics(true, pred, labels)

    elif params['mode'] == 'visualize':
        # В случае с визуализацией, нас сплиты не интересуют
        samples = read_samples(path_csv)

        visualize(samples, path_images, labels_list, transform_list)

    elif params['mode'] == 'inference':
        # В остальных случаях -- просто выполняем инференс
        model = torch.load(params['path-checkpoint'], weights_only=False)

        if Path(params['path-infers']).is_dir():
            image_pathes = [
                f for f in Path(params['path-infers']).iterdir() if f.is_file()
            ]

            for image_path in image_pathes:
                inference(
                    model,
                    image_path,
                    labels_list,
                    transform_list,
                    output=params['infer-output'],
                )

        elif Path(params['path-infers']).is_file():
            inference(
                model,
                Path(params['path-infers']),
                labels_list,
                transform_list,
                output=params['infer-output'],
            )

        else:
            print('Некорректный путь до файла или каталога.')

    else:
        print('Указанного режима работы не сущестует!')


if __name__ == '__main__':
    run()
