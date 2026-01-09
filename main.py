import logging
from datetime import datetime
from pathlib import Path
from test import model_test, show_metrics

import hydra
import kagglehub
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from inference import inference
from samples import create_splits, read_samples, read_splits
from train import model_train
from visualize import visualize

log = logging.getLogger(__name__)


def model_name_format(path_checkpoint, model_date, e, bs, ss):
    return f'{path_checkpoint}{model_date}-{e}-{bs}-{ss}.pt'


@hydra.main(version_base=None, config_path='config', config_name='config')
def run(
    # mode,
    # e,
    # bs,
    # ss,  # Epochs, BatchSize, SampleSize
    # params["path-labels"],
    # params["path-splits"],
    # params["path-checkpoint"],
    # params["path-infers"],
    # params["dataset-name"],
    # params["dataset-splits"],
    cfg: DictConfig,
):
    params = OmegaConf.to_container(cfg['params'])
    print(params)

    # Для начала, скачем датасет! Если оный уже был скачан,
    # KaggleHub сам определит это и ничего перекачивать не будет.
    path_dataset = Path(kagglehub.dataset_download(params["dataset-name"]))
    path_images = path_dataset / 'train-scene classification' / 'train'
    path_csv = path_dataset / 'train-scene classification' / 'train.csv'
    # print(path_dataset)

    # Считаем входные метки, чтобы не делать этого по многу раз
    labels_f = open(params["path-labels"], 'r', encoding='utf-8')
    labels_list = list(map(lambda x: x.replace('\n', ''), labels_f.readlines()))
    labels = {label_no: label_name for label_no, label_name in enumerate(labels_list)}

    # Как мы хочем выполнить препроцессинг изображений
    transform_list = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                [params["ss"], params["ss"]]
            ),  # Ориг. разрешение: [150, 150]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print(f'Подготовка к {params["mode"]} завершена, обработка...')

    if params["mode"] == 'train':
        # Обучаем с нуля модель и сохраняем чек-поинты
        try:
            [train_samples, eval_samples] = read_splits(
                ['train', 'eval'], params["path-splits"]
            )
        except Exception:
            print('Файл сплитов не найден! Создаю новые...')
            create_splits(path_csv, params["dataset-splits"], params["path-splits"])
            [train_samples, eval_samples] = read_splits(
                ['train', 'eval'], params["path-splits"]
            )

        model = model_train(
            train_samples,
            eval_samples,
            path_images,
            labels,
            transform_list,
            params["e"],
            params["bs"],
        )

        model_date = datetime.now().strftime('%y%m%d')
        model_name = model_name_format(
            params["path-checkpoint"],
            model_date,
            params["e"],
            params["bs"],
            params["ss"],
        )

        torch.save(model, model_name)
        print('Обучение завершено!')

    elif params["mode"] == 'test':
        # Загружаем модель и прогоняем на тестовых данных
        try:
            [test_samples] = read_splits(['test'], params["path-splits"])
        except Exception:
            print('Файл сплитов не найден! Создаю новые...')
            create_splits(path_csv, params["dataset-splits"], params["path-splits"])
            [test_samples] = read_splits(['test'], params["path-splits"])

        model = torch.load(params["path-checkpoint"], weights_only=False)

        true, pred = model_test(
            model, test_samples, path_images, transform_list, params["bs"]
        )
        show_metrics(true, pred, labels)

    elif params["mode"] == 'visualize':
        # В случае с визуализацией, нас сплиты не интересуют
        samples = read_samples(path_csv)

        visualize(samples, path_images, labels_list, transform_list)

    else:
        # В остальных случаях -- просто выполняем инференс
        model = torch.load(params["path-checkpoint"], weights_only=False)

        if Path(params["path-infers"]).is_dir():
            image_pathes = [
                file for file in Path(params["path-infers"]).iterdir() if file.is_file()
            ]

            for image_path in image_pathes:
                inference(
                    model,
                    image_path,
                    labels_list,
                    transform_list,
                    output='std',  # output='mpl'
                )

        elif Path(params["path-infers"]).is_file():
            inference(
                model,
                Path(params["path-infers"]),
                labels_list,
                transform_list,
                output='std',
            )

        else:
            print('Некорректный путь до файла или каталога.')


if __name__ == '__main__':
    run()
