from copy import deepcopy
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from .dataset.samples import Sample
from .model.classifier import SceneClassifier
from .model.preprocess import SceneDataset


def model_train(
    samples_train: list[Sample],
    samples_eval: list[Sample],
    path_images: Path,
    labels: dict[int, str],
    transforms,
    epochs: int = 20,
    batch_size: int = 256,
    samples_size: int = 88,
    max_degradation: int = 10,
) -> tuple[SceneClassifier, float, int]:
    # Инициализируем тензорборд
    writer = SummaryWriter()

    # Создаём экземпляры датасетов и модели
    dataset_train = SceneDataset(samples_train, path_images, transforms)

    dataset_eval = SceneDataset(samples_eval, path_images, transforms)

    model = SceneClassifier(
        epochs,
        len(labels),
    )
    model.train()

    # Формируем из датасета батчи
    batched_dataset_train = DataLoader(dataset_train, batch_size, shuffle=True)
    batched_dataset_eval = DataLoader(dataset_eval, batch_size, shuffle=True)

    best_acc = 0.0
    best_epoch = 0
    best_param = None

    try:
        for epoch in range(epochs):
            print(f'Эпоха {epoch + 1} / {epochs}...')
            train_acc = []
            train_l = []

            batched_dataset_train_progress = tqdm(
                batched_dataset_train, unit=f' Б по {batch_size}'
            )

            # Проходимся по батчам --
            # вся логика обучения и классификации реализована внутри модели
            for images, labels in batched_dataset_train_progress:
                train_res = model({'images': images, 'labels': labels})
                train_acc.append(train_res[1])
                train_l.append(train_res[0].item())

                grid = make_grid(images)
                writer.add_image('latest_batch/train', grid, 0)

                batched_dataset_train_progress.set_postfix(
                    Точность=f'{int(train_res[1] * 100) / 100}'
                )

            writer.add_scalar('accuracy/train', np.mean(train_acc), epoch)
            writer.add_scalar('cross_entropy/train', np.mean(train_l), epoch)

            # Переходим к эвалюации...
            model.eval()
            eval_acc = []
            eval_l = []

            for images, labels in batched_dataset_eval:
                eval_res = model({'images': images, 'labels': labels, 'eval': True})
                eval_acc.append(eval_res[1])
                eval_l.append(eval_res[0].item())

                grid = make_grid(images)
                writer.add_image('latest_batch/eval', grid, 0)

            print(
                f'Точность после эвалюации: {np.mean(eval_acc)} (лучшая: {best_acc})\n'
            )
            writer.add_scalar('accuracy/eval', np.mean(eval_acc), epoch)
            writer.add_scalar('cross_entropy/eval', np.mean(eval_l), epoch)

            # ...и обратно в трейн
            model.train()
            model.scheduler.step()

            # Сохраняем веса, если они дают самую лучшую точность
            if best_acc < np.mean(eval_acc):
                best_acc = float(np.mean(eval_acc))
                best_epoch = epoch
                best_param = deepcopy(model.state_dict())

            # Если больше N эпох точность падает, сворачиваем лавочку
            elif (epoch - best_epoch) > max_degradation:
                print('\nМодель безнадёжно деградировала, ехать дальше смысла нет.')
                writer.flush()
                break
        if best_param is not None:
            model.load_state_dict(best_param)

    # Это надо чтобы по Ctrl+C код сперва сохранил модель
    except KeyboardInterrupt:
        print('\nВыполняется досрочное прерывание обучения...')
        if best_param is not None:
            model.load_state_dict(best_param)

    # В самом конце, записываем в тензорборд гиперпараметры и результаты
    writer.add_hparams(
        {'epochs': epochs, 'batch_size': batch_size, 'samples_size': samples_size},
        {'hparam/final_accuracy': best_acc, 'hparam/final_epoch': best_epoch},
    )

    return model, best_acc, best_epoch + 1
