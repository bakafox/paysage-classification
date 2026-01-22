from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.decomposition import PCA

from .dataset.samples import Sample
from .model.preprocess import SceneDataset

matplotlib.use('TkAgg')


# Перед использованием задать ss не больше 64, иначе крашнется!
# TODO: Использовать более подходящую логику визуализации вместо PCA
def visualize(
    samples: list[Sample],
    path_images: Path,
    labels_list: list[str],
    transforms,
):
    # Создаём экземпляры датасета и модели
    dataset = SceneDataset(
        samples,
        path_images,
        transforms,
    )

    counts = Counter(labels_list)

    print([f'Доля {label}: {cnt / len(labels_list)}' for label, cnt in counts.items()])

    images = []
    labels = []
    for image, label in dataset:
        images.append(image.numpy().reshape(-1))
        labels.append(label)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(images)

    import matplotlib.pyplot as plt

    # Выбираем палитру под кол-во классов
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels_list)))

    plt.figure(figsize=(8, 6))
    for idx, cls in enumerate(labels_list):
        plt.scatter(
            X_2d[np.array(labels) == idx, 0],
            X_2d[np.array(labels) == idx, 1],
            color=colors[idx],
            label=cls,
            alpha=0.7,
            edgecolors='k',
            linewidths=0.5,
        )

    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title('PCA визуализация с легендой по классам')
    plt.legend()
    plt.show()
