from typing import Callable, Literal

import matplotlib

from .model.preprocess import process_image

matplotlib.use('TkAgg')


def inference(
    model,
    path_image,
    labels_list: list[str],
    transforms: Callable,
    output: Literal['mpl', 'std'],
):
    tensor = process_image(path_image, transforms)
    if tensor is None:
        return
    tensor = tensor.unsqueeze(0)

    (preds, probas) = model.predict(tensor)

    # Вывод в stdout (можно через pipe перенаправить в файл)
    if output == 'std':
        print(f'\n{path_image}')
        for i, p in enumerate(probas.tolist()[0]):
            print(f'{labels_list[i]} -- {p:.20f}')
        # print()

    # Вывод в виде интерактивного окна MatPlotLib, для демки
    else:
        import matplotlib.pyplot as plt  # локальный импорт
        from PIL import Image

        with Image.open(path_image) as img:
            img_rgb = img.convert('RGB')  # На всякий случай

            probas_with_labels = ', '.join(
                [
                    f'{labels_list[i]} - {p * 100:.2f}%'
                    for i, p in enumerate(probas.tolist()[0])
                ]
            )

            plt.imshow(img_rgb)
            plt.suptitle(labels_list[preds.tolist()[0]])
            plt.title(probas_with_labels, size='small', wrap=True)

            plt.show()
