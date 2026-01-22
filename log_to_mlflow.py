from pathlib import Path

import hydra
import mlflow.pyfunc
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from paysage_classification.inference import inference


# Описываем обертку
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, label_list, ss, infer_output):
        self.labels_list = label_list
        self.ss = ss
        self.output_mode = infer_output

    def load_context(self, context):
        # Импорты внутри, чтобы избежать ошибок сериализации
        self.model = torch.load(
            context.artifacts["model_path"], map_location="cpu", weights_only=False
        )
        self.model.eval()

        self.transform_list = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([self.ss, self.ss]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, context, model_input):
        # Извлекаем путь к картинке
        if isinstance(model_input, pd.DataFrame):
            img_path = str(model_input.iloc[0, 0])
        else:
            img_path = str(model_input)

        try:
            inference(
                self.model,
                img_path,
                self.labels_list,
                self.transform_list,
                output=self.output_mode,
            )
            result = 10
            # Если твоя функция только принтит, придется возвращать кастомную строку
            if result is None:
                return ["Check server console (inference function returned None)"]
            return [result]
        except Exception as e:
            return [f"Error during inference: {str(e)}"]


@hydra.main(version_base=None, config_path='', config_name='config')
def run_mlflow(cfg: DictConfig):
    params = OmegaConf.to_container(cfg['params'])

    # Считаем входные метки, чтобы не делать этого по многу раз
    labels_f = open(params['path-labels'], 'r', encoding='utf-8')
    labels_list = list(map(lambda x: x.replace('\n', ''), labels_f.readlines()))

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="landscape_server",
            python_model=ModelWrapper(
                labels_list, params['ss'], params['infer-output']
            ),
            artifacts={
                "model_path": str(
                    Path(params['path-models']) / params['path-checkpoint']
                )
            },
            code_paths=[
                "paysage_classification"
            ],  # Убедись, что папка рядом со скриптом
        )
        print(f"\nSuccess! RUN_ID: {run.info.run_id}")
        print("Команда для запуска сервера:")
        print(
            "poetry run mlflow models serve",
            f"-m runs:/{run.info.run_id}/landscape_server -p 5001 --no-conda",
        )


# Запуск процесса логирования
if __name__ == "__main__":
    run_mlflow()
