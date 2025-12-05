import torchvision as tv
import torch
from tqdm import tqdm


class SceneClassifier(torch.nn.Module):
    def __init__(self, total_epochs, n_classes):
        super().__init__()

        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html
        self.model = tv.models.efficientnet_v2_s()
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs # Период косинуса (обычно total_epochs)
        )

        self.model = self.model.to(self.device)
        self.mode = 'train'

    def set_mode(self, mode):
        assert mode in ['train', 'test']
    
    def toggle_mode(self):
        self.mode = 'test' if self.mode == 'train' else 'train'

    def train(self, is_eval = False): # type: ignore
        self.mode = 'train' if not is_eval else 'eval'
    
    def eval(self): # type: ignore
        self.mode = 'eval'
    
    def test(self): # type: ignore
        self.mode = 'test'

    def forward(self, batch):
        images = batch['images'].to(self.device)
        labels = batch['labels'].to(self.device)

        if self.mode == 'train':
            self.optimizer.zero_grad() # Обнуляем прошлые градиенты

            output = self.model(images)
            labels = labels.long()
            loss = self.criterion(output, labels)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            accuracy = (torch.argmax(output, dim=1) == labels).sum().item() / len(labels)
            return accuracy

        elif self.mode in ['eval', 'test']:
            try:
                logits = self.model(images)

                # Выполняем нормализацию значений и выбираем наибольшее
                probas = torch.nn.functional.softmax(logits)
                predicted_labels = torch.argmax(probas, dim=1)

                accuracy = (predicted_labels == labels).sum().item() / len(labels)
                return accuracy

            except BaseException:
                img = batch
                output = self.model(img)

                predicted_labels = torch.argmax(output, dim=1)
                return predicted_labels

    def predict(self, images):
        self.model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            logits = self.model(images)

            # Выполняем нормализацию значений и выбираем наибольшее
            probas = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probas, dim=1)

        return preds.cpu()
