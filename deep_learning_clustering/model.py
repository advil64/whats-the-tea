import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return self.tensors[index], self.labels[index]


class TestDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return self.tensors[index]


class Classifier(nn.Module):
    def __init__(self, out_channels=64, kernel_size=3, p=0.25, num_classes=42):
        super(Classifier, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.p = p
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(1, self.out_channels, self.kernel_size)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(self.kernel_size)
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size)
        self.dropout = nn.Dropout(self.p)
        self.fc = nn.Linear(32 * self.out_channels, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def train_model(self, tensors, labels, batch_size, device, criterion, optimizer, num_epochs):
        dataset = TrainDataset(tensors, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()

        for epoch in range(num_epochs):
            correct_predictions = 0
            total_predictions = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            accuracy = (correct_predictions / total_predictions) * 100

            print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%')

    def predict(self, tensors, batch_size):
        dataset = TestDataset(tensors)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.eval()

        predictions = []
        with torch.no_grad():
            for inputs in dataloader:
                outputs = self(inputs)
                predicted = outputs.argmax(1)
                predictions.extend(predicted)

        return np.array(predictions)


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Classifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model
