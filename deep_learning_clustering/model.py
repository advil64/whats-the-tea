import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TweetsDataset(Dataset):
    def __init__(self, tweets):
        self.tweets = tweets

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        return self.tweets[index]


class TweetClassifier(nn.Module):
    def __init__(self, out_channels=64, kernel_size=3, p=0.25, num_classes=42):
        super(TweetClassifier, self).__init__()
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

    def train_model(self, tweets, batch_size, device, criterion, optimizer, num_epochs):
        dataset = TweetsDataset(tweets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i, (inputs, topics) in enumerate(dataloader):
                inputs = inputs.to(device)
                topics = topics.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += topics.size(0)
                correct_predictions += (predicted == topics).sum().item()

                loss = criterion(outputs, topics)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)
            accuracy = (correct_predictions / total_predictions) * 100

            print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%')

    def predict(self, tweet_tensors, batch_size):
        dataset = TweetsDataset(tweet_tensors)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.eval()

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                outputs = self(batch)
                predicted = outputs.argmax(1)
                predictions.extend(predicted)

        return np.array(predictions)


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TweetClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model
