import torch
import torch.nn as nn


class TweetClassifier(nn.Module):
    def __init__(self, out_channels=64, kernel_size=3, p=0.25, num_classes=42):
        super(TweetClassifier, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.p = p
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(1, self.out_channels, self.kernel_size)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool1d(self.kernel_size)
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool1d(self.kernel_size)
        self.dropout = nn.Dropout(self.p)
        self.fc = nn.Linear(32 * self.out_channels, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TweetClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model
