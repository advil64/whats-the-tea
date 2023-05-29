import torch
import torch.nn as nn


class TweetClassifier(nn.Module):
    def __init__(self, embed_dim=300, out_channels=64, kernel_size=3, stride=1, padding=0, p=0.25, num_classes=42):
        super(TweetClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.p = p
        self.num_classes = num_classes

        self.conv = nn.Conv1d(self.embed_dim, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(self.kernel_size, self.stride)
        self.dropout = nn.Dropout(self.p)
        self.fc = nn.Linear(self._linear_layer_in_size(), self.num_classes)

    def _linear_layer_in_size(self):
        conv_out_dim = (self.embed_dim + 2 * self.padding - self.kernel_size) // self.stride + 1
        pool_out_dim = (conv_out_dim - self.kernel_size) // self.stride + 1

        return pool_out_dim * self.out_channels

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TweetClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model
