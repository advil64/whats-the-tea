import torch
import torch.nn as nn


class TextClassificationModel(nn.Module):
    def __init__(self, embed_dim=1, out_channels=64, kernel_size=3, stride=1, padding=0, p=0.25, num_classes=42):
        super(TextClassificationModel, self).__init__()
        # Embedding layer parameters
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.padding = padding

        # Conv layer parameters
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        # Dropout layer parameters
        self.p = p

        # Layers
        self.conv = nn.Conv1d(self.embed_dim, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(self.kernel_size, self.stride)
        self.fc = nn.Linear(self._linear_layer_in_size(), self.num_classes)

        if self.p:
            self.dropout = nn.Dropout(self.p)

    def _linear_layer_in_size(self):
        conv_out_dim = (self.embed_dim - self.kernel_size + 1) // self.stride
        pool_out_dim = (conv_out_dim - self.kernel_size + 1) // self.stride

        return 18944  # pool_out_dim * self.out_channels

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = x.flatten(start_dim=1)

        if self.p:
            x = self.dropout(x)

        x = self.fc(x)

        return x


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TextClassificationModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model
