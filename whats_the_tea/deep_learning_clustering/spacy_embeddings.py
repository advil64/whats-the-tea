from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn

tqdm.pandas()
encoder = LabelEncoder()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class MyDataset(Dataset):
    def __init__(self):
        self.df = None

        self.embeddings = None
        self.labels = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_embeddings(self, file_path):
        # Use if you've already generated spacy embeddings
        self.df = pd.read_json(file_path)
        # Convert the embeddings to nd array
        self.df['vector'] = self.df['vector'].apply(np.array)
        # Separate the embeddings and labels as series
        self.embeddings = self.df['vector']
        self.labels = self.df['num_cat']

    def get_train(self):
        return zip(self.y_train, self.x_train)

    def get_test(self):
        return zip(self.y_test, self.x_test)

    def split_test_train(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.embeddings, self.labels,
                                                                                test_size=0.20)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


huffPo = MyDataset()
huffPo.load_embeddings('/common/users/shared/cs543_fall22_group3/huffpo/spacy_vectors.json')
huffPo.split_test_train()

label_pipeline = lambda x: int(x)


# Function to create batches of data
def collate_batch(batch):
    label_list, embedding_list = [], []

    for (_label, _embedding) in batch:
        label_list.append(label_pipeline(_label))
        embedding = torch.tensor(_embedding, dtype=torch.float32)
        embedding_list.append(embedding)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    embedding_list = torch.stack(embedding_list)

    return label_list.to(device), embedding_list.to(device)


train_iter = huffPo.get_train()
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)


class TextClassificationModel(nn.Module):
    def __init__(self, embed_dim=1, out_channels=64, kernel_size=3, stride=1, padding=0, p=0.25, num_classes=42):
        super(TextClassificationModel, self).__init__()
        # Embedding layer parameters
        self.num_classes = num_classes
        self.embed_size = embed_dim
        self.padding = padding

        # Conv layer parameters
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        # Dropout layer parameters
        self.p = p

        # Layers
        self.conv = nn.Conv1d(self.embed_size, self.out_channels, self.kernel_size, self.stride, self.padding)
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
        x = self.fc(x)

        if self.p:
            x = self.dropout(x)

        return x


def train(dataloader, criterion, optimizer, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, vector) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(vector)
        loss = criterion(predicted_label, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                f"| epoch {epoch:3d} | {idx:5d}/{len(dataloader):5d} batches | accuracy {total_acc / total_count:8.3f}")
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, vector) in enumerate(dataloader):
            print(label, vector)
            predicted_label = model(vector)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    return total_acc / total_count


# Hyperparameters
EPOCHS = 20  # epoch
LR = 0.1  # learning rate
BATCH_SIZE = 64  # batch size for training

num_classes = len(set([label for (label, text) in train_iter]))

model = TextClassificationModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter = huffPo.get_train()
test_iter = huffPo.get_test()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader, criterion, optimizer, epoch)
    accu_val = evaluate(valid_dataloader, model, criterion)

    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val

    print('-' * 59)
    print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | valid accuracy {accu_val:8.3f}')
    print('-' * 59)

test_iter = huffPo.get_test()
test_dataset = to_map_style_dataset(test_iter)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

print(evaluate(test_dataloader, model, criterion))
