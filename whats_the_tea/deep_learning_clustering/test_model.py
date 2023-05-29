from model import TextClassificationModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchtext.data.functional import to_map_style_dataset
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import plotly.express as px
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

category_mapping = {c: huffPo.df.loc[huffPo.df['num_cat'] == c, 'category'].iloc[0] for c in range(42)}

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


def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0
    acc, pred = [], []

    with torch.no_grad():
        for idx, (label, vector) in enumerate(dataloader):
            predicted_label = model(vector)
            loss = criterion(predicted_label, label)

            acc.extend(label.cpu().detach().numpy())
            pred.extend(predicted_label.argmax(1).cpu().detach().numpy())

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    return acc, pred


num_classes = len(set([label for (label, text) in train_iter]))

model = TextClassificationModel().to(device)
criterion = nn.CrossEntropyLoss()
train_iter = huffPo.get_train()
test_iter = huffPo.get_test()
test_dataset = to_map_style_dataset(test_iter)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)

acc, pred = evaluate(test_dataloader, model, criterion)

acc_labels = list(map(lambda x: category_mapping[x], acc))
pred_labels = list(map(lambda x: category_mapping[x], pred))

conf_matrix = confusion_matrix(acc_labels, pred_labels)

fig = px.imshow(conf_matrix)

fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=list(category_mapping.keys()),
        ticktext=list(category_mapping.values())
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=list(category_mapping.keys()),
        ticktext=list(category_mapping.values())
    ),
    font=dict(
        size=7
    )
)

fig.show()
