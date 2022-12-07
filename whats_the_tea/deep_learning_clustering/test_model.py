# %%
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import spacy
import math
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch import nn

tqdm.pandas()
encoder = LabelEncoder()

# %%
if torch.cuda.is_available():
  print('Good to go!')
else:
  print('Please set GPU via Edit -> Notebook Settings.')
device = torch.device('cuda:0')

# %%
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
        
        #Use if you've already generated spacy embeddings
        self.df = pd.read_json(file_path)

        # convert the embeddings to nd array
        self.df['vector'] = self.df['vector'].apply(lambda x: np.array(x))

        # seperate the embeddings and labels as series
        self.embeddings = self.df['vector']
        self.labels = self.df['num_cat']


    def generate_embeddings(self, file_path):

        self.df = pd.read_json(file_path, lines=True).drop(columns=['authors','link','date'])

        # change the dtype to category
        self.df = self.df.astype({'category': 'category'})
        self.df['num_cat'] = self.df['category'].cat.codes

        # append headline and description to get a new column
        self.df['selected_text'] =  self.df['headline'] + ' ' + self.df['short_description']

        # load the spacy model
        self.nlp = spacy.load("en_core_web_lg")
        self.df['vector'] = self.df['selected_text'].apply(lambda x: self.nlp(x).vector)

        # seperate the embeddings and labels as series
        self.embeddings = self.df['vector']
        self.labels = self.df['num_cat']
    
    def get_train(self):
        return zip(self.y_train, self.x_train)
    
    def get_test(self):
        return zip(self.y_test, self.x_test)
    
    def split_test_train(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.embeddings, self.labels, test_size=0.20)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

# %%
huffPo = MyDataset()
huffPo.load_embeddings('/common/users/shared/cs543_fall22_group3/huffpo/spacy_vectors.json')

# %%
huffPo.split_test_train()

# %%
category_mapping = {}

for c in range(42):
    category_mapping[c] = huffPo.df[huffPo.df['num_cat'] == c].iloc[0]['category']

# %%
# embeddings_pipeline = lambda x: float(x)
label_pipeline = lambda x: int(x)

# %%
# function to create batches of our data
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

# %%
class TextClassificationModel(nn.Module):

    def __init__(self, num_class, embed_dim=300, vocab_size=45, pad_index=0,
                 stride=1, kernel_size=3, conv_out_size=64, dropout_rate=0.25):
        super(TextClassificationModel, self).__init__()

        # Embedding layer parameters
        self.embed_size = embed_dim
        self.vocab_size = vocab_size
        self.pad_index = pad_index
       
        # Conv layer parameters
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_out_size = conv_out_size
       
        # Misc
        self.dropout_rate = dropout_rate
        
        self.embed_size = 1
        # Layers
        self.conv = torch.nn.Conv1d(self.embed_size, self.conv_out_size, self.kernel_size, self.stride)
        self.hidden_act = torch.relu
        self.max_pool = torch.nn.MaxPool1d(self.kernel_size, self.stride)
       
        self.flatten = lambda x: x.view(x.shape[0], x.shape[1]*x.shape[2])
       
        self.fc = torch.nn.Linear(self._linear_layer_in_size(), 42)

        if self.dropout_rate:
            self.dropout = torch.nn.Dropout(self.dropout_rate)

    def _linear_layer_in_size(self):
        out_conv_1 = ((self.embed_size - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)
                           
        # return out_pool_1*self.conv_out_size
        return 18944

    def forward(self, x):
        # print(x.shape)

        # x = torch.reshape(x. (x.shape[0],)

        x = torch.unsqueeze(x, 1)
        # x = torch.transpose(x, 1, 2) # (batch, 1, 300)

        x = self.conv(x)
        # print(x.shape)

        x = self.hidden_act(x)
        # print(x.shape)

        x = self.max_pool(x)
        # print(x.shape)

        if self.dropout_rate:
            x = self.dropout(x)

        x = self.flatten(x)
        # print(x.shape)

        x = self.fc(x)

        return x

# %%
m = TextClassificationModel(42)
m.load_state_dict(torch.load('/common/users/shared/cs543_fall22_group3/models/class_model.pt', map_location=device))
m.to(torch.device(device))

# %%
def evaluate(dataloader):
    m.eval()
    total_acc, total_count = 0, 0
    acc, pred = [], []

    with torch.no_grad():
        for idx, (label, vector) in enumerate(dataloader):
            predicted_label = m(vector)
            loss = criterion(predicted_label, label)

            # ground truth label
            acc = acc + list(label.cpu().detach().numpy())
            pred = pred + list(predicted_label.argmax(1).cpu().detach().numpy())

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    # return total_acc/total_count
    return acc, pred

# %%
from torchtext.data.functional import to_map_style_dataset

criterion = torch.nn.CrossEntropyLoss()
test_iter = huffPo.get_test()
test_dataset = to_map_style_dataset(test_iter)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)

# %%
acc, pred = evaluate(test_dataloader)

acc_labels = list(map(lambda x: category_mapping[x], acc))
pred_labels = list(map(lambda x: category_mapping[x], pred))

conf_matrix = confusion_matrix(acc_labels, pred_labels)

# %%
fig = px.imshow(conf_matrix)

fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = list(category_mapping.keys()),
        ticktext = list(category_mapping.values())
    ),
    yaxis = dict(
        tickmode = 'array',
        tickvals = list(category_mapping.keys()),
        ticktext = list(category_mapping.values())
    ),
    font=dict(
        size=7
    )
)
# plotly.io.write_image(fig, file='heat_map.png', scale=5)

fig.show()
# %%
