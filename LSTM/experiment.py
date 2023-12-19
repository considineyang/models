import torch
import torch.nn as nn
from LSTM import LSTM


input_dim = 5
hidden_dim = 10
n_layers = 1

lstm_layer = LSTM(input_dim, hidden_dim, n_layers)


# size of input tensor = (batch size, sequence length, input dimension)
batch_size = 1
seq_len = 1

input_x = torch.randn(batch_size, seq_len, input_dim)
hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
cell_state = torch.randn(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)


out, hidden = lstm_layer(input_x, hidden)
print("Output's shape: ", out.shape)
print("Hidden vector ", hidden)

# The following is a simple example for using LSTM
# https://www.scaler.com/topics/pytorch/lstm-pytorch/
# Dataset is the Reddit clean jokes dataset
import torch
from torch import nn
import pandas as pd
from collections import Counter
import argparse
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# custom dataset class to load the data examples
class Dataset(Dataset):
    def __init__(self):
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        train_df = pd.read_csv('LSTM/jokes.csv')
        text = train_df['Joke'].str.cat(sep=' ')
        return text.split(' ')

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)


    # returns the length of the dataset
    def __len__(self):
        return len(self.words_indexes) - 4

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+4]),
            torch.tensor(self.words_indexes[index+1:index+4+1]))
    

# custom model class to define the model architecture
class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = len(dataset.uniq_words)

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )

        # lstm layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.1,
        )

        # fully connected layer
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

def train(dataset, model):
    model.train()

    dataloader = DataLoader(dataset, batch_size=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        state_h, state_c = model.init_state(4)

        for batch, (x, y) in enumerate(dataloader):

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print({ 'epoch number=': epoch, ' batch numer=': batch, ' loss value=': loss.item() })

def predict(dataset, model, text, next_words=50):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words


dataset = Dataset()
model = Model(dataset)

train(dataset, model)
print(predict(dataset, model, text='I like my coffee'))
