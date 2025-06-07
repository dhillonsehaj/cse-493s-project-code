import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch.utils
from evaluation_utils import *

'''
Logistic Regression
'''
class LogReg(object):
    def __init__(self):
        self._model = LogisticRegression()

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X, empathy_labels, rationale_labels):
        preds = self._model.predict(X)
        flat_acc = accuracy_score(empathy_labels, preds)
        f1 = f1_score(empathy_labels, preds, average='macro')
        # f1_rationale = compute_f1_rationale(preds, rationale_labels)
        # iou_rationale = iou_f1(preds, rationale_labels)
        return preds, flat_acc, f1  #, f1_rationale, iou_rationale

    def vectorizer(max_features):
        return TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, stop_words="english")

'''
RNN
'''
class TwoLayerRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2):
        super(TwoLayerRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        embed = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out[torch.arange(out.size(0)), lengths - 1]
        out = self.fc(out)
        return out

class RNNDataset(torch.utils.data.Dataset):
    def __init__(self, df, word2idx=None, max_len=50):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        if word2idx is None:
            self.word2idx = {"<pad>": 0, "<unk>": 1}
            idx = 2
            for _, row in df.iterrows():
                text = str(row["seeker_post"]) + " " + str(row["response_post"])
                for tok in text.strip().split():
                    if tok not in self.word2idx:
                        self.word2idx[tok] = idx
                        idx += 1
        else:
            self.word2idx = word2idx
        self.vocab_size = len(self.word2idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = (str(row["seeker_post"]) + " " + str(row["response_post"])).strip().split()
        idxs = [self.word2idx.get(tok, self.word2idx["<unk>"]) for tok in text]
        length = min(len(idxs), self.max_len)
        idxs = idxs[:self.max_len]
        pad_length = self.max_len - len(idxs)
        idxs = idxs + [0] * pad_length
        label = int(row["level"])
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(length, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def collate_rnn(batch):
        batch_sorted = sorted(batch, key=lambda x: x[1], reverse=True)
        idxs = torch.stack([item[0] for item in batch_sorted])
        lengths = torch.stack([item[1] for item in batch_sorted])
        labels = torch.stack([item[2] for item in batch_sorted])
        return idxs, lengths, labels

'''
HRED (omitted because of time constraint)
'''

'''
BERT (in train.py)
'''

'''
GPT-2 (in train.py)
'''

'''
DialoGPT (in train.py)
'''

'''
RoBERTa (in train.py)
'''

