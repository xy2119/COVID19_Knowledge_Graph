# coding=utf-8
import collections
import numpy as np
import torch
from torch import nn, optim
from nltk import ngrams
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# 0. set parameters
N_gram = 6
EMBEDDING_DIM = 128 
VOCAB_SIZE = 100000 
BATCH_SIZE = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. create data
train_data = []
tokens = []
with open('../../bio_titles.txt', 'r', encoding='utf8') as f:
    for line in f:
        _line = line.strip().lower()
        # count frequency of words
        tokens.extend(_line.split())
        n_grams_data = ngrams(_line.split(), N_gram)
        for grams in n_grams_data:
            train_data.append( ( (grams[:-1]) , grams[-1]) )
# create count dict, keep top 100k words
counts_dict = dict((collections.Counter(tokens).most_common(VOCAB_SIZE-1)))
counts_dict['UNK']=len(tokens)-np.sum(list(counts_dict.values()))

idx_to_word = []
for word in counts_dict.keys():
    idx_to_word.append(word)
word_to_idx = {word:i for i,word in enumerate(idx_to_word)}

with open('n_gram_word2id.txt', 'w', encoding='utf8') as f:
    for key, value in word_to_idx.items():
        f.write(key + '\t' + str(value) + '\n')

class CustomDataset(Dataset):
    def __init__(self, data, word_to_idx):
        super(CustomDataset, self).__init__()
        self.data = data
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        words, _label = self.data[item]
        inputs = []
        for word in words:
            if word in word_to_idx:
                index = word_to_idx[word]
            else:
                index=word_to_idx['UNK']
            inputs.append(index)
        if _label in word_to_idx:
            label = word_to_idx[_label]
        else:
            label = word_to_idx['UNK']
        return torch.tensor(inputs), label

# create  dataset and dataloader
dataset = CustomDataset(train_data, word_to_idx)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# 2.create model
class n_gram(nn.Module):
    def __init__(self, vocab_size, context_size, n_dim):
        super(n_gram, self).__init__()

        self.embed = nn.Embedding(vocab_size, n_dim)   # (vocab_size,n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, vocab_size)
        )

    def forward(self, x):
        voc_embed = self.embed(x)  # context_size*n_dim
        voc_embed = voc_embed.view(x.shape[0], -1)  # concat 2 word embedding  1*(context_size*n_dim)
        out = self.classify(voc_embed)   # 1*vocab_size
        return out

    def input_embeddings(self):
        return self.embed.weight.data.cpu().numpy()


# 3. train
CONTEXT_SIZE = N_gram-1
model = n_gram(len(word_to_idx), CONTEXT_SIZE, EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# to device
model = model.to(device)

for epoch in range(5):
    train_loss = 0
    for word, label in tqdm(dataloader):
        word, label = word.to(device), label.to(device)
        # forward
        out = model(word)
        loss = criterion(out, label)
        train_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch: {}, Loss: {:.6f}'.format(epoch + 1, train_loss / len(train_data)))
    # save embedding
    embedding_weights = model.input_embeddings()
    np.save("n_gram-embedding-{}.npz".format(EMBEDDING_DIM), embedding_weights)


for test_word in ['cough', 'fever', 'covid-19']:
    print(test_word)
    print(embedding_weights[word_to_idx[test_word]])
