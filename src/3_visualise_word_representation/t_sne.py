# coding=utf-8
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from gensim.models import word2vec


random.seed(99)

# 1. embedding
word2vec_model = word2vec.Word2Vec.load('../word_representations/skip_gram/w2v.model')
all_words = list(word2vec_model.wv.index_to_key)
words = random.sample(all_words, 50)

arr = np.empty((0, 128), dtype='f')

for word in words:
    vec = word2vec_model.wv[word]
    arr = np.append(arr, np.array([vec]), axis=0)

def display_closestwords_tsnescatterplot():

    tsne = manifold.TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(words, x_coords, y_coords):
        # plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.savefig('t_sne.png')
    plt.show()

display_closestwords_tsnescatterplot()