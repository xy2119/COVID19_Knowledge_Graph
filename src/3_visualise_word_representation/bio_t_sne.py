# coding=utf-8
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from gensim.models import word2vec
from sklearn.cluster import KMeans
import seaborn as sns

random.seed(99)


# 1. embedding
word2vec_model = word2vec.Word2Vec.load('../word_representations/skip_gram/new_w2v.model')
all_words = list(word2vec_model.wv.index_to_key)


# 2. word
t = []
with open('../data/bio_dict', 'r' , encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if line:
            class_name, words = line.split('=')[0][2:], line.split('=')[1]
            _t = eval(words)
            _tmp = []
            for word in _t:
                word = word.lower()
                if word in all_words:
                    _tmp.append(word)
            t.append( (class_name, _tmp) )
for i in t:
    print(i)
colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'grey', 'darkred', 'sienna', 'darkkhaki', 'teal','pink', 'darkblue',' indigo']
plt_words =[]
words2color = {}
for i, (class_name, words_list) in enumerate(t):
    plt_words.extend(words_list)
    for _w in words_list:
        words2color[_w] = colors[i]


# class_name, color_name
class_name_list, color_name_list = [], []
for i, (class_name, words_list) in enumerate(t):
    class_name_list.append(class_name)
    color_name_list.append(colors[i])

# words = random.sample(all_words, 50)

arr = np.empty((0, 128), dtype='f')
for word in plt_words:
    vec = word2vec_model.wv[word]
    arr = np.append(arr, np.array([vec]), axis=0)

def display_closestwords_tsnescatterplot():
    # 分辨率参数-dpi，画布大小参数-figsize
    plt.figure(dpi=300,figsize=(32,32))

    tsne = manifold.TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    sca = [None for _ in range(len(class_name_list))]
    # print(sca)
    # exit()
    #

    for label, x, y in zip(plt_words, x_coords, y_coords):
        color = words2color[label]

        index = color_name_list.index(color)
        sca[index] = plt.scatter(x, y, c=color)
        # plt.scatter(x, y, c=color)
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    # plt.legend()
    plt.legend(
        tuple(sca),
        tuple(class_name_list),
        loc = 'best'
    )
    plt.title('Bio_Visualisation',fontsize=30)
    plt.savefig('bio_t_sne.png')
    # plt.show()
    k = 15
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(Y)

    # sns settings
    sns.set(rc={'figure.figsize':(32,32)})
    # colors
    palette = sns.hls_palette(k, l=.4, s=.9)
    # plot
    sns.scatterplot(x_coords,y_coords, hue=y_pred, legend='full', palette=palette, alpha=0.7)
    plt.title('Bio_Visualisation with KMeans Labels',fontsize=30)
    for word, (x,y) in zip(plt_words, Y ):
            plt.text(x+0.005, y+0.005, word)
    
    plt.savefig("KMeans_bio_t_sne.png")
    

display_closestwords_tsnescatterplot()