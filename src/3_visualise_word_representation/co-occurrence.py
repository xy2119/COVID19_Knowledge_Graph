# coding=utf-8
import numpy as np
from gensim.models import word2vec
from tqdm import tqdm

# 0. word
t0 = ['covid-19']
# t1 = ['fever', 'cough', 'asthma', 'hypothermia', 'hemoptysis', 'sinusitis']
# t2 = ['alzheimer', 'headache', 'ataxia', 'meningitis', 'encephalitis']
# t3 = ['hemangioma', 'thrombotic', 'leukemia', 'lymphoma', 'myeloma', 'anemia']


# 1. embedding
word2vec_model = word2vec.Word2Vec.load('../word_representations/skip_gram/new_w2v.model')
all_words = list(word2vec_model.wv.index_to_key)


# 2. word
t_1 = []
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
            t_1.append( (class_name, _tmp) )
for i in t_1:
    print(i)
plt_words =[]
for i, (class_name, words_list) in enumerate(t_1):
    plt_words.extend(words_list)





# 3. contruct matrix
all_t = t0+plt_words
matrix = np.zeros([ len(all_t), len(all_t) ], dtype=np.int)



with open('../bio_titles_abstracts.txt', 'r', encoding='utf8') as f:
    for line in tqdm(f):
        _line = line.lower().strip().split()
        for i, word_i in enumerate(_line[:-1]):
            if word_i in all_t:
                for j, word_j in enumerate(_line[i+1:]):
                    if word_j in all_t:
                        row = all_t.index(word_i)
                        col = all_t.index(word_j)
                        matrix[row][col] += 1
                        matrix[col][row] += 1

print(matrix)

np.savetxt('co-occurrence.csv', matrix, fmt=('%s,'*len(matrix))[:-1], header=','.join(all_t), comments='')