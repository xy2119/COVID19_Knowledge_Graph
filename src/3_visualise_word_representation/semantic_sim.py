# coding=utf-8
from gensim.models import word2vec


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
word2class = {}
for i, (class_name, words_list) in enumerate(t_1):
    plt_words.extend(words_list)
    for _w in words_list:
        word2class[_w] = class_name


t = t0+plt_words
# 1. word represatation
word2vec_model = word2vec.Word2Vec.load('../word_representations/skip_gram/new_w2v.model')
a = word2vec_model.wv.similarity(t[0], t[1])

# 2. sim
res = []
for index, word in enumerate(t):
    if index == 0:
        continue
    score = word2vec_model.wv.similarity(t[0], word)
    res.append( (word, score) )
# sort
res = sorted(res, key=lambda y: y[1], reverse=True)
f = open('new_semantic_sim.txt', 'w', encoding='utf8')
print('closest semantic similarity covid-19: \n')
f.write('closest semantic similarity covid-19: \n')
for word, s in res:
    print(word+': '+word2class[word], '\t', s)
    f.write(word+': '+word2class[word]+'\t'+str(s) )
    f.write('\n')
f.close()
