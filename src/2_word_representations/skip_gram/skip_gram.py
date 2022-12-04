# coding=utf-8
import multiprocessing
from gensim.models import word2vec

# 1. 预处理数据
def make_new_data():
    f_w = open('new_titles_abstracts.txt', 'w', encoding='utf8')
    with open('../../bio_titles_abstracts.txt', 'r', encoding='utf8') as f:
        for line in f:
            f_w.write(line.lower())

# 2. 训练
def train():
    # Settings
    seed = 666
    sg = 1
    window_size = 5
    vector_size = 128
    min_count = 5
    epochs = 5
    batch_words = 10000

    train_data = word2vec.LineSentence('new_titles_abstracts.txt')
    '''
        seed： 亂數種子
        LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
        sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
        window_size： 周圍詞彙要看多少範圍
        vector_size： 轉成向量的維度
        min_count： 詞頻少於 min_count 之詞彙不會參與訓練
        workers： 訓練的並行數量
        epochs： 訓練的迭代次數
        batch_words：每次給予多少詞彙量訓練
    '''
    model = word2vec.Word2Vec(
        train_data,
        min_count=min_count,
        vector_size=vector_size,
        workers=multiprocessing.cpu_count(),
        epochs=epochs,
        window=window_size,
        sg=sg,
        seed=seed,
        batch_words=batch_words,
    )

    model.save('new_w2v.model')

    #不以C语言可以解析的形式存储词向量
    model.wv.save_word2vec_format('new_w2v.vector', binary=False)


# 3. 测试
def test():
    word2vec_model = word2vec.Word2Vec.load('new_w2v.model')
    testwords = ['fever','cough','covid-19']
    for index, word in enumerate(testwords):
        res = word2vec_model.wv.most_similar(word)
        print(word)
        print(res)

# 4. 词向量保存
def save_emb():
    word2vec_model = word2vec.Word2Vec.load('new_w2v.model')
    all_words = list(word2vec_model.wv.index_to_key)
    with open('new_skip_gram_embedding.txt', 'w', encoding='utf8') as f:
        for word in all_words:
            f.write(word+'\t'+str(word2vec_model.wv[word]))


if __name__ == '__main__':
    # 1. make data
    make_new_data()

    # 2. train
    train()

    # 3. test
    test()

    #
    save_emb()