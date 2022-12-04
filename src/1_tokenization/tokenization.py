# coding=utf-8
import os
from tqdm import tqdm
from collections import Counter

# Track 2.1: Use split()
def split_tokenization():
    tokens = []
    with open('../bio_titles.txt', 'r') as f:
        for line in f:
            token_list = line.strip().split()
            tokens.extend(token_list)
    tokens_counter = Counter(tokens)
    # top 100k token
    tokens_counter = tokens_counter.most_common(100000)
    # write
    f = open('split_tokens.txt', 'w', encoding='utf8')
    for token, num in tokens_counter:
        f.write(token + '\t' + str(num) + '\n')
    f.close()

# Track 2.2: Use NLTK or SciSpaCy
def nltk_ssc_tokenization( mode='nltk'):
    if 'nltk' in mode.lower():
        import nltk
        nltk.download('punkt')
        from nltk.tokenize import word_tokenize
        tokens = []
        with open('../bio_titles.txt', 'r') as f:
            for line in tqdm(f):
                line = line.strip()
                words = word_tokenize(line)
                tokens.extend(words)
        tokens_counter = Counter(tokens)
        # top 100000 token
        tokens_counter = tokens_counter.most_common(100000)
        # write
        f = open('nltk_tokens.txt', 'w', encoding='utf8')
        for token, num in tokens_counter:
            f.write(token + '\t' + str(num) + '\n')
        f.close()
    elif 'scispacy' in mode.lower(): # 1:10:03
        import spacy
        nlp = spacy.load("en_core_sci_sm")
        tokens = []
        with open('../bio_titles.txt', 'r') as f:
            for line in tqdm(f):
                line = line.strip()
                doc = nlp(line)
                words = [token.orth_ for token in doc if not token.is_punct | token.is_space]
                tokens.extend(words)
        tokens_counter = Counter(tokens)
        # top 100k token
        tokens_counter = tokens_counter.most_common(100000)
        # write
        f = open('scispacy_tokens.txt', 'w', encoding='utf8')
        for token, num in tokens_counter:
            f.write(token + '\t' + str(num) + '\n')
        f.close()

# Track 2.3: Use Byte-Pair Encoding (BPE)
def bpe_tokenization():
    from tokenizers import BertWordPieceTokenizer
    tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
    tokens = []
    with open('../bio_titles.txt', 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            output = tokenizer.encode(line)
            tokens.extend(output.tokens)
    tokens_counter = Counter(tokens)
    # top 100k token
    tokens_counter = tokens_counter.most_common(100000)
    # write
    f = open('bpe_tokens.txt', 'w', encoding='utf8')
    for token, num in tokens_counter:
        f.write(token + '\t' + str(num) + '\n')
    f.close()

# Track 2.4: Build new Byte-Pair Encoding (BPE)
def new_bpe_tokenization():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    # with open('../bio_titles.txt', 'r') as f:
    #     lines = []
    #     for line in f:
    #         lines.append(line.strip())
    if os.path.exists('tokenizer-bio.json'):
        tokenizer = Tokenizer.from_file("tokenizer-bio.json")
        output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
        print(output.tokens)
    else:
        tokenizer.train(files=['../bio_titles.txt'], trainer=trainer)
        # save tokenizer
        tokenizer.save("tokenizer-bio.json")

if __name__ == '__main__':
    # Track 2.1: Use split()
    # split_tokenization()

    # Track 2.2: Use NLTK or SciSpaCy
    # nltk_ssc_tokenization( mode='NLTK')
    # nltk_ssc_tokenization( mode='SciSpaCy') # 1:10:03

    # Track 2.3: Use Byte-Pair Encoding (BPE)
    # bpe_tokenization()

    # Track 2.4: Build new Byte-Pair Encoding (BPE)
    new_bpe_tokenization()



