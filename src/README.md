
<h2 id="folder-structure"> Folder Structure</h2>

    code
    .           
    ├── 0_parse_the_data                        # Part 0 : Parse the Data                                  
    │   ├── parse_the_data.py                   # extract document titles and abstract, output > bio_titles.txt
    │   └── README.md    
    │
    ├── 1_tokenization                          # Part 1 : Tokenization    
    │   ├── tokenization.py                     # tokenized the text by creating top100k token list, output > *_tokens.txt 
    │   ├── tokenizer-bio.json                  
    │   ├── split_tokens.txt
    │   ├── bpe_tokens.txt
    │   ├── nltk_tokens.txt
    │   ├── scispacy_tokens.txt
    │   ├── bert-base-uncased-vocab.txt
    │   └── README.md    
    │
    ├── 2_word_representation                   # Part 2 : Build Word Representations             
    │   ├── n-gram                              
    │   │   ├── n_gram.py                       # create word embedding through n-gram
    │   │   ├── n_gram_word2id.txt             
    │   │   └── README.md 
    │   │
    │   └── skip_gram                           
    │       ├── skip_gram.py                    # create word embedding through skip-gram
    │       ├── new_w2v.model                   # skip-gram model
    │       └── README.md
    │
    ├── 3_visualise_word_representation         # Part 3 : Explore the Word Representations                    
    │   ├── t_sne.py                            # visualised embeddings through t-sne, output > t_sne.png
    │   ├── bio_t_sne.py  
    │   ├── co-occurrence.py                    # find entities that co-occur with Covid 19, output > co-occurrence.csv
    │   ├── co-occurrence.csv                   # co-occurrence output 
    │   ├── semantic_sim.py                     # find entities that semantically similar with Covid 19, output > semantic_sim.txt
    │   ├── semantic_sim.txt
    │   ├── bio_dict                            # dict of biomedical entities for mapping
    │   └── README.md    
    │ 
    
