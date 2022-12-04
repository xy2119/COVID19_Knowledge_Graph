# coding=utf-8

import os
import json
from tqdm import tqdm


# make file record all titles
titles_file = '../bio_titles.txt'
f_titles = open(titles_file, 'w', encoding='utf8')
count = 0 # count title numbers

# list all json files
for root, dirs, files in os.walk("document_parses", topdown=False):
    n = 0
    for name in tqdm(files):
        # current json file path
        json_file = os.path.join(root, name)
        # extract title form json file
        with open(json_file, 'r', encoding='utf8') as load_f:
            load_dict = json.load(load_f)
            title = load_dict['metadata']['title']
            title = title.strip() + '\n'
            f_titles.write(title)
        count += 1 # count titles number
        # show example of recorded title
        if n<3:
            n += 1
            print(title)


# close file
f_titles.close()
print(count) # 425257