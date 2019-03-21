"""
This scripts preprocesses given data by

1. Adding <bos> and <eos> tags,
2. Remove lines with more than 30 tokens,
3. Padding short sentences with <pad>, 
4. Replaces uncommon words by <unk>, and

save the new data in *.proc format.
"""

import os
from collections import Counter


def preprocess(file):
    with open(file) as f:
        lines = f.readlines()

        # add <bos> and <eos> tags
        lines = ("<bos> " + line.strip() + " <eos>\n" for line in lines)
        # remove lines with length greater than 30
        lines = (line for line in lines if len(line.split(' ')) <= 30)
        # pad short sentences to 30 tokens
        lines = ((line.strip() + " <pad>" * (30 - len(line.split(' ')))
                  ).strip() + '\n' for line in lines)

    with open(file + '.proc.tmp', 'w+') as f:
        f.writelines(lines)


for file in ['sentences.train', 'sentences.eval']:
    preprocess(file)


with open('sentences.train.proc.tmp') as f:
    word_cnts = Counter()
    for line in f:
        word_cnts.update(line.replace('\n', ' ').split(' '))

# using 20k-1 because <unk> is not in the list
keeps = set(map(lambda x: x[0], word_cnts.most_common(19999)))

assert set(['<bos>', '<eos>', '<pad>']) <= keeps

for file in ['sentences.train.proc.tmp', 'sentences.eval.proc.tmp']:
    with open(file, 'r') as f:
        lines = (' '.join(word if word in keeps else '<unk>' for word in line.strip(
        ).split(' ')) + '\n' for line in f)
        with open(file[:-4], 'w') as f:
            f.writelines(lines)

os.remove('sentences.train.proc.tmp')
os.remove('sentences.eval.proc.tmp')

print('Done.')
