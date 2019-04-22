"""
The preprocess function modifies data by:

1. Adding <bos> and <eos> tags,
2. Removing lines with more than 30 tokens,
3. Padding short sentences with <pad>,
4. Saving the result as *.proc.tmp files.

The tokenize function:

1. Creates word-to-index and index-to-word mappings,
2. Saves them as *.pkl files,
3. Replaces uncommon words by <unk>,
4. Saves the result as *.proc files,
5. Saves the tokenized data as *.csv files,
6. Deletes the old *.proc.tmp files.

The reconstruct function:

1. Should be called after we obtain predicted *.csv files,
2. Uses previously constructed mappings to reconstruct text from indices.

"""

import filecmp
import os
import pickle

import pandas as pd

from collections import Counter


def preprocess(file):
    with open(file) as f:
        lines = f.readlines()

        # add <bos> and <eos> tags
        lines = ('<bos> ' + line.strip() + ' <eos>\n' for line in lines)
        # remove lines with length greater than 30
        lines = (line for line in lines if len(line.split(' ')) <= 30)
        # pad short sentences to 30 tokens
        lines = ((line.strip() + ' <pad>' * (30 - len(line.split(' ')))
                ).strip() + '\n' for line in lines)

    with open(file + '.proc.tmp', 'w+') as f:
        f.writelines(lines)

def tokenize():
    with open('sentences.train.proc.tmp') as f:
        word_cnts = Counter()
        for line in f:
            word_cnts.update(line.replace('\n', ' ').split(' '))

    # using 20k-1 because <unk> is not in the list
    keeps = set(map(lambda x: x[0], word_cnts.most_common(19999)))
    assert set(['<bos>', '<eos>', '<pad>']) <= keeps

    # save the mappings while leaving index 0 for <unk>
    word_2_idx = {w:i+1 for i, w in enumerate(keeps)}
    idx_2_word = {i+1:w for i, w in enumerate(keeps)}

    # mapping for token <unk>
    word_2_idx['<unk>'] = 0
    idx_2_word[0] = '<unk>'

    with open('word_2_idx.pkl', 'wb') as f:
        pickle.dump(word_2_idx, f)

    with open('idx_2_word.pkl', 'wb') as f:
        pickle.dump(idx_2_word, f)

    for file in ['sentences.train.proc.tmp', 'sentences.eval.proc.tmp']:
        with open(file, 'r') as f:
            lines = (' '.join(word if word in keeps else '<unk>' for word in line.strip(
            ).split(' ')) + '\n' for line in f)
            with open(file[:-4], 'w') as f:
                f.writelines(lines)

        with open(file, 'r') as f:
            tokens = [[word_2_idx[word] if word in word_2_idx else 0 for word in line.strip(
            ).split(' ')] for line in f]
            df_tokens = pd.DataFrame(tokens)
            df_tokens.to_csv(file[:-4] + '.csv', header=False, index=False)

    os.remove('sentences.train.proc.tmp')
    os.remove('sentences.eval.proc.tmp')

def reconstruct(csv_file):
    with open('idx_2_word.pkl', 'rb') as f:
        idx_2_word = pickle.load(f)

    df = pd.read_csv(csv_file, header=None)

    lines = (' '.join(idx_2_word[idx] if idx in idx_2_word else '<unk>' for idx in list(row)) + '\n'
        for _, row in df.iterrows())

    with open(csv_file[:-4] + '.out', 'w') as f:
        f.writelines(lines)

if __name__ == '__main__':
    for file in ['sentences.train', 'sentences.eval']:
        preprocess(file)
    print('Finished preprocessing.')
    tokenize()
    print('Finished tokenizing.')
    reconstruct('sentences.eval.proc.csv')
    assert filecmp.cmp('sentences.eval.proc', 'sentences.eval.proc.out')
    print('Finished reconstructing.')
