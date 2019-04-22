import numpy as np
import tensorflow as tf
from gensim import models


def load_embedding(session, vocab, path, dim_embedding, vocab_size):
    """
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding
      vocab_size     Vocabulary size of the corpus
    """

    print(f'Loading external embeddings from {path}')

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            external_embedding[idx] = np.random.uniform(low=-0.25,
                                                        high=0.25,
                                                        size=dim_embedding)

    print(f'{matches} words out of {vocab_size} could be loaded')

    return external_embedding
