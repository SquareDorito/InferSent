from random import randint
import matplotlib

import numpy as np
import torch

GLOVE_PATH = '../dataset/GloVe/glove.840B.300d.txt'
model = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
model.use_cuda = False

model.set_glove_path(GLOVE_PATH)
model.build_vocab_k_words(K=500000)

sentences = []
with open('samples.txt') as f:
    for line in f:
        sentences.append(line.strip())

embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))
print('')
print('=============== EMBEDDED VECTORS ===============')
for e in embeddings:
    print(e)
