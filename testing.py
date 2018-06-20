from random import randint
import matplotlib

import numpy as np
import torch
import time

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

start_time = time.time()

GLOVE_PATH = 'dataset/GloVe/glove.840B.300d.txt'
model = torch.load('encoder/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
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


elapsed_time = time.time() - start_time
print("Time elapsed: ",elapsed_time)

with open('samples_output.txt', 'w') as f:
    for e in embeddings:
        f.write('['+' '.join(str(x) for x in e)+']')
        f.write('\n')

pca = PCA(n_components=3)
principal_components=pca.fit_transform(embeddings)

with open('samples_output_pca3.txt', 'w') as f:
    for pc in principal_components:
        f.write('['+' '.join(str(x) for x in pc)+']')
        f.write('\n')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for pc in principal_components:
    ax.scatter(pc[0],pc[1],pc[2],c='b',marker='o')
    ax.annotate('(%s,' %i, xy=(i,j))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
