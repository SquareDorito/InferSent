#!/usr/bin/python
from random import randint
import matplotlib

import numpy as np
import torch
import time

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

class InfersentEmbedding():

    def __init__(self, vocab_k, glove_path, input_file):
        self.vocab_k=vocab_k
        self.glove_path=glove_path
        self.input_file=input_file
        self.embeddings=[]

    def get_3d_data(self):
        pca = PCA(n_components=3)
        principal_components=pca.fit_transform(self.embeddings)

        with open('samples_output_pca3.txt', 'w') as f:
            for pc in principal_components:
                f.write('['+' '.join(str(x) for x in pc)+']')
                f.write('\n')

        return principal_components

    def infersent_embed(self):
        start_time = time.time()

        GLOVE_PATH = self.glove_path
        model = torch.load('InferSent/encoder/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
        model.use_cuda = False

        model.set_glove_path(GLOVE_PATH)
        model.build_vocab_k_words(K=self.vocab_k)

        sentences = []
        with open(self.input_file) as f:
            for line in f:
                sentences.append(line.strip())

        self.embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
        print('nb sentences encoded : {0}'.format(len(self.embeddings)))

        elapsed_time = time.time() - start_time
        print("Time elapsed (embedding): ",elapsed_time)

        with open('samples_output.txt', 'w') as f:
            for e in self.embeddings:
                f.write('['+' '.join(str(x) for x in e)+']')
                f.write('\n')

        return self.embeddings

#a=InfersentEmbedding(500000, 'dataset/GloVe/glove.840B.300d.txt', 'samples.txt')
#a.infersent_embed()
