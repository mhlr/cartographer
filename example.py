# -*- coding: utf-8 -*-

# from builtins import *
import os
from toolz.curried import curry, memoize
import requests

import pandas as pd
import numpy as np
import pylab as plt

import scipy
import networkx as nx
from gensim import summarization
from wordcloud import WordCloud
import sklearn as sk
from sklearn import neighbors, pipeline, cluster
import tensorflow_hub as hub


NAME_OF_TEXT = 'COMMON SENSE'

with open('commonsense_painet.txt', 'r') as f:
    doc = f.read()
    doc = doc.split(NAME_OF_TEXT)[2].replace('\r\n', '\n')

pars = pd.Series(doc.split('\n\n')).str.replace('\n', ' ')

pars.str.len().apply(lambda x:np.log2(x+1)).astype(int).value_counts()

keywords = summarization.keywords(doc, scores=True, lemmatize=True, words=200)
print(f"Number of keywords: {len(keywords)}")

wc = WordCloud(height=1300, width=3000,
               background_color='white',
               relative_scaling=0, prefer_horizontal=.95, max_font_size=int(180))
wc.generate_from_frequencies(dict(keywords))
# wc.generate_from_text(doc) # for comparison
plt.figure(figsize=(48,14.5))
plt.imshow(wc)
plt.axis('off')


os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser("~/.cache/tfhub_modules")
@memoize
def tfload(model_url):
  return hub.load(model_url)

@memoize
def emb(texts, model_url):
  return tfload(model_url)(texts)

lens = pars.str.len()  # paragraph lengths
nice_pars = pars[(lens >= 256) & (lens <= 1024)]  # paragraphs we want to use
# TODO, idea: for paragraphs of text over 1024 just keep the first 1024 characters

len(nice_pars), len(pars)

vecs = emb(tuple(nice_pars), "https://tfhub.dev/google/universal-sentence-encoder-large/5").numpy()

D = sk.metrics.pairwise_distances(vecs, metric='cosine')  # pairwise distances of vectors
R = scipy.sparse.csgraph.minimum_spanning_tree(D).max()  # reduced graph
G = neighbors.radius_neighbors_graph(vecs, R, metric='cosine')

@curry
def clust(g, v, n):
    pipe = pipeline.Pipeline([
        ('agg', cluster.AgglomerativeClustering(n, connectivity=g, linkage='ward', affinity='euclidean'))
    ])
    labels = pipe.fit_predict(v)
    silh = sk.metrics.silhouette_samples(v, labels, metric='cosine')
    return (silh.mean(), n, labels, silh, pipe)

core = nx.k_core(nx.Graph(G))

core_pars = np.array(nice_pars)[core.nodes]
core_vecs = vecs[core.nodes]

sil_u, n, lab, sil, p = clust(nx.adjacency_matrix(core), core_vecs, 8)

len(lab), len(sil)

layers = nx.onion_layers(core)

len(core.nodes)

# TODO, drop items of silhouette <= 0
df = pd.DataFrame(data=[{"Label": par, "Cluster ID": cid, "Silhouette Score": ss} for par, cid, ss in zip(core_pars, lab, sil)])
df['Cluster ID'] = df.apply(lambda row: "T" + str(row['Cluster ID']), axis=1)

for cluster_id in df['Cluster ID'].unique():
    df = df.append({"Label": cluster_id, "Cluster ID": NAME_OF_TEXT, "Silhouette Score": None}, ignore_index=True)
else:
    df = df.append({"Label": NAME_OF_TEXT, "Cluster ID": None, "Silhouette Score": None}, ignore_index=True)

df.to_csv("out.csv", index=False)
