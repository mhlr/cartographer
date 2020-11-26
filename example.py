# -*- coding: utf-8 -*-

# from builtins import *
import re
import os
from toolz.curried import curry, memoize
from toolz import curried as tz
import requests

import pandas as pd
import numpy as np
import pylab as plt

import scipy
import networkx as nx
from gensim.summarization import keywords, summarize
from wordcloud import WordCloud
import sklearn as sk
from sklearn import neighbors, pipeline, cluster
import tensorflow_hub as hub


# NAME_OF_TEXT = 'COMMON SENSE'
#
# with open('commonsense_painet.txt', 'r') as f:
#     doc = f.read()
#     doc = doc.split(NAME_OF_TEXT)[2].replace('\r\n', '\n')

NAME_OF_TEXT = "3 papers: gpt3, transformers, detr"

with open('papers/all.txt', 'r') as f:
    doc = f.read()

pars = pd.Series(doc.split('\n\n')).str.replace('\n', ' ')

pars.str.len().apply(lambda x:np.log2(x+1)).astype(int).value_counts()

keywords = keywords(doc, scores=True, lemmatize=True, words=500)
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

lower_bound_chars, upper_bound_chars = 256, 512
avg_word_len = 4.79
word_count = int((lower_bound_chars + upper_bound_chars) / (2 * (avg_word_len + 1)))
lens = pars.str.len()  # paragraph lengths
nice_pars = pars[(lens >= lower_bound_chars)]  # paragraphs we want to use
def text_reduce_return(paragraph):
    if len(paragraph) < upper_bound_chars:
        return paragraph
    else:
        try:
            return summarize(paragraph, word_count=word_count).replace("\n", " ") or \
                   paragraph[:upper_bound_chars]
        except ValueError:  # usually happens if there aren't multiple sentences in paragraph
            return paragraph[:upper_bound_chars]

nice_pars = nice_pars.apply(text_reduce_return)


len(nice_pars), len(pars)

vecs = emb(tuple(nice_pars), "https://tfhub.dev/google/universal-sentence-encoder-large/5").numpy()

D = sk.metrics.pairwise_distances(vecs, metric='cosine')  # pairwise distances of vectors
R = scipy.sparse.csgraph.minimum_spanning_tree(D).max()  # reduced graph
G = neighbors.radius_neighbors_graph(vecs, R, metric='cosine')

@curry
def clust(g, v, n):
    model = cluster.AgglomerativeClustering(n, connectivity=g, linkage='ward', affinity='euclidean'))
    labels = model.fit_predict(v)
    silh = sk.metrics.silhouette_samples(v, labels, metric='cosine')
    return (silh.mean(), n, labels, silh, model)

core = nx.k_core(nx.Graph(G))

# Capitalize all occurrences of keywords for easy display on the output
pattern = re.compile(f"\\b({tz.pipe(keywords, tz.pluck(0), '|'.join)})\\b")  # TODO, make matching case insensitive
nice_pars = nice_pars.apply(lambda x: re.sub(pattern, lambda m: m.group().upper(), x))  # TODO, add [[]] around our keywords

core_pars = np.array(nice_pars)[core.nodes]
core_vecs = vecs[core.nodes]

sil_u, n, lab, sil, p = clust(nx.adjacency_matrix(core), core_vecs, 8)

len(lab), len(sil)

layers = nx.onion_layers(core)

len(core.nodes)

df = pd.DataFrame(data=[{"Label": par, "Cluster ID": cid, "Silhouette Score": ss} for par, cid, ss in zip(core_pars, lab, sil)])
df = df[df["Silhoutte Score"]] > 0]

df['Cluster ID'] = df.apply(lambda row: "T" + str(row['Cluster ID']), axis=1)

for cluster_id in df['Cluster ID'].unique():
    df = df.append({"Label": cluster_id, "Cluster ID": NAME_OF_TEXT, "Silhouette Score": None}, ignore_index=True)
else:
    df = df.append({"Label": NAME_OF_TEXT, "Cluster ID": None, "Silhouette Score": None}, ignore_index=True)

df.to_csv("out.csv", index=False)
print()
