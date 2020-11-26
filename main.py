# -*- coding: utf-8 -*-


import argparse
from functools import partial

import re
import os
from toolz.curried import curry, memoize
from toolz import curried as tz

import pandas as pd
import numpy as np

import scipy
import networkx as nx
from gensim.summarization import keywords, summarize
import sklearn as sk
from sklearn import neighbors, pipeline, cluster
import tensorflow_hub as hub

from aux.text_extract import get_all_pdf_text_concatenated


os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser("~/.cache/tfhub_modules")

avg_word_len = 4.79  # average number of characters per word in the English language

@memoize
def tfload(model_url):
  return hub.load(model_url)

@memoize
def emb(texts, model_url):
  return tfload(model_url)(texts)

def text_reduce_return(paragraph, upper_bound_chars, max_word_count):
    if len(paragraph) < upper_bound_chars:
        return paragraph
    else:
        try:
            return summarize(paragraph, word_count=max_word_count).replace("\n", " ") or \
                   paragraph[:upper_bound_chars]
        except ValueError:  # usually happens if there aren't multiple sentences in paragraph
            return paragraph[:upper_bound_chars]

@curry
def clust(g, v, n):
    pipe = pipeline.Pipeline([
        ('agg', cluster.AgglomerativeClustering(n, connectivity=g, linkage='ward', affinity='euclidean'))
    ])
    labels = pipe.fit_predict(v)
    silh = sk.metrics.silhouette_samples(v, labels, metric='cosine')
    return (silh.mean(), n, labels, silh, pipe)


def main(args):
    name_of_pdf_dir = os.path.basename(args.directory_with_pdfs)

    all_text = get_all_pdf_text_concatenated(args.directory_with_pdfs)

    pars = pd.Series(all_text.split('\n\n')).str.replace('\n', ' ')

    pars.str.len().apply(lambda x: np.log2(x + 1)).astype(int).value_counts()  # TODO, is this being stored anywhere?

    text_keywords = keywords(all_text, scores=True, lemmatize=True, words=args.num_keywords)

    lower_bound_chars, upper_bound_chars = args.lower_bound_chars, args.upper_bound_chars
    word_count = int((lower_bound_chars + upper_bound_chars) / (2 * (avg_word_len + 1)))
    lens = pars.str.len()  # paragraph lengths
    nice_pars = pars[(lens >= lower_bound_chars)]  # paragraphs we want to use

    nice_pars = nice_pars.apply(
        partial(text_reduce_return,
                upper_bound_chars=upper_bound_chars, max_word_count=word_count)
    )

    vecs = emb(tuple(nice_pars), args.tfhub_sentence_encoder_url).numpy()

    D = sk.metrics.pairwise_distances(vecs, metric='cosine')  # pairwise distances of vectors
    R = scipy.sparse.csgraph.minimum_spanning_tree(D).max()  # reduced graph
    G = neighbors.radius_neighbors_graph(vecs, R, metric='cosine')

    core = nx.k_core(nx.Graph(G))

    # Capitalize all occurrences of keywords for easy display on the output
    # TODO, make matching case insensitive
    pattern = re.compile(f"\\b({tz.pipe(text_keywords, tz.pluck(0), '|'.join)})\\b")
    nice_pars = nice_pars.apply(
        lambda x: re.sub(pattern, lambda m: m.group().upper(), x))  # TODO add [[]] around our keywords for zettelkasten

    core_nodes = core.nodes
    core_pars = np.array(nice_pars)[core_nodes]
    core_vecs = vecs[core_nodes]

    sil_u, n, lab, sil, p = clust(nx.adjacency_matrix(core), core_vecs, 8)

    layers = nx.onion_layers(core_nodes)

    # TODO, drop items of silhouette <= 0
    df = pd.DataFrame(
        data=[{"Label": par, "Cluster ID": cid, "Silhouette Score": ss} for par, cid, ss in zip(core_pars, lab, sil)])
    df['Cluster ID'] = df.apply(lambda row: "T" + str(row['Cluster ID']), axis=1)

    # add footer to dataframe so that csv export will be imported by gsheet's tree map plotter correctly
    for cluster_id in df['Cluster ID'].unique():
        df = df.append({"Label": cluster_id, "Cluster ID": name_of_pdf_dir, "Silhouette Score": None},
                       ignore_index=True)
    else:
        df = df.append({"Label": name_of_pdf_dir, "Cluster ID": None, "Silhouette Score": None}, ignore_index=True)

    df.to_csv(args.output_filename, index=False)

    return {
        "text_keywords": text_keywords
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("directory_with_pdfs",
                            help="Please provide the directory which contains the PDFs"
                                 "which you'd like to build an information map of.",
                            type=str)
    arg_parser.add_argument("--output_filename",
                            default="out.csv")
    arg_parser.add_argument("--num_keywords",
                            help="The number of keywords you'd like to be extracted",
                            type=int,
                            default=500)
    arg_parser.add_argument("--lower_bound_chars",
                            type=int,
                            default=256)
    arg_parser.add_argument("--upper_bound_chars",
                            type=int,
                            default=512)
    arg_parser.add_argument("--tfhub_sentence_encoder_url",
                            type=str,
                            default="https://tfhub.dev/google/universal-sentence-encoder-large/5")
    args = arg_parser.parse_args()

    out = main(args)

    print(f"Number of keywords: {len(out['text_keywords'])}")
