# -*- coding: utf-8 -*-


import argparse
from functools import partial

import os
from toolz.curried import curry, memoize, first, second

import pandas as pd
import numpy as np

import scipy
import networkx as nx
from gensim.summarization import keywords, summarize
import sklearn as sk
from sklearn import neighbors, cluster
import tensorflow_hub as hub
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from aux.text_extract import get_all_pdf_text_concatenated
from aux.helpers import create_output_dir
from aux.zettelkasten_output import zk_output_files


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
        except ValueError:  # usually happens if there aren't multiple sentences in the paragraph
            return paragraph[:upper_bound_chars]

@curry
def clust(g, v, n):
    model = cluster.AgglomerativeClustering(n, connectivity=g, linkage='ward', affinity='euclidean')
    labels = model.fit_predict(v)
    silh = sk.metrics.silhouette_samples(v, labels, metric='cosine')
    return (silh.mean(), n, pd.Series(labels), silh, model)


def main(args):
    try:
        create_output_dir(args.output_dir, delete_if_exists=args.delete_existing)  # try creating output directory
    except RuntimeError as e:
        print("Requested output directory is not empty."
              "Please try a new output dir (safe) or set --delete_existing (destructive)")

    name_of_pdf_dir = os.path.basename(args.directory_with_pdfs)

    all_text = get_all_pdf_text_concatenated(args.directory_with_pdfs)

    if not all_text:
        raise RuntimeError("No text was extracted")

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

    core_nodes = core.nodes
    core_pars = np.array(nice_pars)[core_nodes]
    core_vecs = vecs[core_nodes]

    sil_u, n, lab, sil, p = clust(nx.adjacency_matrix(core), core_vecs, 8)

    layers = nx.onion_layers(core)  # TODO, can I remove this?

    unique_labs = lab.unique()

    from sklearn.feature_extraction.text import TfidfVectorizer
    words = list(map(first, keywords('\n'.join(core_pars), scores=True, lemmatize=False, words=int(2**10))))
    word_vecs = emb(
      tuple(words), "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    ).numpy()

    par_word_sim = core_vecs @ word_vecs.T
    
    topics = pd.Series(
        dict(
            [
                (
                    i,
                    max(
                        [
                            (
                                words[w],
                                min(
                                    sk.metrics.roc_auc_score(lab[ix] == i, par_word_sim[ix, w])
                                    for j in unique_labs
                                    if i != j
                                    for ix in [lab.isin([i, j])]
                                ),
                            )
                            for w in range(len(words))
                        ],
                        key=second,
                    )[0],
                )
                for i in unique_labs
            ]
        )
    ).sort_index()

    df = pd.DataFrame(
      data=[
        {"Topic Paragraph": par, "Cluster ID": topics[cid], "Silhouette Score": ss, "Onion Layer": layer}
        for par, cid, ss,  layer in zip(core_pars, lab, sil, pd.Series(layers))
      ]
    )
 
    # df = pd.DataFrame(
    #     data=[{"Topic Paragraph": par, "Cluster ID": cid, "Silhouette Score": ss} for par, cid, ss in zip(core_pars,
    #                                                                                                       lab,
    #                                                                                                       sil)])

    # only keep the paragraphs which have a positive silhouette score
    # (this gives us the paragraphs which overwhelmingly consist of a single topic)
    df = df[df["Silhouette Score"] > 0]

    # # TODO, replace with Topic Labels
    # df['Cluster ID'] = df.apply(lambda row: "CID" + str(row['Cluster ID']), axis=1)

    keyword_strings = [kwd.strip() for kwd, score in text_keywords if kwd.lower() not in stopwords.words()]
    zk_output_files(args.output_dir, keyword_strings, df, all_text)

    return {
        "text_keywords": text_keywords
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("directory_with_pdfs",
                            help="Please provide the directory which contains the PDFs"
                                 "which you'd like to build an information map of.",
                            type=str)
    arg_parser.add_argument("output_dir",
                            default="out.csv")
    arg_parser.add_argument("--delete_existing",
                            help="Will delete pre-existing output directory",
                            action="store_true")
    arg_parser.add_argument("--num_keywords",
                            help="The upper limit on number of keywords you'd like to be extracted",
                            type=int,
                            default=100)
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
