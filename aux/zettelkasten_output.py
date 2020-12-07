import re
import os

from typing import List
import pandas as pd


def term_zettelkasten_link_formatter(match: re.Match):
    term = match.group()  # get the actual word (string)
    return f"[[{term}]]"  # modification to allow for linking by Zettelkasten software (e.g, Obsidian)

def term_zettelkasten_tag_formatter(match: re.Match):
    term = match.group()  # get the actual word (string)
    term = term.replace(" ", "-")  # deal with bi-gram tags
    return f"#{term.lower()}"  # modification to allow for tagging by Zettelkasten software (e.g, Obsidian)

def zk_output_files(output_dir: str, key_terms: List[str], df: pd.DataFrame, raw_text: str) -> None:
    unique_topic_labels = df['Cluster ID'].unique()

    # format all text with zettelkasten formatter
    pattern = re.compile(f"\\b({'|'.join(key_terms)})\\b", flags=re.IGNORECASE)
    # format the topic paragraphs
    df['Topic Paragraph'] = df['Topic Paragraph'].apply(
        lambda par: re.sub(pattern, term_zettelkasten_tag_formatter, par)
    )
    # format the raw text (so it can be linked back by the user to the original paper)
    raw_text = re.sub(pattern, term_zettelkasten_tag_formatter, raw_text)

    # Write out all input text
    with open(os.path.join(output_dir, "RAW_TXT.md"), "w") as f:
        f.write(raw_text)

    # # KEYWORD FILES: create blank files - one per keyword
    # for kwd in key_terms:
    #     with open(os.path.join(output_dir, kwd+".md"), "w") as f:
    #         f.write(" ")

    # write out topic paragraphs
    for idx, row in df.iterrows():
        file_title = row["Cluster ID"] + str(idx)
        filename = file_title+".md"
        file_body = row["Topic Paragraph"]

        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(file_body)

        # TODO, write out the silhouette score?
