import os

from typing import List
import pandas as pd


def zk_output_files(output_dir: str, keywords: List[str], df: pd.DataFrame, raw_text: str) -> None:
    unique_topic_labels = df['Cluster ID'].unique()

    # Write out all input text
    with open(os.path.join(output_dir, "RAW_TXT.md"), "w") as f:
        f.write(raw_text)

    # KEYWORD FILES: create blank files - one per keyword
    for kwd in keywords:
        with open(os.path.join(output_dir, kwd+".md"), "w") as f:
            f.write(" ")

    # write out topic paragraphs
    for idx, row in df.iterrows():
        file_title = row["Cluster ID"] + str(idx)
        filename = file_title+".md"
        file_body = row["Topic Paragraph"]

        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(file_body)

        # TODO, write out the silhouette score?
