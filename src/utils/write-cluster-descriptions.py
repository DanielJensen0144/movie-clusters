# integrations
import os, json
import numpy as np
import pandas as pd
from dotenv import load_dotenv; load_dotenv()

import re

# main

# load clusters and labels
df = pd.read_csv("../data/top-rated.csv.gz", compression="gzip")
clusters = np.load("../data/clusters-kmeans-30.npy")
with open("../data/cluster-labels-llm.json", "r") as f:
    labels = json.load(f)

# write to more readable individual files
for c, label in enumerate(labels):
    label = label[:70].replace("2. ","### 2. ") + label[70:]
    df_cluster = df.iloc[clusters == c]
    with open(f"../cluster-labels/cluster-{c+1}.md", "w") as f:
        f.write("# ")
        f.write(label)
        f.write("\n\n\n")
        f.write("*Movies in cluster:*\n\n")
        f.write("\n\n".join([
            f"### {row['title']}: {row['document']}"
            for _, row in df_cluster.iterrows()
        ]).replace(": Title: ",":\nTitle: "))

print(f"wrote labels into files: {os.getenv("HOMEPATH")}/src/cluster-labels/")