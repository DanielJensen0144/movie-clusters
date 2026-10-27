# integrations
import sys, os, json, tqdm
from dotenv import load_dotenv; load_dotenv()
# print(f"workdir: {os.getcwd()}")

import tmdbsimple as tmdb
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.figsize": (30,18),
    "font.size": 14,
    "axes.grid": True,
    "grid.alpha": 0.2
})

from google import genai
# print(f"Google GenAI version: {genai.__version__}")


# helper functions

# google sdk client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# cluster labelling function
def label_cluster(df_cluster):
    titles_and_overviews = "\n".join([
        f"- {row['title']}: {row['document']}"
        for _, row in df_cluster.iterrows()
    ])

    prompt = (
        "Here is a sample of movies that belong to the same unsupervised cluster:\n"
        f"{titles_and_overviews}\n\n"
        "Analyze these movies and provide:\n"
        "1. A short, catchy 2-3 word thematic label for this cluster (e.g., 'Gritty Crime Thrillers' or 'Whimsical Animated Family').\n"
        "2. A one-sentence explanation of what unites them."
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    return response.text

def label_all_clusters(df, clusters):
    labels = []
    for c in tqdm.tqdm(np.unique(clusters), desc="auto-labeling movie clusters", leave=True):
        df_cluster = df.iloc[clusters == c]
        try:
            labels.append(label_cluster(df_cluster))
        except:
            print(f"early stopping at cluster c={c}")
            return labels
    return labels



# main
if __name__ == "__main__":
    df = pd.read_csv("../data/top-rated.csv.gz", compression="gzip")
    clusters = np.load("../data/clusters-kmeans-30.npy")
    labels = label_all_clusters(df, clusters)

    with open("../data/cluster-labels-llm.json", "w") as f:
        json.dump(labels, f, indent=4)
    print(f"labeled all {len(np.unique(clusters))} movie clusters!")

