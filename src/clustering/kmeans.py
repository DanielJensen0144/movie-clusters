# integrations
import sys, os, json
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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# main
embeddings = np.load("../data/embeddings.npy")
pc1 = np.load("../data/embeddings-pc1.npy")
pc2 = np.load("../data/embeddings-pc2.npy")

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=67, n_init="auto")
clusters = kmeans.fit_predict(embeddings)

with open(f"../data/clusters-kmeans-{n_clusters}.npy", "wb") as f:
    np.save(f, clusters)
print("saved kmeans clusters (k={n_clusters})")

# plot projected embeddings by clusters
if int(os.getenv("MAKEPLOT2")):
    
    # scatterplot
    n_plot = int(os.environ.get("N_PLOT", 100))
    plt.scatter(pc1[:n_plot], pc2[:n_plot], marker="x", label="Movies", cmap="tab10", c=clusters[:n_plot], s=100)

    # add movie titles
    if int(os.getenv("MOVIE_NAMES")):
        df = pd.read_csv("../data/top_rated.csv.gz", usecols=["title"], compression="gzip")
        for x, y, title in zip(pc1, pc2, df.title.tolist()):
            plt.annotate(title, (x,y), textcoords="offset points", xytext=(0,5), ha="center", fontsize=10, alpha=0.5)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("TMDB Top Rated Movie Latent Space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../imgs/top-rated-movie-latent-space-kmeans-{n_clusters}.jpg")