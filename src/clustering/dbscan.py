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
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN

# main
embeddings = np.load("../data/embeddings.npy")
pc1 = np.load("../data/embeddings-pc1.npy")
pc2 = np.load("../data/embeddings-pc2.npy")

# dbscan struggles with curse of dimensionality, reduce dims
n_dims = 50
embeddings_pca = normalize(np.load("../data/embeddings-pca.npy")[:,:n_dims])

# clustering
db = DBSCAN(eps=0.5, min_samples=5, metric="euclidean", n_jobs=-1)
clusters = db.fit_predict(embeddings)

n_clusters = len(set(clusters)) + (-1 if -1 in clusters else 0)
print(f"clusters: {n_clusters}, noise: {sum(clusters == -1)}")

with open(f"../data/clusters-dbscan-{n_clusters}.npy", "wb") as f:
    np.save(f, clusters)
print("saved dbscan clusters")

# plot projected embeddings by clusters
if int(os.getenv("MAKEPLOT2")):
    
    # scatterplot
    n_plot = int(os.environ.get("N_PLOT", 100))
    plt.scatter(pc1[:n_plot], pc2[:n_plot], marker="x", label="Movies", cmap="tab10", c=clusters[:n_plot], s=100)

    # add movie titles
    if int(os.getenv("MOVIE_NAMES")):
        df = pd.read_csv("../data/top-rated.csv.gz", usecols=["title"], compression="gzip")
        for x, y, title in zip(pc1, pc2, df.title.tolist()):
            plt.annotate(title, (x,y), textcoords="offset points", xytext=(0,5), ha="center", fontsize=10, alpha=0.5)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("TMDB Top Rated Movie Latent Space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../imgs/top-rated-movie-latent-space-dbscan-{n_clusters}.jpg")