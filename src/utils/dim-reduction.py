# integrations
import sys, os, json
from dotenv import load_dotenv; load_dotenv()
print(f"workdir: {os.getcwd()}")

import tmdbsimple as tmdb
import pandas as pd
import numpy as np

from google import genai
print(f"Google GenAI version: {genai.__version__}")

import matplotlib.pyplot as plt
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.figsize": (30,18),
    "font.size": 14,
    "axes.grid": True,
    "grid.alpha": 0.2
})

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# main

# load embeddings
embeddings = np.load("../data/embeddings.npy"); embeddings

# embeddings ---> 3072 -- pca -> 50 -- t-sne -> 2
embeddings_pca = PCA(n_components=50, random_state=67).fit_transform(embeddings)
embeddings_tsne = TSNE(
    n_components=2, perplexity=30,
    init="pca", random_state=67
).fit_transform(embeddings_pca)

# scatterplot
if int(os.getenv("MAKEPLOT1")):

    n_plot = int(os.environ.get("N_PLOT", 100))
    tsne1 = embeddings_tsne[:,0]
    tsne2 = embeddings_tsne[:,1]

    with open("../data/embeddings-tsne1.npy", "wb") as f:
        np.save(f, tsne1)
    with open("../data/embeddings-tsne2.npy", "wb") as f:
        np.save(f, tsne2)

    # scatter projected embeddings
    plt.scatter(tsne1[:n_plot], tsne2[:n_plot], marker="x", label="Movies", cmap="tab10", c=np.arange(n_plot))

    # add movie titles
    if int(os.getenv("MOVIE_NAMES")):
        for x, y, title in zip(tsne1, tsne2, df.title.tolist()):
            plt.annotate(title, (x,y), textcoords="offset points", xytext=(0,5), ha="center", fontsize=10, alpha=0.5)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("TMDB Top Rated Movie Latent Space")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../imgs/top-rated-movie-latent-space-tsne.jpg")