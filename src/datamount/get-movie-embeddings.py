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

# main
df = pd.read_csv("../data/top_rated.csv.gz", compression="gzip")

# google gemini api
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# movie embedding function
def get_movie_embeddings(text_contents, batch_size=100):
    all_embeddings = []

    # process chunks
    for i in range(0, len(text_contents), batch_size):
        batch = text_contents[i : i + batch_size]
        contents = [genai.types.Content(parts=[genai.types.Part.from_text(text=t)]) for t in batch]
        resp = client.models.embed_content(
            model="gemini-embedding-2",
            contents=contents
        )
        all_embeddings.extend([e.values for e in resp.embeddings])

    return all_embeddings

# embed all movies in dataset
documents = df.document.tolist()
# embeddings = np.array(get_movie_embeddings(documents))

embeddings = np.load("../data/embeddings.npy")


# save embeddings
with open("../data/embeddings.npy", "wb") as f:
    np.save(f, embeddings)
print("saved embedding vectors!")


# scatterplot
if int(os.getenv("MAKEPLOT1")):

    # plot movie embeddings
    p = PCA(whiten=True, n_components=2, random_state=67)
    embeddings_pca = p.fit_transform(embeddings)
    print(f"Explained variance: {p.explained_variance_ratio_.sum()*100:.2f}%")

    n_plot = int(os.environ.get("N_PLOT", 100))
    pc1 = embeddings_pca[:,0]
    pc2 = embeddings_pca[:,1]

    with open("../data/embeddings-pc1.npy", "wb") as f:
        np.save(f, pc1)
    with open("../data/embeddings-pc2.npy", "wb") as f:
        np.save(f, pc2)

    # scatter projected embeddings
    plt.scatter(pc1[:n_plot], pc2[:n_plot], marker="x", label="Movies", cmap="tab10", c=np.arange(n_plot))

    # add movie titles
    if int(os.getenv("MOVIE_NAMES")):
        for x, y, title in zip(pc1, pc2, df.title.tolist()):
            plt.annotate(title, (x,y), textcoords="offset points", xytext=(0,5), ha="center", fontsize=10, alpha=0.5)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("TMDB Top Rated Movie Latent Space")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../imgs/top-rated-movie-latent-space.jpg")