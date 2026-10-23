# pull data on a lot of top rated movies
import os, tqdm
import tmdbsimple as tmdb
import pandas as pd
from dotenv import load_dotenv; load_dotenv()

print(f"workdir: {os.getcwd()}")
tmdb.API_KEY = os.getenv("TMDB_API_KEY")

n_pages = 1
movie_data = []

for page in tqdm.tqdm(range(1,n_pages+1), desc="pulling pages", leave=True):
    resp = tmdb.Movies().popular(page=page)
    for m in tqdm.tqdm(resp["results"], desc="pulling popular movies", leave=False):
        movie_id = m["id"]
        movie_details = tmdb.Movies(movie_id).info()

        title = movie_details.get("title","")
        overview = movie_details.get("overview","")
        tagline = movie_details.get("tageline","")
        genres = ", ".join(g["name"] for g in movie_details.get("genres",[]))
        release_date = movie_details.get("release_date","")[:4]
        origin_country = movie_details.get("origin_country","")
        production_countries = ", ".join(c["name"] for c in movie_details.get("production_countries",[]))
        production_companies = ", ".join(c["name"] for c in movie_details.get("production_companies",[]))
        runtime = str(movie_details.get("runtime","")); runtime += " minutes" if runtime else ""

        rich_text = (
            f"Title: {title} ({release_date}). "
            f"Genres: {genres}. "
            f"Tagline: {tagline} "
            f"Overview: {overview}. "
            f"Production companies: {production_companies}. "
            f"Origin country: {origin_country}. "
            f"Production contries: {production_countries}"
        )

        movie_data.append({
            "id": movie_id,
            "title": title,
            "genres": genres,
            "document": rich_text
        })
print("done fetching movies!")

df = pd.DataFrame(movie_data).set_index("id")
df.to_csv("../data/top_rated.csv")
print(f"Movie dataset: {df.shape}")
print(df.head())