# Movie Clustering

## Inspiration
I found this idea from an instagram reel and thought it looked like a fun project.

## Idea
The idea is to...
- collect movie data using The Movie DataBase (TMDB) API
- produce embeddings on the movies using modern embedding model like gemini-embedding-2
- use unsupervised learning to cluster the movies into groups of similar movies given metric of similarity
- use generative ai to autoproduce semantic labels characterizing the clusters
- possibly cluster the clusters to get subgenres
- produce a local frontend data app to explore the results

## Future Extensions
I have considered using collaborative filtering to produce a reccomendation system based on personal movie ratings. This is probably where I'd like to take the project going forward.

## Main resources
TMDB API (pip install tmdbsimple)
Pandas (pip install pandas)
Google API (pip install google-genai)


## Current State
So far I have explored a few of the pipelines in some jupyter notebooks, and I'm also pulling some movie states, so stay tuned for that.

Here is an image of the current clustering of some movies I have:
![Movie Clusters](./src/imgs/popular-movie-latent-space-clusters.jpg)
