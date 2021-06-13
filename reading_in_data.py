import pandas as pd
import numpy as np

rating = pd.read_csv('ml-latest-small/ratings.csv')
user_rating_matrix = pd.DataFrame(rating,
                                  columns=['userId', 'movieId', 'rating'])
user_rating = user_rating_matrix.pivot(index="userId",
                                       columns="movieId",
                                       values="rating")
ratings_pivot = user_rating.replace(np.nan, 0)
movies = pd.read_csv('ml-latest-small/movies.csv')
movies_df = pd.DataFrame(movies, columns=['movieId', 'title'])

movies_df['year'] = 0
for i, data in movies_df.iterrows():
    try:
        movies_df.loc[i, 'year'] = int(data['title'].split('(')[-1].replace(
            ')', ''))
    except:
        movies_df.loc[i, 'year'] = 0

movies = movies_df[['title', 'year', 'movieId']]
linkfile = pd.read_csv('ml-latest-small/links.csv')

link = pd.DataFrame(linkfile, columns=['movieId', 'tmdbId'])

