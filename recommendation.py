"""TODO: USE THE TITLE DURING THE TRAINING PROCESS (MAYBE COMBINE THE TITLE WITH THE DESCRIPTION? FOR GENERATING
PREDICTIONS) """
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import jaccard_score
from operator import itemgetter

# Load netflix data from csv and drop rows with missing values
netflix_df = pd.read_csv('data/netflix_titles.csv')
print('Data size before removing missing data:', len(netflix_df))
netflix_df = netflix_df.dropna(axis=0)
print('Data size after removing missing data:', len(netflix_df))
# Most common country and content type distribution
print('Most popular country:', netflix_df.country.mode())
# sns.catplot(x='type', kind='count', data=netflix_df)
# plt.show()

# Recommendation system, construct vector representations of movies(while keeping a dict with their titles)
# A trick to keep correct ordering: First we figure out what shape the following tf idf matrix will have (spoiler alert:
# it's 4673 x 4673).
# Then we construct a dictionary, ignoring Netflix's id convention so we can get a direct mapping from dictionary
# keys, to array rows (1, 2, 3...). This will be very helpful later on. All that truly matters long term is the title ->
# description dictionary, to be able to figure out which movie contains which vectors.
movie_dict = {}  # Title->description
id_dict = {k: '' for k in range(4673)}  # Show_Id->title
cur_id = 0
for title, show_type, description in zip(netflix_df['title'], netflix_df['type'],
                                         netflix_df['description']):
    if show_type == 'Movie':
        movie_dict[title] = description
        # Remove s character for ordering
        id_dict[cur_id] = title
        cur_id += 1

# Bow vectorization to be used with jaccard distance later on
count_vect = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=3000, binary=True)
X_bow = count_vect.fit_transform(movie_dict.values()).toarray()
# Tf-idf vectorization
tfidf_vect = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = tfidf_vect.fit_transform(movie_dict.values()).toarray()
# Cosine similarities with linear kernel function
print('Calculating cosine similarities...')
cosine_similarities = linear_kernel(X_tfidf, X_tfidf)

'''TODO: Improve speed by only appending up to 100 movies while looping over the array, and taking advantange of the 
fact that the dictionary is symmetrical. Not much of an issue for small datasets, but efficiency is important.'''
# Store 100 top similar movies for each movie in dictionary of lists of tuples, ignore duplicates
# We could save time by using the fact that the dictionary is symmetrical, maybe f
similarity_dict = {k: [] for k in range(4673)}
for (row, col), similarity in np.ndenumerate(cosine_similarities):
    if 0.0 < similarity < 1.0:
        similarity_dict[row].append((col, similarity))

print('Storing the 100 most similar movies...')
# Sort dictionary and keep 100 most similar movies for each movie. Sort by second element of each (id, similarity)
# tuple using itemgetter.
for movie, similarities in similarity_dict.items():
    similarity_dict[movie] = sorted(similarities, key=itemgetter(1), reverse=True)[:100]


def get_similar_movies_by_title(title, N, method):
    similarities = []
    print('Searching for similar movies...please wait')
    # Transform to respective vector representation
    if method == 'tf-idf':
        vect_description = tfidf_vect.transform([title]).toarray()
        # Find similarity with each movie, and append to a list of id, similarity tuples
        row = 0
        for movie in X_tfidf:
            similarities.append((id_dict[row], linear_kernel(vect_description, [movie])))
            row += 1
    elif method == 'boolean':
        # Calculate jaccard similarities
        vect_description = count_vect.transform([title]).toarray()
        row = 0
        for movie in X_bow:
            similarities.append((id_dict[row], jaccard_score(y_true=[movie], y_pred=vect_description, average='micro')))
            row += 1

    return sorted(similarities, key=itemgetter(1), reverse=True)[:N]


def get_similar_movies_by_description(description, N, method):
    similarities = []
    print('Searching for similar movies...please wait')
    # Transform to respective vector representation
    if method == 'tf-idf':
        vect_description = tfidf_vect.transform([description]).toarray()
        # Find similarity with each movie, and append to a list of id, similarity tuples
        row = 0
        for movie in X_tfidf:
            similarities.append((id_dict[row], linear_kernel(vect_description, [movie])))
            row += 1
    elif method == 'boolean':
        # Use jaccard similarities
        vect_description = count_vect.transform([description]).toarray()
        row = 0
        for movie in X_bow:
            similarities.append((id_dict[row], jaccard_score(y_true=[movie], y_pred=vect_description, average='micro')))
            row += 1

    return sorted(similarities, key=itemgetter(1), reverse=True)[:N]


# Test the function
title = ''
description = ''
title_or_description = input('Welcome to the Netflix Movie recommendation system! Would you like to find similar movies'
                             ' by title or by description? Write title or description to proceed:\n')
if title_or_description == 'title':
    title = input('Give the movie\'s title:\n')
elif title_or_description == 'description':
    description = input('Give the movie\'s description:\n')
how_many = int(input('How many relevant movie titles would you like to fetch?\n'))
method = input('Would you like to match movies using Bag of Words or tf-idf model? Write boolean or tf-idf\n')
if title_or_description == 'title':
    print('Found these:', get_similar_movies_by_title(title=description, method=method, N=how_many))
elif title_or_description == 'description':
    print('Found these:', get_similar_movies_by_description(description=description, method=method, N=how_many))
