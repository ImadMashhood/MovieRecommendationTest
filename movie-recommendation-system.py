#!/usr/bin/env python
# coding: utf-8

# In[1]:

movieName = input("Enter Movie Name: ")

import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
cv = CountVectorizer(max_features = 3000, stop_words = 'english')
ps = PorterStemmer()


def recommend(movie):
    index = parsedMovies[parsedMovies['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(similarity[index])),reverse=True, key=lambda x:x[1])[1:11]
    final = []
    count = 0
    for i in movies_list:
        final.append([parsedMovies.iloc[i[0]].title, parsedMovies.iloc[i[0]].vote_average])
        count+=1
    print('Sorted by Similarity: ', final)
    final = sorted(list(final),reverse=True, key=lambda x:x[1])
    return final

def convertToArray(obj):
    list = []
    for i in eval(obj):
        list.append(i['name'])
    return list

def convertToArrayTopTen(obj):
    list = []
    counter = 0
    for i in eval(obj):
        if counter != 10:
            list.append(i['name'])
            counter+=1
        else:
            break
    return list

def getDirector(obj):
    list = []
    for i in eval(obj):
        if i['job'] == 'Director':
            list.append(i['name'])
            break
    return list

def stem(text):
    list = []
    for i in text.split():
        list.append(ps.stem(i))
    return " ".join(list)

print('Created Initial Arrays')
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits,on='title')
print('Remove unused columns')
movies = movies[['movie_id','title','overview','genres','keywords','cast', 'crew', 'vote_average']]
print("Removing all blank values")
movies.dropna(inplace=True)
print('Cleaning Generes')
movies.iloc[0].genres

print('Converting all fields to Arrays')
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['genres'] = movies['genres'].apply(convertToArray)
movies ['keywords'] = movies['keywords'].apply(convertToArray)
movies ['cast'] = movies['cast'].apply(convertToArrayTopTen)
movies ['crew'] = movies['crew'].apply(getDirector)

print("Cleaning field's strings.")
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies ['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies ['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies ['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

print('Create tags array')
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

print("Finalizing tags array")
movies['tags'] = movies['tags'].apply(lambda x:" ".join(x))
movies['tags'] = movies['tags'].apply(lambda x:x.lower())
print('Stemming tags')
movies['tags'] = movies['tags'].apply(stem)
parsedMovies = movies[['movie_id','title','tags','vote_average']]

print("Mapping Vectors")
tagsVectors = cv.fit_transform(parsedMovies['tags']).toarray()

print("Calculating cosine distance")
similarity = cosine_similarity(tagsVectors)

finalArray = recommend(movieName)

print('Sorted by Score: ', finalArray)

