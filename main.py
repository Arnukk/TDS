__author__ = 'akarapetyan'
from pandas import DataFrame, read_csv
import functions
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number
from wnaffect import WNAffect
from emotion import Emotion
from nltk.corpus import wordnet as wn


"""
Inittializing Wordnet-Affect
@DEPENDENCIES: NLTK, WordNet 1.6 (unix-like version is utilised), WordNet-Domains 3.2
"""
wna = WNAffect('./wordnet-1.6/', './wn-domains-3.2/')


def searc_database(word, year):
    """
    Searches the database of 1-grams for the given word in the given year and returns the number of occurrences
    :param word, year:
    :return occur_count
    """
    path_pattern = "data\googlebooks-eng-1M-1gram-20090715-"
    occur_count = 0
    for source in [path_pattern + '%d.csv' % i for i in range(10)]:
        df = pd.read_csv(source, names=['word', 'year', 'occurred', 'pages', 'books'], sep='\t')
        if len(df[(df['word'] == word) & (df['year'] == year)].index.tolist()) > 0:
            occur_count = df.loc[[df[(df['word'] == word) & (df['year'] == year)].index.tolist()[0]]].iloc[0]['occurred']
            print occur_count
    return occur_count

def get_emotion_terms(emotion):
    """
    Given the emotion, the function returns all the terms related to that emotion
    @param emotion: name of the emotion - string
    @return terms_array
    """
    terms_array = [emotion]
    for term in Emotion.emotions[emotion].get_children():
        terms_array.append(term) if term not in terms_array else None
        for synset in wn.synsets(term):
            for lemma in synset.lemmas:
                terms_array.append(lemma.name) if lemma.name not in terms_array else None
    return terms_array

def occurence_count(word, year, database):
    """
    Function for returning the number of occurrences of a word in a given year
    @param  word, year, database: the word and the year to look in, the database of 1-grams
    @return: count
    """
    pass

print len(get_emotion_terms('joy'))
searc_database('and', 1900)