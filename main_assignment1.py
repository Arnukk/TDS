__author__ = 'akarapetyan'

import matplotlib.pyplot as plt
from wnaffect import WNAffect
from emotion import Emotion
from nltk.corpus import wordnet as wn
from cursor_spinning import SpinCursor
import time
import sys
import numpy as np
import PorterStemmer as ps

#CONSTANTS
#array of inapproriate words to be excluded
toexclude = ["world", "blue", "weight", "self", "hero", "identification", "will", "sympathy", "empathy", "preference", "compatibility", "softheartedness", "puppy"]

def fancy_output(msg, command, starting_time, *args):
    """
    Just a fancy way of outputting the progress of a command
    @param msg, command, params: the message to output, command to be executed, params
    @return output: the result of the command
    """
    spin = SpinCursor(msg=msg, minspin=5, speed=5)
    spin.start()
    output = command(*args)
    if output: spin.stop()
    sys.stdout.write("Elapsed time - %3.6f seconds" % (time.time()-starting_time))
    print '\n'
    return output


def preprocess_database(year_range):
    """
    Filter the database of 1-grams according to the year range chosen
    @param year_range
    @return filtered_db
    """
    path_pattern = "data\googlebooks-eng-1M-1gram-20090715-"
    filtered_db = {}
    for source in [path_pattern + '%d.csv' % i for i in range(10)]:
        #df = pd.read_csv(source, names=['word', 'year', 'occurred', 'pages', 'books'], sep='\t', error_bad_lines=False)
        #if len(df[(df['word'] == word) & (df['year'] == year)].index.tolist()) > 0:
            #occur_count = df.loc[[df[(df['word'] == word) & (df['year'] == year)].index.tolist()[0]]].iloc[0]['occurred']
            #return occur_count
        with open(source) as f:
            for line in f:
                data = line.split('\t')
                if int(data[1]) in year_range:
                    if int(data[1]) in filtered_db:
                        filtered_db[int(data[1])].append(line)
                    else:
                        filtered_db[int(data[1])] = []
                        filtered_db[int(data[1])].append(line)
    return filtered_db


def get_mood_score(mood, year, filtered_db):
    """
    Calculates the mood score of the give mood for a given year
    :param mood, year, filtered_db:
    :return moodscore
    """
    moodcount = 0
    the_count = 0
    for item in filtered_db[year]:
        data = item.split('\t')
        if data[0] in mood or data[0].lower() in mood:
            moodcount += int(data[2])
        if data[0] == "the" or data[0].lower() == "the":
            the_count += int(data[2])
    moodscore = (1.0 * moodcount/the_count)/1.0*len(mood)
    return moodscore



def get_emotion_terms(emotion):
    """
    Given the emotion, the function returns all the terms related to that emotion
    @param emotion: name of the emotion - string
    @return terms_array
    """
    terms_array = [emotion]
    for term in Emotion.emotions[emotion].get_children([]):
        if "-" in term:
            for t in term.split("-"):
                if t not in toexclude:
                    terms_array.append(t.lower()) if t not in terms_array and t.lower() not in terms_array else None
        else:
            terms_array.append(term) if term not in terms_array and term.lower() not in terms_array else None
        if "-" in term:
            for t in term.split("-"):
                if t not in toexclude:
                    for synset in wn.synsets(t):
                        for lemma in synset.lemmas():
                            if "_" not in str(lemma.name()):
                                terms_array.append(str(lemma.name()).lower()) if str(lemma.name()) not in terms_array and str(lemma.name()).lower() not in terms_array else None
        else:
            for synset in wn.synsets(term):
                for lemma in synset.lemmas():
                    if "_" not in str(lemma.name()):
                        terms_array.append(str(lemma.name()).lower()) if str(lemma.name()) not in terms_array and str(lemma.name()).lower() not in terms_array else None
    return terms_array


def get_stems():
    """
    Returns the array of the filtered stems according to the conditions mentioned in the paper
    @return: stemarray
    """
    stemarray = []
    p = ps.PorterStemmer()
    infile = open("./part-of-speech.txt", 'r')
    while 1:
        output = ''
        line = infile.readline()
        line = line.split('\t')[0]
        if line == '':
            break
        for c in line:
            if c.isalpha():
                word += c.lower()
            else:
                if word:
                    output += p.stem(word, 0,len(word)-1)
                    word = ''
                output += c.lower()
        stemarray.append(output) if (len(output) > 2 and output not in stemarray) else None
    infile.close()
    return stemarray


if __name__ == "__main__":
    starting_time = time.time()
    print "\n+++++++++++++++++++++++++++++++++++"
    print "TDS - Assignment 1"
    print "+++++++++++++++++++++++++++++++++++\n"
    """
    Inittializing Wordnet-Affect
    @DEPENDENCIES: NLTK 3.1 or higher, WordNet 1.6 (unix-like version is utilised), WordNet-Domains 3.2
    """
    YEAR_RANGE = range(1900, 2001, 7)

    wna = fancy_output("Initializing Wordnet", WNAffect, starting_time, './wordnet-1.6/', './wn-domains-3.2/')

    joy_terms = fancy_output("Getting the terms for the mood category JOY", get_emotion_terms, starting_time, 'joy')
    joy_terms.extend([term for term in fancy_output("", get_emotion_terms, starting_time, 'liking') if term not in joy_terms])
    joy_terms.extend([term for term in fancy_output("", get_emotion_terms, starting_time, 'love') if term not in joy_terms])
    #joy_terms.extend([term for term in fancy_output("", get_emotion_terms, starting_time, 'levity') if term not in joy_terms])
    #joy_terms.extend([term for term in fancy_output("", get_emotion_terms, starting_time, 'gratitude') if term not in joy_terms])

    sadness_terms = fancy_output("Getting the terms for the mood category SADNESS", get_emotion_terms, starting_time, 'sadness')

    filtered_dataset = fancy_output("Preprocessing the dataset", preprocess_database, starting_time, YEAR_RANGE)

    spin = SpinCursor(msg="Computing the mood scores", minspin=5, speed=5)
    spin.start()
    joy_mood_scores = {}
    sadness_mood_scores = {}
    for year in YEAR_RANGE:
        joy_mood_scores[year] = get_mood_score(joy_terms, year, filtered_dataset)
        sadness_mood_scores[year] = get_mood_score(sadness_terms, year, filtered_dataset)
    if len(joy_mood_scores) == len(YEAR_RANGE): spin.stop()
    sys.stdout.write("Elapsed time - %3.6f seconds" % (time.time()-starting_time))
    print '\n'

    joy_mood_scores_mean = np.mean(joy_mood_scores.values())
    joy_mood_scores_std = np.std(joy_mood_scores.values())

    sadness_mood_scores_mean = np.mean(sadness_mood_scores.values())
    sadness_mood_scores_std = np.std(sadness_mood_scores.values())

    normalize = lambda mood_val: (mood_val - joy_mood_scores_mean)/(1.0 * joy_mood_scores_std)
    joy_normalized = {}
    for key in joy_mood_scores.keys():
        joy_normalized[key] = normalize(joy_mood_scores[key])

    normalize = lambda mood_val: (mood_val - sadness_mood_scores_mean)/(1.0 * sadness_mood_scores_std)
    sadness_normalized = {}
    for key in sadness_mood_scores.keys():
        sadness_normalized[key] = normalize(sadness_mood_scores[key])

    x = [year for year in YEAR_RANGE]
    y = [joy_normalized[key] - sadness_normalized[key] for key in YEAR_RANGE]

    markerline, stemlines, baseline = plt.stem(x, y, '-.')
    plt.grid()
    axes = plt.gca()
    axes.set_xlim([x[0]-3, x[-1]+3])
    plt.title('Historical periods of positive and negative moods')
    plt.xlabel('Year')
    plt.ylabel('Joy - Sadness (Z scores)')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    plt.setp(stemlines, linewidth=1, color=[0.08,0.4,1])
    plt.grid()
    print "====== Simulation finished in ", time.time() - starting_time, " seconds =========\n"
    plt.show()

