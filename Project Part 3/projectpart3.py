# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:31:20 2016

@author: tomec
"""

import urllib
import pandas as pd
from datetime import timedelta
import datetime
import csv
import re
import unicodedata
import nltk
from nltk.sentiment.util import mark_negation
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.keys import Keys
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import sys
from bokeh.io import output_file, show, vplot
from bokeh.plotting import figure
from bokeh.models import Span

### Create function to break apart contractions to its derivative words
### A text file containing this('contractions.txt') should be located at the 
### working directory along with this script.

def break_contractions(text):
    #### Import dictionary of contractions: contractions.txt
    with open('contractions.txt','r') as inf:
        contractions = eval(inf.read())
    
    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')
    result = pattern.sub(lambda x: contractions[x.group()], text)
    return(result)

    
### Create function to lemmatize (stem) words to their root
### This requires the NLTK wordnet dataset.

def lemmatize_words(text):
    # Create a lemmatizer object
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    out = []
    for word in text:
        word = ''.join(w.lower() for w in word if w.isalpha())
        out.append(wordnet_lemmatizer.lemmatize(word))
    return(out)
    

#### Create function to remove stopwords (e.g., and, if, to)
#### Removes stopwords from a list of words (i.e., to be used on lyrics after splitting).
#### This requires the NLTK stopwords dataset.
def remove_stopwords(text):
    # Create set of all stopwords
    stopword_set = set(w.lower() for w in nltk.corpus.stopwords.words())
    out = []
    for word in text:
        # Convert words to lower case alphabetical letters only
        # word = ''.join(w.lower() for w in word if w.isalpha())
        if word not in stopword_set:
            out.append(word)
    # Return only words that are not stopwords
    return(out)

    
#### Create a class that stores the NRC Word-Emotion Assocations dataset as a
#### a dictionary (once the word_association object is constructed), then 
#### provides the 'count_emotions' method to count the number occasions for 
#### emotion.
class word_assocations:
    
    def __init__(self):
        # Import NRC Word-Emotion Association data
        with open("NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt", "r", 
              newline = '', encoding = 'utf-8') as f:
            file = f.readlines()
        file = file[46:] # First 45 lines are comments

        # Create dictionary with words and their associated emotions
        associations = {}
        for line in file:
            elements = line.split()
            if elements[2] == '1':
                if elements[0] in associations:
                    associations[elements[0]].append(elements[1])
                else:
                    associations[elements[0]] = [elements[1]]

        # Initializes associations dictionary (so not to repeat it)
        self.associations = associations

    def count_emotions(self, text):
         # Clean up the string of characters
        temp0 = break_contractions(text)                                         # Break up contractions
        temp1 = lemmatize_words(temp0.split())                                   # Split string to words, then lemmatize
        temp2 = mark_negation(temp1, double_neg_flip = True)                     # Account for negations
        temp3 = remove_stopwords(temp2)                                          # Remove any stopwords
        
        # check_spelling(temp2)  # Function is no longer useful
        
        # Count number of emotional associations for each valid word
        bank = []
        wordcount = 0
        for word in temp3:
            if word in self.associations:
               bank.extend(self.associations[word])
               wordcount += 1

        # Returns a tuple of integers for negative, positive, anger, fear, anticipation,
        # surprise, trust, sadness, joy, disgust, and total word count, respectively.
        return((bank.count('negative'),
                bank.count('positive'),
                bank.count('anger'),
                bank.count('fear'),
                bank.count('anticipation'),
                bank.count('surprise'),
                bank.count('trust'),
                bank.count('sadness'),
                bank.count('joy'), 
                bank.count('disgust'),
                wordcount))       


# This function removes parentheses and also the contents of the parentheses
# for the purposes of improving search matches when finding lyrics.
def remove_parenth(text):
    
    patt = re.compile('\s*\(.*?\)\s*') 
    out = re.findall(patt, text)
    if len(out) > 0:
        text = text.replace(out[0], "")
    return(text)


# This function converts characters (byte string) that are otherwise
# not caught by the replace_accents normalization function.
def replace_special(text):
    
    temp1 = text.encode('utf-8')
    temp2 = temp1.replace(b"\xc3\x98", b"O")
    temp3 = temp2.replace(b"|", b"L")
    temp4 = temp3.decode()
    return(temp4)


# This function uses unicodedata to attempt to convert exotic characters, such 
# as accents, to a byte-friendly alternative that can be used in a url.
def replace_accents(text):
    
    temp1 = unicodedata.normalize('NFKD', text)
    temp2 = temp1.encode('ASCII', 'ignore')
    temp3 = temp2.decode()
    return(temp3)


# This function removes html comment text embedded inside the lyric text.
def remove_comments(text):
    
    patt = re.compile('(<!--.+?-->)')
    out = re.findall(patt, text)
    if len(out) > 0:
        temp = text.replace(out[0], "")
    else:
        temp = text
    return(temp)


# This function produces decimal text based on their integer code. This is
# needed to decode the lyrics during webscraping (which is in coded in decimal).
def decode_decimal(letters):
    
    iletters = []
    for i in letters:
        if len(i) < 4:
            iletters.append(int(i))
    lyrics = ""
    for i in iletters:
        lyrics = lyrics + chr(i)
    return(lyrics)


def getlyrics(track, artist):
       
    # Main regex search pattern
    Pattern = re.compile('lyricbox..>(.+?)<div class=..lyricsbreak')
    
    # Attempt initial search using the raw song and artist name
    url = "http://lyrics.wikia.com/wiki/" + artist + ":" + track
    url = remove_parenth(url)             # url: remove parentheses and its contents
    url = url.strip().replace(" ", "_")   # url: replace spaces with underscores
    url = replace_special(url)            # url: replace non-convertible special characters
    url = replace_accents(url)            # url: remove accents on characters
    req = urllib.request.Request(url)     # create Request object      
    print(req.get_full_url())             # print full url passed to urlopen
    
    try:
        data = urllib.request.urlopen(req)    # open site and pull html
        getdata = str(data.read())            # convert html to byte string
        output = re.findall(Pattern, getdata) # search suing main regex pattern 
        
        # If the search fails, but there is a recommended url:
        if len(output) == 0:
            patt = re.compile('Did you mean <a href=.(.+?)..title=')
            output = re.findall(patt, getdata)
            
            # If search still fails, but a redirect exists:
            if len(output) == 0:
                patt = re.compile('redirectText.><li><a href=.(.+?)..title=')
                output = re.findall(patt, getdata)                 
                
            url = "http://lyrics.wikia.com"        
            url = url + str(output[0])            # url: create new url
            url = url.strip().replace(" ", "_")   # url: replace spaces with underscores
            url = replace_special(url)            # url: replace non-convertible special characters
            url = replace_accents(url)            # url: remove accents on characters
            req = urllib.request.Request(url)     # url: create Request object
            print(req.get_full_url())             # print full url passed to urlopen
            
            data = urllib.request.urlopen(req)    # open site and pull html
            getdata = str(data.read())            # convert html to byte string
            output = re.findall(Pattern, getdata) # search using main regex pattern 
            
        text = remove_comments(output[0])         # data: remove html comments
        text = text.replace("<br />", "&#32;")    # data: replace breaks with spaces
        text = text.replace("<i>", "")            # data: remove italic formatting
        text = text.replace("</i>", "")           # data: remove italic formatting
        text = text.replace("&#", "")             # data: remove throwaway characters
        letters = text.split(sep = ";")           # data: split data based on semicolon
        letters.pop()                             # data: remove last element (always blank)
        lyrics = decode_decimal(letters)          # data: convert integers to decimal characters
        
        # Write to output file
        return(lyrics)
       
    # This is the last-resort case where there are no reasonable matches   
    except Exception:
        return('Not found')
        pass


# This function creates a string list of all days between the start and end dates
# including the start date, but excluding the end date
def days_between(start, end):
    # Start and end must be date objects
    delta = end - start
    out = []
    for i in range(0, delta.days):
        out.append(str(start + timedelta(i)))
    return(out)

    
# This function combines streaming data from spotifycharts.com based on the
# requested start and end dates (output is written to "spotifycharts.csv")
def spotify_charts(start, end):
    
    headers = ['Position', 'Track Name', 'Artist', 'Streams', 'URL', 'Date']
    
    # Write headers into output CSV file
    with open("spotifycharts.csv", "w", newline = '', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = headers)
        writer.writeheader()

    # Create string list of days between requested start and end dates
    datelist = days_between(start, end)
    
    # Collect CSV file for each date, and write to output file
    for i in datelist:
        # Open connection to URL
        url = 'https://spotifycharts.com/regional/us/daily/' + i + '/download'
        f = urllib.request.urlopen(url)
        output = pd.read_csv(f)
        
        for line in output.iterrows():
            with open("spotifycharts.csv", "a", newline = '', encoding = 'utf-8') as f:
                writer = csv.DictWriter(f, fieldnames = headers)
                writer.writerow({'Position': line[1][0],
                                 'Track Name': line[1][1],
                                 'Artist': line[1][2],
                                 'Streams': line[1][3],
                                 'URL': line[1][4],
                                 'Date': i})
        f.close() # Close connection

   
def spotify_charts_emotions():
    
    # Read the data
    df = pd.read_csv('spotifycharts.csv')
    
    # Create track name and artist concatenation (to determine unique tracks)
    df['name'] = df[['Track Name', 'Artist']].apply(lambda x: '--'.join(x), axis = 1)
    
    # Create flag variable for unique tracks
    bank = []       # Create a bank of unique uid's
    duplicates = [] # This will become a Boolean list: 0=First instance, 1=Duplicate
    for i in df['name']:
        if i not in bank:
            duplicates.append(0)
            bank.append(i)
        else:
            duplicates.append(1)
    
    df['Duplicates'] = duplicates

    # Create data frame of only unique tracks
    uniquetracks = df[['Track Name', 'Artist', 'name']].loc[df['Duplicates'] == 0]

    associator = word_assocations()

    headers = ['Track Name', 'Artist', 'name', 'negative', 'positive', 'anger', 
               'fear', 'anticipation', 'surprise', 'trust', 'sadness', 'joy', 
               'disgust', 'wordcount', 'lyrics', 'negative_percent', 'positive_percent',
               'anger_percent', 'fear_percent', 'anticipation_percent',
               'surprise_percent', 'trust_percent', 'sadness_percent', 'joy_percent',
               'disgust_percent']
    
    # Write headers into output CSV file
    with open("spotifychartsemotions.csv", "w", newline = '', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = headers)
        writer.writeheader()    
    
    for line in uniquetracks.iterrows():
        temp_track          = line[1][0]
        temp_artist         = line[1][1]
        temp_name           = line[1][2]
        temp_lyrics         = getlyrics(temp_track, temp_artist)
        temp_emotions       = associator.count_emotions(temp_lyrics)
        if temp_emotions[0] > 0:
            negative_percent    = temp_emotions[0] / temp_emotions[10]
            positive_percent    = temp_emotions[1] / temp_emotions[10]
            anger_percent       = temp_emotions[2] / temp_emotions[10]
            fear_percent        = temp_emotions[3] / temp_emotions[10]
            anticipation_percent= temp_emotions[4] / temp_emotions[10]
            surprise_percent    = temp_emotions[5] / temp_emotions[10]
            trust_percent       = temp_emotions[6] / temp_emotions[10]
            sadness_percent     = temp_emotions[7] / temp_emotions[10]
            joy_percent         = temp_emotions[8] / temp_emotions[10]
            disgust_percent     = temp_emotions[9] / temp_emotions[10]
            with open("spotifychartsemotions.csv", "a", newline = '', encoding = 'utf-8') as f:
                writer = csv.DictWriter(f, fieldnames = headers)
                writer.writerow({'Track Name': temp_track,
                                 'Artist': temp_artist,
                                 'name': temp_name,
                                 'negative': temp_emotions[0],
                                 'positive': temp_emotions[1],
                                 'anger': temp_emotions[2],
                                 'fear': temp_emotions[3],
                                 'anticipation': temp_emotions[4],
                                 'surprise': temp_emotions[5],
                                 'trust': temp_emotions[6],
                                 'sadness': temp_emotions[7],
                                 'joy': temp_emotions[8],
                                 'disgust': temp_emotions[9],
                                 'wordcount': temp_emotions[10], 
                                 'lyrics': temp_lyrics,
                                 'negative_percent': negative_percent,
                                 'positive_percent': positive_percent,
                                 'anger_percent': anger_percent,
                                 'fear_percent': fear_percent,
                                 'anticipation_percent': anticipation_percent,
                                 'surprise_percent': surprise_percent,
                                 'trust_percent': trust_percent,
                                 'sadness_percent': sadness_percent,
                                 'joy_percent': joy_percent,
                                 'disgust_percent': disgust_percent})

def articles_emotions():
    
    # Read the data
    df = pd.read_csv('getdayarticles.csv')
    
    associator = word_assocations()

    headers = ['id', 'header', 'date', 'location', 'categories', 'description', 
               'socialmediascore','negative', 'positive', 'anger', 
               'fear', 'anticipation', 'surprise', 'trust', 'sadness', 'joy', 
               'disgust', 'wordcount', 'negative_percent', 'positive_percent',
               'anger_percent', 'fear_percent', 'anticipation_percent',
               'surprise_percent', 'trust_percent', 'sadness_percent', 'joy_percent',
               'disgust_percent']
    
    # Write headers into output CSV file
    with open("getdayarticlesemotions.csv", "w", newline = '', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = headers)
        writer.writeheader()    
    
    for line in df.iterrows():
        temp_emotions = associator.count_emotions(line[1][1] + ' ' + line[1][5])
        if temp_emotions[0] > 0:
            negative_percent    = temp_emotions[0] / temp_emotions[10]
            positive_percent    = temp_emotions[1] / temp_emotions[10]
            anger_percent       = temp_emotions[2] / temp_emotions[10]
            fear_percent        = temp_emotions[3] / temp_emotions[10]
            anticipation_percent= temp_emotions[4] / temp_emotions[10]
            surprise_percent    = temp_emotions[5] / temp_emotions[10]
            trust_percent       = temp_emotions[6] / temp_emotions[10]
            sadness_percent     = temp_emotions[7] / temp_emotions[10]
            joy_percent         = temp_emotions[8] / temp_emotions[10]
            disgust_percent     = temp_emotions[9] / temp_emotions[10]
            with open("getdayarticlesemotions.csv", "a", newline = '', encoding = 'utf-8') as f:
                writer = csv.DictWriter(f, fieldnames = headers)
                writer.writerow({'id': line[1][0], 
                                 'header': line[1][1], 
                                 'date': line[1][2], 
                                 'location': line[1][3], 
                                 'categories': line[1][4], 
                                 'description': line[1][5], 
                                 'socialmediascore': line[1][6],
                                 'negative': temp_emotions[0],
                                 'positive': temp_emotions[1],
                                 'anger': temp_emotions[2],
                                 'fear': temp_emotions[3],
                                 'anticipation': temp_emotions[4],
                                 'surprise': temp_emotions[5],
                                 'trust': temp_emotions[6],
                                 'sadness': temp_emotions[7],
                                 'joy': temp_emotions[8],
                                 'disgust': temp_emotions[9],
                                 'wordcount': temp_emotions[10], 
                                 'negative_percent': negative_percent,
                                 'positive_percent': positive_percent,
                                 'anger_percent': anger_percent,
                                 'fear_percent': fear_percent,
                                 'anticipation_percent': anticipation_percent,
                                 'surprise_percent': surprise_percent,
                                 'trust_percent': trust_percent,
                                 'sadness_percent': sadness_percent,
                                 'joy_percent': joy_percent,
                                 'disgust_percent': disgust_percent})

                           
def articles_emotions_perday():
    
    # Read the data
    df = pd.read_csv('getdayarticlesemotions.csv')
    headers = ['date', 'anger_percent_weighted', 'sadness_percent_weighted', 'joy_percent_weighted']
    
    # Convert string to integer
    df['socialmediascore'] =  [int(x.replace(',', '')) for x in list((df['socialmediascore']))]          
    df['anger_percent_weighted'] = np.multiply(list(df['socialmediascore']), list(df['anger_percent']))     
    df['sadness_percent_weighted'] = np.multiply(list(df['socialmediascore']), list(df['sadness_percent']))     
    df['joy_percent_weighted'] = np.multiply(list(df['socialmediascore']), list(df['joy_percent']))     
    
    sums = df['socialmediascore'].groupby(df['date']).sum()
    anger_percent_weighted = df['anger_percent_weighted'].groupby(df['date']).sum()
    sadness_percent_weighted = df['sadness_percent_weighted'].groupby(df['date']).sum()
    joy_percent_weighted = df['joy_percent_weighted'].groupby(df['date']).sum()
    
    out = pd.concat([sums, anger_percent_weighted, sadness_percent_weighted, joy_percent_weighted], axis = 1)
    out['anger_percent_weighted'] = np.divide(list(out['anger_percent_weighted']), list(out['socialmediascore']))
    out['sadness_percent_weighted'] = np.divide(list(out['sadness_percent_weighted']), list(out['socialmediascore']))
    out['joy_percent_weighted'] = np.divide(list(out['joy_percent_weighted']), list(out['socialmediascore']))
    
    # Write headers into output CSV file
    with open("getdayarticlesemotionsperday.csv", "w", newline = '', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = headers)
        writer.writeheader()    
    
    for line in out.iterrows():
        with open("getdayarticlesemotionsperday.csv", "a", newline = '', encoding = 'utf-8') as f:
            writer = csv.DictWriter(f, fieldnames = headers)
            writer.writerow({'date': line[0][5:], 
                             'anger_percent_weighted': line[1][1],
                             'sadness_percent_weighted': line[1][2],
                             'joy_percent_weighted': line[1][3]})

     
#### Generates K-means centroids (as a CSV file) and also returns the labels            
def kmeans_centroids(df, var, k, name):
    
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(df[var])
    labels = kmeans.labels_  # Save labels for use later
    centroids = kmeans.cluster_centers_
    kmeansout = pd.DataFrame(centroids, columns = var) # Create dataframe of centroids
    kmeanscounts = pd.Series(labels, name = "Counts").value_counts() # Create number of points in each cluster
    kmeansout = pd.concat([kmeansout, kmeanscounts], axis = 1)
    kmeansout.to_csv(name, sep=',', index = True, header = True)
    return(labels)  # Return labels to be used later
    
 
#### Generates silhouette scores for K-means using a list of the number of clusters    
def kmeans_silhouette(df, var, k, name):
    
    with open(name, "w", newline = None, encoding = 'utf-8') as file:
        file.write("\n\nThe following are silhouette scores for K-means with varying number of K clusters: \n\n")
    
    with open(name, "a", newline = None, encoding = 'utf-8') as file:
        for c in k: 
            kmeans = KMeans(n_clusters = c)
            kmeans.fit(df[var])
            labels = kmeans.labels_
            file.write("For K=" + str(c) + ", the silhouette score is: " + str(silhouette_score(df[var], labels)) + "\n")

  
#### Generates Ward group means (as a CSV file) and also returns the labels                    
def ward_groupmeans(df, var, k, name):
    
    ward = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
    ward.fit(df[var])
    labels = ward.labels_  # Save labels for use later
    wardout = df[var].groupby(labels).mean()      # Create grouped means
    wardcounts = pd.Series(labels, name = "Counts").value_counts() # Create number of points in each cluster
    wardout = pd.concat([wardout, wardcounts], axis = 1)
    wardout.to_csv(name, sep=',', index = True, header = True) # Save to file
    return(labels)  # Return labels to be used later
            
            
#### Generates silhouette scores for Ward using a list of the number of clusters
def ward_silhouette(df, var, k, name):
    
    with open(name, "w", newline = None, encoding = 'utf-8') as file:
        file.write("\n\nThe following are silhouette scores for Ward's method with varying number of K clusters: \n\n")
    
    with open(name, "a", newline = None, encoding = 'utf-8') as file:
        for c in k: 
            ward = AgglomerativeClustering(n_clusters = c, linkage = 'ward')
            ward.fit(df[var])
            labels = ward.labels_
            file.write("For K=" + str(c) + ", the silhouette score is: " + str(silhouette_score(df[var], labels)) + "\n")
            

#### Generates 3D scatterplots   
def scatterplotclusters(df, var, labels, title, savename):
    
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111, projection = '3d')
    colors = cm.rainbow(np.linspace(0, 1, len(set(labels)))) # Use automatic color selection based on cluster count
    for name, group in df[var].groupby(labels):
       ax.scatter(group[var[0]], group[var[1]], group[var[2]], 
                  alpha = 0.8, c = colors[name], label = name)
    ax.set_xlabel(var[0])
    ax.set_ylabel(var[1])
    ax.set_zlabel(var[2])
    plt.title(title)
    ax.legend()
    plt.savefig(savename)
    plt.clf()
    plt.close()       

#### Pulls news articles from EventRegisty.org for each day within the specified range
def getdayarticles(start, end, directory, login_email, login_password):

    # Create CSV file with appropriate headers
    with open("getdayarticles.csv", "w", newline = '', encoding='utf-8') as f:
        fieldnames = ['id', 'header', 'date', 'location', 'categories', 'description', 'socialmediascore']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()  
    
    # Open new browser and login to eventregistry.org
    browser = webdriver.Firefox(firefox_binary = directory)
    browser.get("http://eventregistry.org/login?redirectUrl=%2FsearchEvents")
    time.sleep(5)
    username = browser.find_element_by_id("email") 
    password = browser.find_element_by_id("pass")
    username.send_keys(login_email)                                    # Enter email
    password.send_keys(login_password)                                 # Enter password
    browser.find_element_by_xpath('//*[@id="form-id"]/button').click() # Click submit
    time.sleep(5)
    
    for day in days_between(start, end):
   
        # Open new tab
        browser.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 't') 
        
        # Create URL based on day
        url = "http://eventregistry.org/searchEvents?query=%7B%22" + \
          "locations%22:%5B%7B%22label%22:%22United%20States" + \
          "%22,%22uri%22:%22http:%2F%2Fen.wikipedia.org%2Fwiki%2F" + \
          "United_States%22,%22negate%22:false%7D%5D,%22dateStart%22:%22" + \
          day + "%22,%22dateEnd%22:%22" + \
          day + "%22,%22lang%22:%22eng%22,%22minArticles%22:50,%22" + \
          "preferredLang%22:%22eng%22%7D&tab=events"
          
        browser.get(url) # Open URL
        time.sleep(50)   # Wait 20 seconds for page to load
        
        # Click "sort events by social media hotness" to get most popular events
        browser.find_element_by_xpath('//*[@id="tab-events"]/div/div/div[3]/div[2]/div/div[2]/button[4]').click()
        time.sleep(5)    # Wait 5 seconds for page to reload
        
        out = browser.page_source.encode("utf-8") # Save source code
        
        # Save social media score for each news event
        temp1 = BeautifulSoup(out, "lxml").findAll("span", {'class': "score ng-binding"})
        socialmedia = []
        for i in temp1:
            socialmedia.append(i.contents[0])
        
        # Save header for each news event
        temp2 = BeautifulSoup(out, "lxml").findAll("h4", {'class': "media-heading"})
        articleheader = []
        for i in temp2:
            articleheader.append(i.contents[0].contents[0])
        
        # Save time and date and location for each news event
        temp3 = BeautifulSoup(out, "lxml").findAll("span", {'class': "info-val ng-binding"})
        timedate = []
        for i in temp3:
            timedate.append(i.contents[0])
        dates = timedate[::2]
        location = timedate[1::2]
        
        # Save categories for each news event
        temp4 = BeautifulSoup(out, "lxml").findAll("div", {'class': "categories"})
        categories = []
        for i in temp4:
            k = i.findAll("span", {'class': "ng-binding"})
            t = []
            for j in k:
                t.append(j.contents[0].replace('→',', '))
            categories.append(t)
        
        # Save description of each news event
        temp5 = BeautifulSoup(out, "lxml").findAll("div", {'class': "lighter smaller ng-binding"})
        description = []
        for i in temp5:
            description.append(i.contents[0])
            
        # Save news event ID
        temp6 = BeautifulSoup(out, "lxml").find_all("a", {'target': "_blank", 'class': "ng-binding"}, href=True)
        eventids = []
        for i in temp6:
            eventids.append(i['href'])
        eventids = eventids[1:] # Remove first element (contains no information)
        ids = []
        for i in eventids:
            ids.append(re.findall('/event/(.......).lang=eng', i)[0])
    
        articles = pd.DataFrame([ids, articleheader, dates, location, categories, description, socialmedia])
        
        for j in range(0, articles.shape[1]):    
            # Write to output file
            with open("getdayarticles.csv", "a", newline = '', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames = fieldnames)
                writer.writerow({'id': articles[j][0], 
                                 'header': articles[j][1],
                                 'date': articles[j][2],
                                 'location': articles[j][3],
                                 'categories': articles[j][4],
                                 'description': articles[j][5],
                                 'socialmediascore': articles[j][6]})    
    
    browser.quit()

#### Applies the longitudinal multi-layer perceptron model with 100 hidden layers, number of lags of the 
#### predictor value and returns the predicted values. It also plots the results as 'name.png'.
def nn_tester(df, predictor, predicted, lag, name):

    length = lag
    start = 0
    temp = df[[predictor, predicted]]
    iterations = len(temp[predicted])
    X = pd.DataFrame(np.zeros(length)).T
    y = [0]
    for i in range(length, iterations):
        temp_y = temp[predicted][i]
        temp_X = pd.DataFrame(temp[predictor][start:(i)]).T.reset_index(drop = True)
        temp_X.columns = [x for x in range(0, lag)]
        y.extend([temp_y])
        X = pd.concat([X, temp_X])
        start = start + 1
    X.reset_index(inplace = True, drop = True)
    X.drop(X.index[[0]], inplace = True)
    X.reset_index(inplace = True, drop = True)
    y = y[1:]
    X_train = X[0:100]   # Training set
    X_test = X[100:]     # Test set
    y_train = y[0:100]   # Training set
    y_test = y[100:]     # Test set
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPRegressor(activation = 'logistic', solver = 'lbfgs', max_iter = 10000, tol = 1e-5,
                       alpha = .01, hidden_layer_sizes = (100,), random_state = 1)
    mlp.fit(X_train, y_train)
    print(mlp.score(X_test, df[predicted][-len(y_test):]))    
    
    plt.figure(figsize = (20, 12))
    plot_pred = plt.plot(df['date'][lag:], mlp.predict(scaler.transform(X)))
    plot_ytest = plt.plot(df['date'][lag:], y)
    plt.setp(plot_pred, color = 'black', linestyle = '--', linewidth = 1.0)
    plt.setp(plot_ytest, color = 'black', linewidth = 1.0)
    plt.figtext(.8, .85, "R-Squared = " + 
                str(mlp.score(X_test, df[predicted][-len(y_test):]).round(3)), fontsize = 12)
    plt.axvline(df['date'][len(df['date']) - len(y_test) - 1], color = 'r', linewidth = 1.0)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation = 90)
    plt.savefig(name)
    plt.clf()
    plt.close()
    return(mlp.predict(scaler.transform(X)))


    
    
    
def main():
    
    ##################################################################################
    # The following lines of code can be uncommented and run for test purposes, but 
    # we recommend running them with a smaller date window. It should also be noted
    # EventRegistry.org (the site where the news article data were pulled) is currently
    # undergoing restructuring due to Google's move to fund their project -- so it is
    # possible that the results will incomplete if run at the current time.
    #
    # IMPORTANT: The getdayarticles() function requires the installation of the 
    # selenium Python package (through pip), the geckodriver application (which is
    # included in the with this code for Windows), and a valid installation of Firefox
    # along with the directory to its application (which needs to be placed in the 
    # 'directory' argument).
    ##################################################################################
    #
    # start_date = date(2016, 6, 1)
    # end_date = date(2016, 11, 21)
    # spotify_charts(start_date, end_date)
    # spotify_charts_emotions()
    # getdayarticles(start = start_date, 
    #                end = end_date, 
    #                directory = r'C:\Program Files (x86)\Mozilla Firefox\firefox.exe', 
    #                login_email = "jmc511@georgetown.edu", 
    #                login_password = "password123")
    # articles_emotions()
    # articles_emotions_perday()
    #
    ##################################################################################
    
    
    ##### Clustering for Songs #####
    
    # Import data
    df = pd.read_csv('spotifychartsemotions.csv')
    
    # Silhouette scores
    kmeans_silhouette(df, var = ['anger_percent', 'sadness_percent', 'joy_percent'], 
                      k = [2,3,4,5,6,7,8,9,10], name = "kmeans_silhouettescores_songs.txt")
    ward_silhouette(df, var = ['anger_percent', 'sadness_percent', 'joy_percent'], 
                    k = [2,3,4,5,6,7,8,9,10], name = "ward_silhouettescores_songs.txt")
    
    # Get labels for K-means and Ward
    labels1 = kmeans_centroids(df, var = ['anger_percent', 'sadness_percent', 'joy_percent'], k = 3,
                               name = 'kmeans_centroids_songs.csv')        
    labels2 = ward_groupmeans(df, var = ['anger_percent', 'sadness_percent', 'joy_percent'], k = 3,
                               name = 'ward_groupedmeans_songs.csv')
    
    # Plot 3D scatterplot
    scatterplotclusters(df = df, var = ['anger_percent', 'sadness_percent', 'joy_percent'], 
                        labels = labels1, 
                        title = 'K-Means Scatterplot by Cluster', 
                        savename = "kmeans_3Dscatterplot_songs")
    scatterplotclusters(df = df, var = ['anger_percent', 'sadness_percent', 'joy_percent'], 
                        labels = labels2, 
                        title = 'Ward Scatterplot by Cluster', 
                        savename = "ward_3Dscatterplot_songs")    
    
 
    ##### Group Song longitudinal by Clusters #####
                    
    # Import data                     
    df2 = pd.read_csv('spotifycharts.csv')
    df2['name'] = df2[['Track Name', 'Artist']].apply(lambda x: '--'.join(x), axis = 1)

    # Use labels and group longitudinal data by clusters
    positive_tracks = df['name'].loc[labels1 == 1]
    negative_tracks = df['name'].loc[labels1 == 0]
    null_tracks = df['name'].loc[labels1 == 2]
    positive_tracks_labels = df2.loc[df2['name'].isin(positive_tracks)]
    negative_tracks_labels = df2.loc[df2['name'].isin(negative_tracks)]
    null_tracks_labels = df2.loc[df2['name'].isin(null_tracks)]

                                                                                               
    ##### Get average stream counts for each of emotion class #####
    
    positive_grouped = positive_tracks_labels.groupby('Date').mean()
    negative_grouped = negative_tracks_labels.groupby('Date').mean()
    null_grouped = null_tracks_labels.groupby('Date').mean()
    emotion_grouped = pd.concat([positive_grouped['Streams'], 
                                 negative_grouped['Streams'],
                                 null_grouped['Streams']], axis = 1,
                                keys = ['positive_grouped', 'negative_grouped', 'null_grouped'])
    emotion_grouped['date'] = emotion_grouped.index.values
    emotion_grouped = emotion_grouped.reset_index(drop = True)
    
    # Get emotion percentages
    article_emotions = pd.read_csv('getdayarticlesemotionsperday.csv')
    article_emotions['date'] = [datetime.datetime.strptime(x, '%B %d, %Y') for x in article_emotions['date']]
    article_emotions = article_emotions.sort_values('date')
    article_emotions['date'] = [str(x)[:10] for x in article_emotions['date']]
    article_emotions = article_emotions.reset_index(drop = True)
    
    # Merge data for plotting
    plotdata = pd.merge(emotion_grouped, article_emotions[['anger_percent_weighted', 'sadness_percent_weighted', 
                                                           'joy_percent_weighted', 'date']], 
                        on = 'date', how = 'left')
    plotdata.fillna(0, inplace = True) # Some article percentages are NaN, so convert these to zero
    plotdata['date'] = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in plotdata['date']]
    plotdata.to_csv('emotion_analysis.csv', sep=',', index = True, header = True) # Write to CSV
       

    ##### Line Plots (matplotlib images) #####
    
    # Import data
    plotdata = pd.read_csv('emotion_analysis.csv')
    plotdata['date'] = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in plotdata['date']]

    # Plot of news article emotion percentages    
    plt.figure(figsize = (10, 8))
    anger_article = plt.plot(plotdata['date'], plotdata['anger_percent_weighted'])
    sadness_article = plt.plot(plotdata['date'], plotdata['sadness_percent_weighted'])
    joy_article = plt.plot(plotdata['date'], plotdata['joy_percent_weighted'])
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, .6))
    plt.setp(anger_article, color = 'r', linewidth = 1.0)
    plt.setp(sadness_article, color = 'b', linewidth = 1.0)
    plt.setp(joy_article, color = 'g', linewidth = 1.0)    
    plt.tick_params(axis = 'y', which = 'major', labelsize = 10)
    plt.tick_params(axis = 'y', which = 'minor', labelsize = 10)
    plt.tick_params(axis = 'x', which = 'major', labelsize = 9)
    plt.tick_params(axis = 'x', which = 'minor', labelsize = 9)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation = 90)
    plt.legend()
    plt.savefig('emotion_articles')
    plt.clf()
    plt.close()
    
    # Plot of average song streaming counts by emotion
    plt.figure(figsize = (10, 8))
    positive_streamed = plt.plot(plotdata['date'], plotdata['positive_grouped'])
    negative_streamed = plt.plot(plotdata['date'], plotdata['negative_grouped'])
    null_streamed = plt.plot(plotdata['date'], plotdata['null_grouped'])
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 500000))
    plt.setp(positive_streamed, color = 'g', linewidth = 1.0)
    plt.setp(negative_streamed, color = 'r', linewidth = 1.0)
    plt.setp(null_streamed, color = 'black', linewidth = 1.0)
    plt.tick_params(axis = 'y', which = 'major', labelsize = 10)
    plt.tick_params(axis = 'y', which = 'minor', labelsize = 10)
    plt.tick_params(axis = 'x', which = 'major', labelsize = 9)
    plt.tick_params(axis = 'x', which = 'minor', labelsize = 9)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation = 90)
    plt.legend()
    plt.savefig('emotion_streamed')
    plt.clf()
    plt.close()
   
    
    ##### Hypothesis Tests #####
    
    # Perform Granger Causality Tests (use sys.stdout to capture printed outputs)
    
    plotdata.set_index(keys = plotdata['date'], inplace = True)    # Set date as index

    # Article percentages are all stationary (ADF should be significant)
    print(adfuller(plotdata['anger_percent_weighted'], autolag = 'bic', 
                   regression = 'ct', maxlag = 10))
    print(adfuller(plotdata['sadness_percent_weighted'], autolag = 'bic', 
                   regression = 'ct', maxlag = 10))
    print(adfuller(plotdata['joy_percent_weighted'], autolag = 'bic', 
                   regression = 'ct', maxlag = 10))
    
    # Make positive_grouped stationary via moving average
    moving_avg = pd.rolling_mean(plotdata['positive_grouped'], 6)
    plotdata['positive_grouped_ma'] = plotdata['positive_grouped'] - moving_avg
    print(adfuller(plotdata['positive_grouped_ma'].dropna(), autolag = 'bic', 
                   regression = 'ct', maxlag = 10))

    # Make positive_grouped stationary via moving average
    moving_avg = pd.rolling_mean(plotdata['negative_grouped'], 6)
    plotdata['negative_grouped_ma'] = plotdata['negative_grouped'] - moving_avg
    print(adfuller(plotdata['negative_grouped_ma'].dropna(), autolag = 'bic', 
                   regression = 'ct', maxlag = 10)) 
    
    # Make null_grouped stationary via moving average
    moving_avg = pd.rolling_mean(plotdata['null_grouped'], 6)
    plotdata['null_grouped_ma'] = plotdata['null_grouped'] - moving_avg
    print(adfuller(plotdata['null_grouped_ma'].dropna(), autolag = 'bic', 
                   regression = 'ct', maxlag = 10))    
    
    
    # Perform Granger tests using ma variables (and save t0 grangertests.txt)
    former, sys.stdout = sys.stdout, open('grangertests.txt', 'w')
  
    print('\n\nOutput for Granger: anger_percent_weighted Granger causes positive_grouped_ma\n')
    grangercausalitytests(plotdata[['positive_grouped_ma', 'anger_percent_weighted']].dropna(), maxlag = 7)
    
    print('\n\nOutput for Granger: sadness_percent_weighted Granger causes positive_grouped_ma\n')
    grangercausalitytests(plotdata[['positive_grouped_ma', 'sadness_percent_weighted']].dropna(), maxlag = 7)    
    
    print('\n\nOutput for Granger: joy_percent_weighted Granger causes positive_grouped_ma\n')
    grangercausalitytests(plotdata[['positive_grouped_ma', 'joy_percent_weighted']].dropna(), maxlag = 7)    
        
    print('\n\nOutput for Granger: anger_percent_weighted Granger causes negative_grouped_ma\n')
    grangercausalitytests(plotdata[['negative_grouped_ma', 'anger_percent_weighted']].dropna(), maxlag = 7)
    
    print('\n\nOutput for Granger: sadness_percent_weighted Granger causes negative_grouped_ma\n')
    grangercausalitytests(plotdata[['negative_grouped_ma', 'sadness_percent_weighted']].dropna(), maxlag = 7)    
    
    print('\n\nOutput for Granger: joy_percent_weighted Granger causes negative_grouped_ma\n')
    grangercausalitytests(plotdata[['negative_grouped_ma', 'joy_percent_weighted']].dropna(), maxlag = 7)      
    
    print('\n\nOutput for Granger: anger_percent_weighted Granger causes null_grouped_ma\n')
    grangercausalitytests(plotdata[['null_grouped_ma', 'anger_percent_weighted']].dropna(), maxlag = 7)
    
    print('\n\nOutput for Granger: sadness_percent_weighted Granger causes null_grouped_ma\n')
    grangercausalitytests(plotdata[['null_grouped_ma', 'sadness_percent_weighted']].dropna(), maxlag = 7)    
    
    print('\n\nOutput for Granger: joy_percent_weighted Granger causes null_grouped_ma\n')
    grangercausalitytests(plotdata[['null_grouped_ma', 'joy_percent_weighted']].dropna(), maxlag = 7)      
    
    results, sys.stdout = sys.stdout, former
    results.close()

    
    ##### Prediction using Neural Networks #####
    
    # Run multi-layer perceptron with 100 hidden units, alpha = .01, lbfgs optimizer, and
    # logistic activation function. Functions return predicted values.
    lag = 7
    nn_positive_anger = nn_tester(df = plotdata, predictor = 'anger_percent_weighted', 
                                  predicted = 'positive_grouped', lag = lag, 
                                  name = 'nn_positive_anger')
    nn_positive_sadness = nn_tester(df = plotdata, predictor = 'sadness_percent_weighted', 
                                    predicted = 'positive_grouped', lag = lag, 
                                    name = 'nn_positive_sadness')    
    nn_positive_joy = nn_tester(df = plotdata, predictor = 'joy_percent_weighted', 
                                predicted = 'positive_grouped', lag = lag, 
                                name = 'nn_positive_joy')        
    nn_negative_anger = nn_tester(df = plotdata, predictor = 'anger_percent_weighted', 
                                  predicted = 'negative_grouped', lag = lag, 
                                  name = 'nn_negative_anger')
    nn_negative_sadness = nn_tester(df = plotdata, predictor = 'sadness_percent_weighted', 
                                    predicted = 'negative_grouped', lag = lag, 
                                    name = 'nn_negative_sadness')    
    nn_negative_joy = nn_tester(df = plotdata, predictor = 'joy_percent_weighted', 
                                predicted = 'negative_grouped', lag = lag, 
                                name = 'nn_negative_joy')      
    nn_null_anger = nn_tester(df = plotdata, predictor = 'anger_percent_weighted', 
                              predicted = 'null_grouped', lag = lag, 
                              name = 'nn_null_anger')
    nn_null_sadness = nn_tester(df = plotdata, predictor = 'sadness_percent_weighted', 
                                predicted = 'null_grouped', lag = lag, 
                                name = 'nn_null_sadness')    
    nn_null_joy = nn_tester(df = plotdata, predictor = 'joy_percent_weighted', 
                            predicted = 'null_grouped', lag = lag, 
                            name = 'nn_null_joy')        
    
    
    ##### Interactive plot of findings (bokeh) #####
    
    # x-axis
    x = plotdata['date'][lag:] 
    
    # Different y-axes
    y1_1 = plotdata['positive_grouped'][lag:]
    y1_2 = nn_positive_anger
    y2_1 = plotdata['positive_grouped'][lag:]
    y2_2 = nn_positive_sadness
    y3_1 = plotdata['positive_grouped'][lag:]
    y3_2 = nn_positive_joy
    y4_1 = plotdata['negative_grouped'][lag:]
    y4_2 = nn_negative_anger
    y5_1 = plotdata['negative_grouped'][lag:]
    y5_2 = nn_negative_sadness
    y6_1 = plotdata['negative_grouped'][lag:]
    y6_2 = nn_negative_joy
    y7_1 = plotdata['null_grouped'][lag:]
    y7_2 = nn_null_anger
    y8_1 = plotdata['null_grouped'][lag:]
    y8_2 = nn_null_sadness
    y9_1 = plotdata['null_grouped'][lag:]
    y9_2 = nn_null_joy

    # Plot predictions for Average Positive Stream
    output_file("plots1.html", title = "Prediction of Song Playcounts")
    s1 = figure(width = 900, plot_height = 300, title = "Positive Streams by Article Anger", 
                x_axis_type = "datetime", tools="pan,wheel_zoom,box_zoom,reset")
    s1.line(x, y1_1, color = 'green', legend = "Average Positive Stream Count", line_width = 2,
            line_alpha = 0.7)
    s1.line(x, y1_2, color = 'black', line_alpha = 0.7,
            legend = "Predicted by Percent Anger", line_width = 2, line_dash = 'dotted')
    s1.left[0].formatter.use_scientific = False
    vline1 = Span(location = plotdata['date'][lag + 100 - 1].timestamp()*1000, 
                 dimension = 'height', line_color = 'red', line_width = 1)    
    s1.renderers.extend([vline1])
    s2 = figure(width = 900, plot_height = 300, title = "Positive Streams by Article Sadness", 
                x_axis_type = "datetime", tools="pan,wheel_zoom,box_zoom,reset")
    s2.line(x, y2_1, color = 'green', legend = "Average Positive Stream Count", line_width = 2,
            line_alpha = 0.7)
    s2.line(x, y2_2, color = 'black', line_alpha = 0.7,
            legend = "Predicted by Percent Sadness", line_width = 2, line_dash = 'dotted')
    s2.left[0].formatter.use_scientific = False
    vline2 = Span(location = plotdata['date'][lag + 100 - 1].timestamp()*1000, 
                 dimension = 'height', line_color = 'red', line_width = 1)    
    s2.renderers.extend([vline2])
    s3 = figure(width = 900, plot_height = 300, title = "Positive Streams by Article Joy", 
                x_axis_type = "datetime", tools="pan,wheel_zoom,box_zoom,reset")
    s3.line(x, y3_1, color = 'green', legend = "Average Positive Stream Count", line_width = 2,
            line_alpha = 0.7)
    s3.line(x, y3_2, color = 'black', line_alpha = 0.7,
            legend = "Predicted by Percent Joy", line_width = 2, line_dash = 'dotted')
    s3.left[0].formatter.use_scientific = False
    vline3 = Span(location = plotdata['date'][lag + 100 - 1].timestamp()*1000, 
                 dimension = 'height', line_color = 'red', line_width = 1)    
    s3.renderers.extend([vline3])    
    p = vplot(s1, s2, s3)
    show(p)    
    
    # Plot predictions for Average Negative Stream
    output_file("plots2.html", title = "Prediction of Song Playcounts")
    s4 = figure(width = 900, plot_height = 300, title = "Negative Streams by Article Anger", 
                x_axis_type = "datetime", tools="pan,wheel_zoom,box_zoom,reset")
    s4.line(x, y4_1, color = 'red', legend = "Average Negative Stream Count", line_width = 2,
            line_alpha = 0.7)
    s4.line(x, y4_2, color = 'black', line_alpha = 0.7,
            legend = "Predicted by Percent Anger", line_width = 2, line_dash = 'dotted')
    s4.left[0].formatter.use_scientific = False
    vline4 = Span(location = plotdata['date'][lag + 100 - 1].timestamp()*1000, 
                 dimension = 'height', line_color = 'red', line_width = 1)    
    s4.renderers.extend([vline4])
    s5 = figure(width = 900, plot_height = 300, title = "Negative Streams by Article Sadness", 
                x_axis_type = "datetime", tools="pan,wheel_zoom,box_zoom,reset")
    s5.line(x, y5_1, color = 'red', legend = "Average Negative Stream Count", line_width = 2,
            line_alpha = 0.7)
    s5.line(x, y5_2, color = 'black', line_alpha = 0.7,
            legend = "Predicted by Percent Sadness", line_width = 2, line_dash = 'dotted')
    s5.left[0].formatter.use_scientific = False
    vline5 = Span(location = plotdata['date'][lag + 100 - 1].timestamp()*1000, 
                 dimension = 'height', line_color = 'red', line_width = 1)    
    s5.renderers.extend([vline5])
    s6 = figure(width = 900, plot_height = 300, title = "Negative Streams by Article Joy", 
                x_axis_type = "datetime", tools="pan,wheel_zoom,box_zoom,reset")
    s6.line(x, y6_1, color = 'red', legend = "Average Negative Stream Count", line_width = 2,
            line_alpha = 0.7)
    s6.line(x, y6_2, color = 'black', line_alpha = 0.7,
            legend = "Predicted by Percent Joy", line_width = 2, line_dash = 'dotted')
    s6.left[0].formatter.use_scientific = False
    vline6 = Span(location = plotdata['date'][lag + 100 - 1].timestamp()*1000, 
                 dimension = 'height', line_color = 'red', line_width = 1)    
    s6.renderers.extend([vline6])       
    p = vplot(s4, s5, s6)
    show(p)    
    
    # Plot predictions for Average Null Stream
    output_file("plots3.html", title = "Prediction of Song Playcounts")    
    s7 = figure(width = 900, plot_height = 300, title = "Null Streams by Article Anger", 
                x_axis_type = "datetime", tools="pan,wheel_zoom,box_zoom,reset")
    s7.line(x, y7_1, color = 'black', legend = "Average Null Stream Count", line_width = 2,
            line_alpha = 0.7)
    s7.line(x, y7_2, color = 'black', line_alpha = 0.7,
            legend = "Predicted by Percent Anger", line_width = 2, line_dash = 'dotted')
    s7.left[0].formatter.use_scientific = False
    vline7 = Span(location = plotdata['date'][lag + 100 - 1].timestamp()*1000, 
                 dimension = 'height', line_color = 'red', line_width = 1)    
    s7.renderers.extend([vline7])
    s8 = figure(width = 900, plot_height = 300, title = "Null Streams by Article Sadness", 
                x_axis_type = "datetime", tools="pan,wheel_zoom,box_zoom,reset")
    s8.line(x, y8_1, color = 'black', legend = "Average Null Stream Count", line_width = 2,
            line_alpha = 0.7)
    s8.line(x, y8_2, color = 'black', line_alpha = 0.7,
            legend = "Predicted by Percent Sadness", line_width = 2, line_dash = 'dotted')
    s8.left[0].formatter.use_scientific = False
    vline8 = Span(location = plotdata['date'][lag + 100 - 1].timestamp()*1000, 
                 dimension = 'height', line_color = 'red', line_width = 1)    
    s8.renderers.extend([vline8])
    s9 = figure(width = 900, plot_height = 300, title = "Null Streams by Article Joy", 
                x_axis_type = "datetime", tools="pan,wheel_zoom,box_zoom,reset")
    s9.line(x, y9_1, color = 'black', legend = "Average Null Stream Count", line_width = 2,
            line_alpha = 0.7)
    s9.line(x, y9_2, color = 'black', line_alpha = 0.7,
            legend = "Predicted by Percent Joy", line_width = 2, line_dash = 'dotted')
    s9.left[0].formatter.use_scientific = False
    vline9 = Span(location = plotdata['date'][lag + 100 - 1].timestamp()*1000, 
                 dimension = 'height', line_color = 'red', line_width = 1)    
    s9.renderers.extend([vline9])       
    p = vplot(s7, s8, s9)
    show(p)
    
    
if __name__ == "__main__":
    
    main()