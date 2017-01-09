# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:29:46 2016

@author: tomec
"""

import pandas as pd
import numpy as np
import ast
from collections import Counter
import nltk
import re
from bs4 import BeautifulSoup
import time
import requests
import csv
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from nltk.sentiment.util import mark_negation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pandas.tools.plotting import scatter_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import ttest_ind
from statsmodels.formula.api import ols
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from scipy.stats import chisquare

#### Create a class that helps produce association rules
class associationrules:
    
    # Create occurrence matrix based on minimum support and a list of lists
    # These eliminate singleton items that occur less than the minimum support.
    def __init__(self, listoflists, minsup):
        
        length = len(listoflists) # Keep length of list on hand
        
        # Create list of all possible contents of listoflists:
        allelements = [] # Create a list to store elements of 'toptag'
        for i in listoflists:
            allelements.extend(i)
        
        # Count how many times each unique item in allelements shows up. Then use
        # minsup to eliminate those that are not valid.
        count = Counter(allelements) # Create counter object
        mostcommon = count.most_common()
        validtags = []
        for i in mostcommon:
            if i[1] >= minsup:
                validtags.append(i[0])
        
        output = pd.Series(range(0, length), name = 'number') # temporary index to start concat
        # Create occurrence matrix: for each item that fulfills minimum support, create a 
        # Boolean column for each row of listsoflists containing True if has the item present,
        # and false if not present.
        for tag in validtags:
            temp = pd.Series([tag in x for x in listoflists], name = tag)
            output = pd.concat([output, temp], axis = 1)
        del output['number'] # delete temporary index
        
        # Keep desired outputs
        self.validtags = validtags
        self.occurrences = output
        self.minsup = minsup
        self.listoflists = listoflists

    # Calculate confidence based on list of precedents and list of antecedents
    def confidence(self, precedent, antecedent):
        # Denominator is based on a subset of the occurrences matrix containing
        # only the precedent columns, and summing up the number of True cells. If the 
        # number of True cells is the same as the length of the precedent, it is considered
        # valid. The sum of valid cases comprises the denominator.
        denom = sum(self.occurrences[precedent].apply(sum, axis = 1) == len(precedent))
        # Numerator is based on a subset of the occurrences matrix containing
        # only the precedent and antecedent columns, and summing up the number of True cells.  
        # If the number of True cells is the same as the length of both the precedent and the 
        # antecedent, it is considered valid. The sum of valid cases comprises the numerator.
        numer = sum(self.occurrences[precedent + antecedent].apply(sum, axis = 1) == len(precedent + antecedent))
        return([numer, denom])    
    
    def associationcombtest(self, maxitems, minconf, filename):
     
        with open(filename, "w", newline = None, encoding = 'utf-8') as file:
            file.write("\n\nThe following tags had minimum support >= " + str(self.minsup) + ": \n")
            
        with open(filename, "a", newline = None, encoding = 'utf-8') as file:
            for i in self.validtags:
                file.write(i + "\n")
        
        with open(filename, "a", newline = None, encoding = 'utf-8') as file:
            file.write("\n\nThe following are support scores for association rules with minimum confidence >= " + 
                       str(minconf) + ": \n\n" + 'CONF'.ljust(5, " ") + " " + 'NUMER'.ljust(9, " ") + 
                       " " + 'DENOM'.ljust(9, " ") + " RULE" + "\n" )                    
        with open(filename, "a", newline = None, encoding = 'utf-8') as file:
            for k in range(1, maxitems + 1):
                combin_length = k
                # Get all possible combinations of validtags (outputs as list of tuples)
                combin = list(combinations(self.validtags, combin_length))
                combin = [list(i) for i in combin] # Convert list of tuples to list of lists
                # Loop through every combination and each possible antecedent
                for i in combin:
                    # First, check if the support of combin is greater than minsup
                    if sum(self.occurrences[i].apply(sum, axis = 1) == len(i)) >= self.minsup:
                        for j in self.validtags:
                            if j not in i:
                                print("Processing: " + str(i) + " --> " + j)
                                conf = self.confidence(precedent = i, antecedent = [j])
                                if (conf[0]/conf[1]) >= minconf: 
                                    temp1 = str(np.around((conf[0]/conf[1]), decimals = 3)).ljust(5, "0")
                                    temp2 = str(conf[0]).ljust(9, " ")
                                    temp3 = str(conf[1]).ljust(9, " ")
                                    out = temp1 + " " + temp2 + " " + temp3 + " " + str(i) + " --> " + j + "\n"
                                    file.write(out)
        

#### Adds a new variable that is a discretized version of existing variable
def binning(df, variable, levels):            
   
    df[variable + '_binned'] = pd.cut(df[variable], levels, include_lowest = True)
    with open("binning.txt", "w", newline = None, encoding = 'utf-8') as file:
        file.write("\n\nThe following are the first 5 rows of the newly binned and original data" + variable + ": \n\n")
    
    with open("binning.txt", "a", newline = None, encoding = 'utf-8') as file:
        for i in range(0, 5): 
            file.write(df[variable + '_binned'][i] + " " + str(df[variable][i]) + "\n")    


### Create function to break apart contractions to its derivative words. Only takes
### in a single string (probably a whole sentence or document). Lower-case conversion
### is applied. A text file containing this('contractions.txt') should be located at 
### the working directory along with this script.
def break_contractions(text):
    
    #### Import dictionary of contractions: contractions.txt
    with open('contractions.txt','r') as inf:
        contractions = eval(inf.read())
    # Create regex pattern based on contractions dictionary
    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')
    text = ''.join(w.lower() for w in text) # Convert all characters to lower case
    # Substitute any substring matching the regex pattern with the appropriate
    # non-contraction word.
    result = pattern.sub(lambda x: contractions[x.group()], text)
    return(result)


#### Create a function that imports data files from Project Part 1, then removes
#### missing values, and adds several variables. 
def cleandata():
 
    ### Merge output files from PROJECT PART 1
    df1 = pd.read_csv('gettracks_output_newfeatures.csv')
    df2 = pd.read_csv('getlyrics_output_newfeatures.csv')
    df3 = pd.read_csv('getinfo_output.csv')
    
    df1 = df1.drop(['duration', 'listeners'], 1)     # Drop 2 columns, which are redundant
    df2 = df2.drop('url', 1)                         # Drop the 'url' column, which is redundant
    df3 = df3.drop(['url', 'name', 'artistname'], 1) # Drop 3 columns, which are redundant
    df = pd.concat([df1, df2, df3], axis=1) # Merge the three datasets together
    
    ########### Removal of Missing Values ###########
    
    prelen = len(df) # Keep length original dataframe (prior to removals)
    
    # Save text file containing number of missing (or equivalent) cases
    with open("missingdata.txt", "w", newline = None, encoding = 'utf-8') as file:
        # Write header rows
        file.write("\n\nThe following were considered missing values and were removed: \n\n")
        file.write('{:>50} {:>10}\n'.format("Duplicate songs:", str(prelen - len(df.loc[df.duplicate != 1]))))
        file.write('{:>50} {:>10}\n'.format("Lyrics is 'not found':", str(prelen - len(df.loc[df.lyrics != 'Not found']))))
        file.write('{:>50} {:>10}\n'.format("Number of characters in lyriclength < 35:", str(prelen - len(df.loc[df.lyricslength > 34]))))
        file.write('{:>50} {:>10}\n'.format("Duration is zero:", str(prelen - len(df.loc[df.duration != 0]))))
       
    df = df.loc[df.duplicate != 1]        # Remove rows that were previously flagged as duplicates
    df = df.loc[df.lyrics != 'Not found'] # Remove rows where no lyrics were found
    df = df.loc[df.lyricslength > 34]     # Remove rows where lyrics are "instrumentals" or similar
    df = df.loc[df.duration != 0]         # Remove rows where duration is zero
    df = df.reset_index(drop=True)        # Reset dataframe index

    with open("missingdata.txt", "a", newline = None, encoding = 'utf-8') as file:
        # Write header rows
        file.write('{:>50} {:>10}\n'.format("Total removed:", str(prelen - len(df))))

        
    ########### Add new variables ###########
    associator = word_assocations()  # Initialize word association object (can take several seconds)       
    associations_list = []

    for i in df.lyrics:
        associations_list.append(associator.count_emotions(i))
    
    headers = ['negative', 'positive', 'anger', 'fear', 'anticipation', 'surprise',
               'trust', 'sadness', 'joy', 'disgust', 'wordcount']
    emotions = pd.DataFrame(associations_list, columns = headers)
    df_emotions = pd.concat([df, emotions], axis = 1)
    
    # Create percentages for each emotion (based on word count)
    df_emotions['negative_percent']     = df_emotions.negative / df_emotions.wordcount
    df_emotions['positive_percent']     = df_emotions.positive / df_emotions.wordcount
    df_emotions['anger_percent']        = df_emotions.anger / df_emotions.wordcount
    df_emotions['fear_percent']         = df_emotions.fear / df_emotions.wordcount
    df_emotions['anticipation_percent'] = df_emotions.anticipation / df_emotions.wordcount
    df_emotions['surprise_percent']     = df_emotions.surprise / df_emotions.wordcount
    df_emotions['trust_percent']        = df_emotions.trust / df_emotions.wordcount
    df_emotions['sadness_percent']      = df_emotions.sadness / df_emotions.wordcount
    df_emotions['joy_percent']          = df_emotions.joy / df_emotions.wordcount
    df_emotions['disgust_percent']      = df_emotions.disgust / df_emotions.wordcount
    
    # Convert NaN's (resulting from divide by zero) to zero
    df_emotions = df_emotions.fillna(0)

    ##### Create weighted word sums of each emotion from news articles
    # Get articles data
    df_articles = pd.read_csv('getarticles_output.csv')
    articles_associations_list = []
    for a, b in zip(df_articles.header, df_articles.description):
        articles_associations_list.append(associator.count_emotions(a + " " + b))
    
    df_articles_emotions = pd.DataFrame(articles_associations_list, columns = headers)
    df_articles_emotions['socialmediascore'] = df_articles.socialmediascore

    # Convert column to integer (from string)
    df_articles_emotions.socialmediascore = pd.to_numeric(df_articles_emotions.socialmediascore.str.replace(',', ''))
    
    # Weigh the columns by social media score and turn results to a dataframe
    templist = []
    for index in df_articles_emotions.iterrows():
        temp = []
        temp.append(index[1][0]*index[1][11])
        temp.append(index[1][1]*index[1][11])
        temp.append(index[1][2]*index[1][11])
        temp.append(index[1][3]*index[1][11])
        temp.append(index[1][4]*index[1][11])
        temp.append(index[1][5]*index[1][11])
        temp.append(index[1][6]*index[1][11])
        temp.append(index[1][7]*index[1][11])
        temp.append(index[1][8]*index[1][11])
        temp.append(index[1][9]*index[1][11])
        temp.append(index[1][10]*index[1][11])
        templist.append(temp)
        
    df_templist = pd.DataFrame(templist)
    
    # Sum up the weighted columns into a single list
    sums = []
    for column in df_templist:
        sums.append(sum(df_templist[column]))

    # Create list of percentages (divided by weighted wordcount)
    templist2 = ['negative', 'positive', 'anger', 'fear', 'anticipation', 
                 'surprise', 'trust', 'sadness', 'joy', 'disgust']
    templist3 = []

    # Write the summed weighted columns to a file (article_emotions.txt)
    with open("article_emotions.txt", "w", newline = None, encoding='utf-8') as file:
        file.write('The weighted number of emotion words in news articles are:\n\n')
    for i in range(0, len(templist2)):
        # Print number of emotional words (weighted by social media score) in news articles
        with open("article_emotions.txt", "a", newline = None, encoding='utf-8') as file:
            file.write('{:>50} {:>10}'.format('The weighted number of "' + templist2[i] + '" words is:', str(sums[i])) + '\n')
        templist3.append(sums[i]/sums[10]) # Append percentages to templist3 for later

    ##### Create emotion cohesion distance scores
    var_list = ['negative_percent', 'positive_percent', 'anger_percent',
                'fear_percent', 'anticipation_percent', 'surprise_percent',
                'trust_percent', 'sadness_percent', 'joy_percent',
                'disgust_percent']
    df_emotions_subset = df_emotions[var_list]
    distances = []
    
    # Cohesion distance is calculated by subtracting the emotion percentages of each
    # song by the emotion percentages from the news articles. These differences were
    # squared and summed for each song and square-rooted (i.e., a 2-norm).
    # This was applied only to the six basic emotions: anger, fear, surprise,
    # sadness, joy, and disgust. The lower the score, the greater degree of cohesion 
    # between the news articles and the song lyrics.
    for index in df_emotions_subset.iterrows():            
        distances.append( ( (index[1][2] -  templist3[2])**2 + 
                            (index[1][3] -  templist3[3])**2 + 
                            (index[1][4] -  templist3[4])**2 +
                            (index[1][5] -  templist3[5])**2 + 
                            (index[1][6] -  templist3[6])**2 + 
                            (index[1][7] -  templist3[7])**2 + 
                            (index[1][8] -  templist3[8])**2 + 
                            (index[1][9] -  templist3[9])**2 )**(.5) )
    df_emotions['cohesiondistance'] = distances
    df_emotions.to_csv('cleaneddata.csv')   


def dbscan_groupmeans(df, var, minpts, eps):
    
    dbscan = DBSCAN(eps = eps, min_samples = minpts)
    dbscan.fit(df[var])
    labels = dbscan.labels_  # Save labels for use later
    dbscanout = df[var].groupby(labels).mean() # Create grouped means
    dbscancounts = pd.Series(labels, name = "Counts").value_counts() # Create number of points in each cluster
    dbscanout = pd.concat([dbscanout, dbscancounts], axis = 1)
    dbscanout.to_csv('dbscan_groupedmeans.csv', sep=',', index = True, header = True)
    return(labels)  # Return labels to be used later

   
#### Generates DBSCAN K-distance plot (knee plot) in order to determine eps
def dbscan_kdistanceplot(df, var, k):

    neighbors = NearestNeighbors(n_neighbors = k, metric = 'euclidean') 
    neighbors.fit(df[var])
    # Note that first nearest neigbhor always has distance of zero, since it includes the point itself
    distances, indices = neighbors.kneighbors(df[var])
    out = distances[:, k-1] # Pull the kth nearest neighbor distances (Euclidean)
    out.sort() # Sort the distances by size prior to plotting
    plt.plot(out, linestyle = "-")
    plt.suptitle("Nearest Neighbor K-Distances")
    plt.savefig("dbscan_kdistance")
    plt.clf()
    plt.close() 
    
 
#### Create a function that generates descriptive statistics
def descriptives(df):

    ########### Create text output of descriptive statistics ###########
    
    ### First, create means, medians, and SDs of quantitative variables
    
    # The following are not all of the quantitative variables, but the ones 
    # where analyses were plausible
    columnlist = ['fullrank', 'artistfreq','lyricslength', 'duration',
                  'listeners', 'playcount', 'anger', 'fear', 'anticipation',
                  'surprise', 'trust', 'sadness', 'joy', 'disgust',
                  'wordcount', 'anger_percent', 'fear_percent', 
                  'anticipation_percent', 'surprise_percent', 'trust_percent',
                  'sadness_percent', 'joy_percent', 'disgust_percent', 
                  'cohesiondistance']

    with open("descriptives.txt", "w", newline = None, encoding = 'utf-8') as file:
        # Write header rows
        file.write("\n\nDescriptive Statistics for Quantitative Variables: \n")
        file.write('{:>25} {:>13} {:>13} {:>13}\n'.format('VARIABLE', 'MEAN', 'MEDIAN', 'SD'))

    with open("descriptives.txt", "a", newline = None, encoding='utf-8') as file:
        for item in columnlist:
            mean_stat = df[item].mean()
            median_stat = df[item].median()
            std_stat = df[item].std()
            # Write the rows to file formatted nicely
            file.write('{:>25} {:>13} {:>13} {:>13}\n'.format(
                       item, 
                       "{0:.6g}".format(mean_stat), 
                       "{0:.6g}".format(median_stat), 
                       "{0:.6g}".format(std_stat)))
    
    ### Second, create list of top
             
    taglist = [] # Create a list to store elements of 'toptag'
    for i in df['toptags']:
        taglist.extend(ast.literal_eval(i))
        
    taglist = [x.lower() for x in taglist]          # Convert all elements to lower case
    taglist = [x.replace(" ", "") for x in taglist] # Replace all spaces
    taglist = [x.replace("-", "") for x in taglist] # Replace all dashes
               
    count = Counter(taglist) # Create counter object
    mostcommon = count.most_common()
    
    # Print the top 10 most common song tags
    with open("descriptives.txt", "a", newline = None, encoding = 'utf-8') as file:
        # Write more header rows
        file.write("\n\nThe 10 most common tags are: (# of times tagged) \n")
        for i in range(0, 10):
            file.write('{:>20}: {:>7}\n'.format(mostcommon[i][0], mostcommon[i][1]))
    
    artistlist = [] # Create a list to store elements of 'artistname'
    for i in df['artistname']:
        artistlist.append(i)
           
    count = Counter(artistlist) # Create counter object
    mostcommon = count.most_common()
    
    # Print the top 10 most common artists
    with open("descriptives.txt", "a", newline = None, encoding = 'utf-8') as file:
        # Write the rows to file formatted nicely
        file.write("\n\nThe 10 most common artists are: (# of times in top 5000) \n")
        for i in range(0, 10):
            file.write('{:>20}: {:>7}\n'.format(mostcommon[i][0], mostcommon[i][1]))

   
#### Create a function that uses 'getevents' function to pull raw html from 
#### eventregistry.org based on speciifc dates. The directory is location of 
#### of the current machine's Firefox executable, which is necessary for this
#### function to work. The output is a csv file with the top 25 most popular
#### news events (as based on eventregistry's social media score).
def getarticles(datestart, dateend, directory):
    
    ##### Webscrape eventregistry.org #####
    
    # Gather parameters for get_events function
    url = "http://eventregistry.org/searchEvents?query=%7B%22" + \
          "locations%22:%5B%7B%22label%22:%22United%20States" + \
          "%22,%22uri%22:%22http:%2F%2Fen.wikipedia.org%2Fwiki%2F" + \
          "United_States%22,%22negate%22:false%7D%5D,%22dateStart%22:%22" + \
          datestart + "%22,%22dateEnd%22:%22" + \
          dateend + "%22,%22lang%22:%22eng%22,%22minArticles%22:200,%22" + \
          "preferredLang%22:%22eng%22%7D&tab=events"
    directory = r'C:\Program Files (x86)\Mozilla Firefox\firefox.exe'
    
    # Find location of Firefox executable
    binary = FirefoxBinary(directory)
    driver = webdriver.Firefox(firefox_binary = binary)
    
    # Open the url in Firefox
    driver.get(url)
    time.sleep(30)  # Wait 30 seconds for page to load
    
    # Click "sort events by social media hotness" to get most popular events
    driver.find_element_by_xpath('//*[@id="tab-events"]/div/div/div[3]/div[2]/div/div[2]/button[4]').click()
    time.sleep(10)  # Wait 10 seconds for page to reload
    
    # Save the entire page source (it's huge!)
    out = driver.page_source.encode("utf-8")
    
    # Close the webdriver
    driver.quit()

    ##### Take out relevant parts of raw html ##### 
    
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
    date = timedate[::2]
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

    ##### Write the lists into a CSV #####
    
    # Create header for csv output file
    with open("getarticles_output.csv", "w", newline = '', encoding='utf-8') as f:
        fieldnames = ['id', 'header', 'date', 'location', 'categories', 'description', 'socialmediascore']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
    
    for a,b,c,d,e,f,g in zip(ids, articleheader, date, location, categories, description, socialmedia):    
        # Write to output file
        with open("getarticles_output.csv", "a", newline = '', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames = fieldnames)
            writer.writerow({'id': a, 
                             'header': b,
                             'date': c,
                             'location': d,
                             'categories': e,
                             'description': f,
                             'socialmediascore': g})

            
#### Creates correlation matrix (written to CSV) given the dataframe and variable list
def getcorrelations(df, var):
    
    # Write to CSV after rounding to 4 significant digits
    df[var].corr().round(4).to_csv('correlations.csv', sep=',', index = True, header = True)        
         
    
#### Creates histograms given the dataframe and variable list
def gethistograms(df, var):            
   
    for v in var:  
        name = "histogram_" + v
        df[v].hist()
        plt.suptitle("Histogram for " + v)
        plt.savefig(name)
        plt.clf()
        plt.close()    


### Gets additional song information from Last.fm
def getinfo(api_key):
    
    # Get track and artist info from csv that was collected earlier
    with open("gettracks_output.csv", "r", newline = '', encoding = "utf-8") as file:
        input_file = csv.DictReader(file)
        tracks = []
        for row in input_file:
            tracks.append(row)  

    # Create header for csv output file
    with open("getinfo_output.csv", "w", newline = '', encoding='utf-8') as f:
        fieldnames = ['url', 'name', 'artistname', 'duration', 'listeners',
                      'playcount', 'toptags']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
    
    for i in tracks:
        url = {'method': 'track.getInfo',
               'track': i['name'],
               'artist': i['artistname'],
               'api_key': api_key,
               'autocorrect': 1,
               'format': 'json'}
        response = requests.get("http://ws.audioscrobbler.com/2.0/", url)
        txt = response.json()

        url = txt['track']['url']
        name = txt['track']['name']
        artistname = txt['track']['artist']['name']
        duration = txt['track']['duration']
        listeners = txt['track']['listeners']
        playcount = txt['track']['playcount']

        toptags = []
        for tag in txt['track']['toptags']['tag']:
            toptags.append(tag['name'])
        
        with open("getinfo_output.csv", "a", newline = '', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames = fieldnames)
            writer.writerow({'url': url,
                             'name': name,
                             'artistname': artistname,
                             'duration': duration,
                             'listeners': listeners,
                             'playcount': playcount,
                             'toptags': toptags})

            
#### Creates a scatterplot matrix using desired variables   
def getscatterplots(df, var):
    
    scatter_matrix(df[var], alpha = 0.2, figsize = (12, 12), diagonal = 'hist')
    plt.suptitle("Scatterplot Matrix")
    plt.savefig("scatterplotmatrix")
    plt.clf()
    plt.close()

            
#### Generates K-means centroids (as a CSV file) and also returns the labels            
def kmeans_centroids(df, var, k):
    
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(df[var])
    labels = kmeans.labels_  # Save labels for use later
    centroids = kmeans.cluster_centers_
    kmeansout = pd.DataFrame(centroids, columns = var) # Create dataframe of centroids
    kmeanscounts = pd.Series(labels, name = "Counts").value_counts() # Create number of points in each cluster
    kmeansout = pd.concat([kmeansout, kmeanscounts], axis = 1)
    kmeansout.to_csv('kmeans_centroids.csv', sep=',', index = True, header = True)
    return(labels)  # Return labels to be used later
    
 
#### Generates silhouette scores for K-means using a list of the number of clusters    
def kmeans_silhouette(df, var, k):
    
    with open("kmeans_silhouettescores.txt", "w", newline = None, encoding = 'utf-8') as file:
        file.write("\n\nThe following are silhouette scores for K-means with varying number of K clusters: \n\n")
    
    with open("kmeans_silhouettescores.txt", "a", newline = None, encoding = 'utf-8') as file:
        for c in k: 
            kmeans = KMeans(n_clusters = c)
            kmeans.fit(df[var])
            labels = kmeans.labels_
            file.write("For K=" + str(c) + ", the silhouette score is: " + str(silhouette_score(df[var], labels)) + "\n")
   

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
    
    
#### Generates Ward group means (as a CSV file) and also returns the labels                    
def ward_groupmeans(df, var, k):
    
    ward = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
    ward.fit(df[var])
    labels = ward.labels_  # Save labels for use later
    wardout = df[var].groupby(labels).mean()      # Create grouped means
    wardcounts = pd.Series(labels, name = "Counts").value_counts() # Create number of points in each cluster
    wardout = pd.concat([wardout, wardcounts], axis = 1)
    wardout.to_csv('ward_groupedmeans.csv', sep=',', index = True, header = True) # Save to file
    return(labels)  # Return labels to be used later

    
#### Generates silhouette scores for Ward using a list of the number of clusters
def ward_silhouette(df, var, k):
    
    with open("ward_silhouettescores.txt", "w", newline = None, encoding = 'utf-8') as file:
        file.write("\n\nThe following are silhouette scores for Ward's method with varying number of K clusters: \n\n")
    
    with open("ward_silhouettescores.txt", "a", newline = None, encoding = 'utf-8') as file:
        for c in k: 
            ward = AgglomerativeClustering(n_clusters = c, linkage = 'ward')
            ward.fit(df[var])
            labels = ward.labels_
            file.write("For K=" + str(c) + ", the silhouette score is: " + str(silhouette_score(df[var], labels)) + "\n")


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
        temp0 = break_contractions(text)                     # Break up contractions (and convert to lower case)
        temp1 = lemmatize_words(temp0.split())               # Split string to words, then lemmatize
        temp2 = mark_negation(temp1, double_neg_flip = True) # Account for negations
        temp3 = remove_stopwords(temp2)                      # Remove any stopwords
        
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


########## Functions for Predictive Analysis ##########

def T_test(var1, var2):
    return(str(ttest_ind(var1, var2, axis = 0, equal_var = False)))

    
def Linear_Regression(outcome, predictors):
    # Linear Regression.
    Y = outcome
    X = predictors
    data = pd.DataFrame([Y] + X)
    model = ols("Y ~ X", data).fit()
    return(str(model.summary()))

    
def SVM(Y, X):
    test_size = 0.30
    seed = 7 
    # Cross validation
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state = seed)
    # Fit the model
    model = SVC() 
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validate)
    Y_validate_table = pd.crosstab(index = Y_validate, columns = "count")['count']
    predictions_table_temp = pd.crosstab(index = predictions, columns = "count")['count']
    # Create list of zeroes first and fill in with counts (so zero cells are not dropped)
    predictions_table = pd.Series([0]*len(Y_validate_table), index = Y_validate_table.index.values)
    for i in predictions_table_temp.index.values:
        predictions_table.loc[i] = predictions_table_temp.loc[i]
    return([str(accuracy_score(Y_validate, predictions, normalize=True)), 
            str(confusion_matrix(Y_validate, predictions)),
            str(chisquare(predictions_table, Y_validate_table))])

    
def Naive_Bayes(Y, X):
    test_size = 0.30
    seed = 7 
    # Cross validation
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state = seed)
    # Fit the model
    model = GaussianNB() 
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validate)
    Y_validate_table = pd.crosstab(index = Y_validate, columns = "count")['count']
    predictions_table_temp = pd.crosstab(index = predictions, columns = "count")['count']
    # Create list of zeroes first and fill in with counts (so zero cells are not dropped)
    predictions_table = pd.Series([0]*len(Y_validate_table), index = Y_validate_table.index.values)
    for i in predictions_table_temp.index.values:
        predictions_table.loc[i] = predictions_table_temp.loc[i]
    return([str(accuracy_score(Y_validate, predictions, normalize=True)), 
            str(confusion_matrix(Y_validate, predictions)),
            str(chisquare(Y_validate_table, predictions_table))])

    
def Random_Forest(Y, X):
    test_size = 0.30
    seed = 7 
    # Cross validation
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state = seed)
    # Fit the model
    model = RandomForestClassifier() 
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validate)
    Y_validate_table = pd.crosstab(index = Y_validate, columns = "count")['count']
    predictions_table_temp = pd.crosstab(index = predictions, columns = "count")['count']
    # Create list of zeroes first and fill in with counts (so zero cells are not dropped)
    predictions_table = pd.Series([0]*len(Y_validate_table), index = Y_validate_table.index.values)
    for i in predictions_table_temp.index.values:
        predictions_table.loc[i] = predictions_table_temp.loc[i]
    return([str(accuracy_score(Y_validate, predictions, normalize=True)), 
            str(confusion_matrix(Y_validate, predictions)),
            str(chisquare(predictions_table, Y_validate_table))])


def Decision_Tree(Y, X):
    test_size = 0.30
    seed = 7 
    # Cross validation
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state = seed)
    # Fit the model
    model = DecisionTreeClassifier() 
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validate)
    Y_validate_table = pd.crosstab(index = Y_validate, columns = "count")['count']
    predictions_table_temp = pd.crosstab(index = predictions, columns = "count")['count']
    # Create list of zeroes first and fill in with counts (so zero cells are not dropped)
    predictions_table = pd.Series([0]*len(Y_validate_table), index = Y_validate_table.index.values)
    for i in predictions_table_temp.index.values:
        predictions_table.loc[i] = predictions_table_temp.loc[i]
    return([str(accuracy_score(Y_validate, predictions, normalize=True)), 
            str(confusion_matrix(Y_validate, predictions)),
            str(chisquare(predictions_table, Y_validate_table))])


def KNN(Y, X):
    test_size = 0.30
    seed = 7 
    # Cross validation
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state = seed)
    # Fit the model
    model = KNeighborsClassifier() 
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validate)
    Y_validate_table = pd.crosstab(index = Y_validate, columns = "count")['count']
    predictions_table_temp = pd.crosstab(index = predictions, columns = "count")['count']
    # Create list of zeroes first and fill in with counts (so zero cells are not dropped)
    predictions_table = pd.Series([0]*len(Y_validate_table), index = Y_validate_table.index.values)
    for i in predictions_table_temp.index.values:
        predictions_table.loc[i] = predictions_table_temp.loc[i]
    return([str(accuracy_score(Y_validate, predictions, normalize=True)), 
            str(confusion_matrix(Y_validate, predictions)),
            str(chisquare(predictions_table, Y_validate_table))])


    
########## MAIN ##########
    
def main():     

    #### First, grab some additional data from Last.fm.
    
    # While this function will work fine, I suggest only running it for test purposes
    # only since the information it grabs is time sensitive (i.e., it always grabs the
    # information from the past 7 days, and not historical data)
    
    # --------------------------------------------------------------------------
    # getinfo(api_key = '21b84b7984893188a7dd09863171b052')                          # <--------------Uncomment for testing purposes
    # --------------------------------------------------------------------------

    
    #### Second, grab some additional news data from eventregistry.org.
    
    # This function will open a Firefox browser for approx. 60 seconds. Please do not
    # close the brower (it will close on its own after completion). Please only use
    # if the Selenium module and Firefox are installed (plus the geckodriver).
        
    # --------------------------------------------------------------------------
    # getarticles(datestart = '2016-09-19', 
    #             dateend = '2016-09-26',                                            
    #             directory = r'C:\Program Files (x86)\Mozilla Firefox\firefox.exe') # <--------------Uncomment for testing purposes
    # --------------------------------------------------------------------------
    
                  
    #### Third, clean the data and add several new variables (output is 'cleaneddata.csv' and 'articleemotions.txt')
    
    cleandata()
    
    
    #### Import 'cleaneddata.csv'
    
    df = pd.read_csv('cleaneddata.csv', encoding = 'latin-1')
    
    
    #### Creates a descriptives output file ('descriptives.txt')
    
    descriptives(df = df)
    
    
    #### Add a binning variable to dataframe df
    
    # Currently, the data is discretized to three levels (high, medium, and low cohesion)
    binning(df = df, variable = 'cohesiondistance', levels = [0, .5, 1, 10])
    
    
    #### Plot a histogram for "playcount", "negative", and "positive"
    
    # The resulting histograms are saved in the working directory as PNG files
    VariableList1 = ["anger_percent", "sadness_percent", "joy_percent"]
    gethistograms(df = df, var = VariableList1)
     
    
    #### Create correlation table
    
    # The resulting correlation table is written to 'correlations.csv'
    VariableList2 = ["anger_percent", "sadness_percent", "joy_percent"]
    getcorrelations(df = df, var = VariableList2)

    
    #### Plot the three variables in a scatterplot
    
    # The resulting scatterplot matrix are saved to 'scatterplotmatrix.png'
    getscatterplots(df = df, var = VariableList2)

    
    #### Cluster analysis: K-means
    
    # Choose the variables for clustering, and the number of clusters to check (for K-means and Ward)
    VariableList3 = ["anger_percent", "sadness_percent", "joy_percent"]
    clusters = [2, 3, 4, 5, 6, 7, 8]
    
    # First, generate models for K=2 through 8, and determine mean silhouette score. The closer
    # the score is to 1, the better the model. This will be written directly into the output
    # file "kmeans_silhouettescores.txt".
    kmeans_silhouette(df = df, var = VariableList3, k = clusters)
    
    # Since the silhouette scores indicate the 2 cluster model is best, we save the centroids
    # in a CSV file 'kmeans_centroids.csv'. Also save the labels into 'labels1'.
    labels1 = kmeans_centroids(df = df, var = VariableList3, k = 3)

    
    #### Cluster analysis: Ward
    
    # First, generate models for K=2 through 8, and determine mean silhouette score. The closer
    # the score is to 1, the better the model. This will be written directly into the output
    # file "ward_silhouettescores.txt".
    ward_silhouette(df = df, var = VariableList3, k = clusters)
        
    
    # Since the silhouette scores indicate the 3 cluster model is best, we save the grouped
    # means in a CSV file. Also save the labels into 'labels2'.
    labels2 = ward_groupmeans(df = df, var = VariableList3, k = 3)
  
    
    #### Cluster analysis: DBSCAN
    # First, use rule of thumb for minpts (i.e, 1 + number of dimensions) and determine eps through KNN by 
    # plotting distances to Kth nearest neighbor. K is analogous to minpts and the distance of inflection for the 
    # plot (i.e., the knee) will be the the eps. PLot is saved as 'dbscan_kdistance.png'.
    dbscan_kdistanceplot(df = df, var = VariableList3, k = 4)
    
    
    # The plot above suggests an eps=0.08 with minpts=4. We will run the DBSCAN using these. We will
    # Save the grouped means in a CSV file 'dbscan_groupedmeans.csv'.
    labels3 = dbscan_groupmeans(df = df, var = VariableList3, minpts = 4, eps = 0.08)

    
    #### Plot all clusters
    
    # Plot K-means results on 3D scatterplot
    scatterplotclusters(df = df, var = VariableList3, labels = labels1, title = 'K-Means Scatterplot by Cluster', 
                        savename = "kmeans_3Dscatterplot")
       
    # Plot Ward results on 3D scatterplot
    scatterplotclusters(df = df, var = VariableList3, labels = labels2, title = 'Ward Scatterplot by Cluster', 
                        savename = "ward_3Dscatterplot")    
    
    # Plot DBSCAB results on 3D scatterplot
    scatterplotclusters(df = df, var = VariableList3, labels = labels3, title = 'DBSCAN Scatterplot by Cluster', 
                        savename = "dbscan_3Dscatterplot")    
    
        
    #### Association Rules
    
    # Create a list of lists for the 'toptags' attribute. This also eliminates whitespaces and
    # some special characters to faciliate counting frequencies of each tag
    taglist = [] # Create a list to store elements of 'toptag'
    for i in df['toptags']:
        temp = ast.literal_eval(i)
        temp = [x.lower() for x in temp]          # Convert all elements to lower case
        temp = [x.replace(" ", "") for x in temp] # Replace all spaces
        temp = [x.replace("-", "") for x in temp] # Replace all dashes
        taglist.append(temp)
    
    # Create assocationrules object. It creates an occurrence matrix for all tags that occur
    # frequently enough, as based on minsup.
    
    # First, with minimum support 500, minimum confidence = .50, and maximum 4 items in precedent
    associationtester1 = associationrules(taglist, minsup = 400)
    associationtester1.associationcombtest(maxitems = 4, minconf = 0.20, filename = "associationrules1.txt")
    
    # Second, with minimum support 500, minimum confidence = .60, and maximum 4 items in precedent
    associationtester2 = associationrules(taglist, minsup = 500)
    associationtester2.associationcombtest(maxitems = 4, minconf = 0.20, filename = "associationrules2.txt")

    # Third, with minimum support 500, minimum confidence = .70, and maximum 4 items in precedent
    associationtester3 = associationrules(taglist, minsup = 600)
    associationtester3.associationcombtest(maxitems = 4, minconf = 0.20, filename = "associationrules3.txt")

    df.cohesiondistance_binned.value_counts()
    
    
    #### Predictive Analyses
    
    with open("Hypothesis_Testing.txt", "w") as file:

        # t-test: Cluster 0 vs 2 from K-means
        file.write("\n1) Hypothesis: Positive and Negative clusters from K-means have identical mean values for playcount.\n\nTest statistics for t-test:\n")
        file.write("\nThe means (standard deviations) are:\n")
        file.write('Cluster 0: {0:.12g} ({1:.12g})\nCluster 2: {2:.12g} ({3:.12g})'.format(
                     df['playcount'].loc[labels1 == 0].mean(),
                     df['playcount'].loc[labels1 == 0].std(), 
                     df['playcount'].loc[labels1 == 2].mean(), 
                     df['playcount'].loc[labels1 == 2].std()))
        file.write("\n\n")
        file.write(T_test(df['playcount'].loc[labels1 == 0], df['playcount'].loc[labels1 == 2]))
        
        # Linear Regression: Playcount is predicted by cohesion distance and song duration
        file.write("\n\n\n\n2) Hypothesis: Playcount is predicted by cohesion distance and song duration.\n\nTest statistics for Regression:\n")
        file.write(Linear_Regression(df['playcount'], df[['cohesiondistance', 'duration']]))
        
        file.write("\n\n\n\n3) Hypothesis: cohesiondistance_binned can be predicted by duration, listeners, playcount, fullrank, and wordcount\n\n")

        # Support Vector Machines: Classify cohesiondistance_binned
        model1 = SVM(df['cohesiondistance_binned'], df[['duration', 'listeners', 'playcount', 'fullrank']])
        file.write("a) Accuracy Percent for SVM: " + model1[0] + "\n\nConfusion matrix is:\n" + model1[1] + "\n\nChi-square goodness-of-fit is:\n" + model1[2] + "\n\n") 
        
        # Naive Bayes: Classify cohesiondistance_binned
        model2 = Naive_Bayes(df['cohesiondistance_binned'], df[['duration', 'listeners', 'playcount', 'fullrank']])
        file.write("b) Accuracy Percent for Naive Bayes: " + model2[0] + "\n\nConfusion matrix is:\n" + model2[1] + "\n\nChi-square goodness-of-fit is:\n" + model2[2] + "\n\n") 

        # Random Forest: Classify cohesiondistance_binned
        model3 = Random_Forest(df['cohesiondistance_binned'], df[['duration', 'listeners', 'playcount', 'fullrank']])
        file.write("c) Accuracy Percent for Random Forest: " + model3[0] + "\n\nConfusion matrix is:\n" + model3[1] + "\n\nChi-square goodness-of-fit is:\n" + model3[2] + "\n\n") 

        # Decision Tree: Classify cohesiondistance_binned
        model4 = Decision_Tree(df['cohesiondistance_binned'], df[['duration', 'listeners', 'playcount', 'fullrank']])
        file.write("c) Accuracy Percent for Decision Tree: " + model4[0] + "\n\nConfusion matrix is:\n" + model4[1] + "\n\nChi-square goodness-of-fit is:\n" + model4[2] + "\n\n") 

        # KNN: Classify cohesiondistance_binned
        model5 = KNN(df['cohesiondistance_binned'], df[['duration', 'listeners', 'playcount', 'fullrank']])
        file.write("e) Accuracy Percent for k-NN: " + model5[0] + "\n\nConfusion matrix is:\n" + model5[1] + "\n\nChi-square goodness-of-fit is:\n" + model5[2] + "\n\n") 
        
      
if __name__ == "__main__":
    
    main()

