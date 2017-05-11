COSC 587 Project: Part 3
December 7, 2016

This README will explain the contents of the project ZIP file and explain the functions used in the main 
Python script. For the most part, the contents of the ZIP file (excluding the legacy folders from Parts 1 and 2)
contain only datasets, plots, and code for the additional analyses produced for Part 3 of the project. Since
the final story included aspects from Parts 1 and 2, the deliverables for those parts have been included in this
ZIP folder (for convenience).

The final story portion of the project has been posted online at: https://sites.google.com/site/anly501project3/


##################################################################################################################
IMPORTANT!!:
Prior to running the Python function getdayarticles(), there are several prerequites that must be fulfilled.

1) Install the Selenium module:
- This module is not pre-installed in Anaconda, but it can be installed fairly easily through pip. The module
  is used to pull source code from websites through the Firefox browser.

2) Install Firefox:
- It can be downloaded from their website: https://www.mozilla.org/en-US/firefox/new/. The directory to the 
  executable (e.g., C:\Program Files (x86)\Mozilla Firefox\firefox.exe) should also be known, since it is
  required for the use of Selenium.

3) Place the 'geckodriver' file in your Python working directory
- The 'geckodriver' file is necessary to use Firefox in the Selenium module.
- We have provided this file for Windows (64-bit) in the ZIP folder, but if you are using a different operating 
  system, please download the appropriate file from https://github.com/mozilla/geckodriver/releases.

Prior to running the Python function spotify_charts_emotions(), articles_emotions(), or any function that uses 
these two functions the following prerequite must be fulfilled.

4) Download relevant files for the NLTK module:
- The NLTK module is pre-installed in Anaconda, but several supplemental files need to be downloaded. The 
  three files (corpora) are: stopwords, wordnet, and words.
- To download these files, run the script "import nltk; nltk.download()" which will prompt a second window 
  to open and allow selection and download of the appropriate files.


*Please note that the functions "spotify_charts()", "spotify_charts_emotions()", "getdayarticles()" (which 
utilizes the Selenium package), "articles_emotions()", and "articles_emotions_perday()" are CURRENTLY 
COMMENTED-OUT (i.e., these are not necessary for data analyses and plots). These functions are near the top of 
the main function. However, I do recommend testing these functions out (although they can take a while to run).

Please contact me at jmc511@georgetown.edu if anything goes wrong.

##################################################################################################################



I.  DESCRIPTION OF FILES IN ZIP FOLDER:

FILENAME                                                DESCRIPTION
------------------------------------------------------------------------------------------------------------------

contractions.txt........................................This is a text file that contains common contractions
                                                        (e.g., 'couldn't' is contraction of 'could not'). This was
                                                        created so that the data cleaning functions will have
                                                        access to this dictionary and break up contractions when
                                                        necessary during the determination of word emotion 
                                                        associations.

emotion_analysis.csv....................................This is the finalized longitudinal data that was used for
                                                        plotting and analysis.

emotion_articles.png....................................This is a matplotlib plot of news emotion percentages over
                                                        time.

emotion_streamed.png....................................This is a matplotlib plot of song playcounts aggregated by
                                                        cluster group over time.
                                                        
geckodriver.exe.........................................This is the geckodriver to be used for Windows 64 for the
                                                        use of the Selenium module. See important notes above.

getdayarticles.csv......................................This is a file produced by the getdayarticles() function.
                                                        It uses the selenium package to simulate website visits to
                                                        EventRegistry.org. The file contains news article data from
                                                        each day specified in the getdayarticles() function.

getdayarticlesemotions.csv..............................This is a CSV produced by the articles_emotions() function.
                                                        It is essentially a cleaned version of 'getdayarticles.csv'
                                                        that adds the sentiment analysis variables (percent of each
                                                        emotion).

getdayarticlesemotionsperday.csv........................This is a CSV produced by the getdayarticlesemotionsperday() 
                                                        function. It is essentially the percent emotions in 
                                                        'getdayarticlesemotions.csv' which are aggregated over each
                                                        day after being weighted by the social media score.

grangertests.txt........................................This file contains the results of the Granger Causality tests.

kmeans_3Dscatterplot_songs.png..........................This is a 3D scatterplot generated for three variables 
                                                        ("anger_percent", "sadness_percent", "joy_percent") which
                                                        were examined for clustering using K-means. The three 
                                                        numbers on the legend are cluster labels.

kmeans_centroids_songs.csv..............................This file contains the centroid positions after running
                                                        K-means algorithm with 3 clusters.

kmeans_silhouettescores_songs.txt.......................This file contains the silhouette scores for the K-means
                                                        algorithm for cluster counts varying from 2 to 8. A
                                                        score close to 1 indicates better fit (so K=3 was selected
                                                        in this particular case).

nn_negative_anger.png...................................This is a matplotlib plot of the results of the neural network
                                                        model predicting playcount of the negative cluster songs from
                                                        the lagged values of news article anger percentage.

nn_negative_joy.png.....................................This is a matplotlib plot of the results of the neural network
                                                        model predicting playcount of the negative cluster songs from
                                                        the lagged values of news article joy percentage.

nn_negative_sadness.png.................................This is a matplotlib plot of the results of the neural network
                                                        model predicting playcount of the negative cluster songs from
                                                        the lagged values of news article sadness percentage.

nn_null_anger.png.......................................This is a matplotlib plot of the results of the neural network
                                                        model predicting playcount of the null cluster songs from
                                                        the lagged values of news article anger percentage.

nn_null_joy.png.........................................This is a matplotlib plot of the results of the neural network
                                                        model predicting playcount of the null cluster songs from
                                                        the lagged values of news article joy percentage.

nn_null_sadness.png.....................................This is a matplotlib plot of the results of the neural network
                                                        model predicting playcount of the null cluster songs from
                                                        the lagged values of news article sadness percentage.

nn_positive_anger.png...................................This is a matplotlib plot of the results of the neural network
                                                        model predicting playcount of the positive cluster songs from
                                                        the lagged values of news article anger percentage.

nn_positive_joy.png.....................................This is a matplotlib plot of the results of the neural network
                                                        model predicting playcount of the positive cluster songs from
                                                        the lagged values of news article joy percentage.

nn_positive_sadness.png.................................This is a matplotlib plot of the results of the neural network
                                                        model predicting playcount of the positive cluster songs from
                                                        the lagged values of news article sadness percentage.

NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt....This text file contains the NRC word emotion association
                                                        database, which associates a given word with a particular
                                                        emotion. This is used in the class 'word_associations'
                                                        in order to give a count of the number of occurrences 
                                                        of each emotion for a given song or news article. Note
                                                        that documents are first cleaned by breaking contractions,
                                                        words are lemmatized, negations are marked, and stop words
                                                        removed.

plots.docx..............................................This document contains all of the non-interactive plots that were
                                                        were generated from the current part of the project (part 3).

plots1.html.............................................This is the interactive plot of predictions for average positive 
                                                        playcounts as predicted by anger, sadness, and joy news 
                                                        percentages (3 plots in one)

plots2.html.............................................This is the interactive plot of predictions for average negative
                                                        playcounts as predicted by anger, sadness, and joy news 
                                                        percentages (3 plots in one)

plots3.html.............................................This is the interactive plot of predictions for average null 
                                                        playcounts as predicted by anger, sadness, and joy news 
                                                        percentages (3 plots in one)

Project3Presentation.pptx...............................This is the PowerPoint presentation used in class (12/7).

projectpart3.py.........................................This file contains all relevant Python code to perform the
                                                        additional data collection, data cleaning, plots, and 
                                                        predictive analyses. The description
                                                        of each of the functions are shown below.

spotifycharts.csv.......................................This is a data file that results from the spotify_charts()
                                                        function. It stacked data for each's day's worth of songs
                                                        in the Spotify top 200 streamed list.

spotifychartsemotions.csv...............................This is a data file that results from the 
                                                        spotify_charts_emotions() function. It is essentially the 
                                                        same dataset as "spotifycharts.csv" with the addition of 
                                                        sentiment variables (emotion percentages).

ward_3Dscatterplot_songs.png............................This is a 3D scatterplot generated for three variables 
                                                        ("anger_percent", "sadness_percent", "joy_percent") which
                                                        were examined for clustering using Ward. The three 
                                                        numbers on the legend are cluster labels.

ward_groupedmeans_songs.csv.............................This is table showing grouped means for three variables
                                                        ("anger_percent", "sadness_percent", "joy_percent") and 
                                                        each cluster label examined using Ward.

ward_silhouettescores_songs.txt.........................This file contains the silhouette scores for the Ward
                                                        algorithm for cluster counts varying from 2 to 8. A
                                                        score close to 1 indicates better fit (so K=3 was selected
                                                        in this particular case).

[Folder] Project Part 1.................................This folder contains all exact deliverables from Part 1 of 
                                                        project (for comparions, if necessary). This folder contains
                                                        the datasets and code presented in the final story.

[Folder] Project Part 2.................................This folder contains all exact deliverables from Part 2 of 
                                                        project (for comparions, if necessary). This folder contains
                                                        the datasets and code presented in the final story.

------------------------------------------------------------------------------------------------------------------





II. DESCRIPTION OF FUNCTIONS/CLASSES/METHODS:

FUNCTION NAME                            DESCRIPTION
------------------------------------------------------------------------------------------------------------------

break_contractions.......................Function : - text : A string -- usually in the form of an entire document
                                                        or sentence.

                                                    Description: This function uses the file 'contractions.txt' 
                                                    which contains a dictionary of contraction words (e.g., can't,
                                                    wouldn't) and their deconstructed words. Function searches
                                                    the text and replaces instances of contractions with the
                                                    comprised words.

lemmatize_words..........................Function : - text : A list of strings

                                                    Description:
                                                    This function reduces words to their root word (e.g., 'words'
                                                    becomes 'word') which can then be used for emotion 
                                                    association. The function uses the NLTK Wordnet lemmatizer.
                                                    The function returns a list of lemmatized words.

remove_stopwords.........................Function : - text : A list of strings

                                                    Description:
                                                    This function uses the NLTK stopword corpus to remove elements
                                                    that are stopwords (e.g., and, if, then) which are basically
                                                    meaningless for the purposes of emotion association. Output is
                                                    list of words that are not stopwords.

word_assocations.........................Class    : - None

                                                    Description:
                                                    Constructor produces 'word_associations' objects. It was 
                                                    created as a convenient way of keeping the dictionary of 
                                                    NRC word emotion associations in memory for faster access.
                                                    Then, the count_emotions method can be applied to any list
                                                    of words and compared using the NRC dictionary.

word_assocations.count_emotions..........Method   : - text : A string variable (typically entire documents or 
                                                        sentences.

                                                    Description:
                                                    Takes in a list of words, breaks contractions back to their
                                                    component words, lemmatizes the words, accounts for negations,
                                                    removes stop words, then compares the words to the NRC
                                                    dictionary. The method keeps count of the number of times 
                                                    each emotion is conveyed in the words (with some words having
                                                    more than one emotion) and returns the number of occurrences
                                                    for each emotion as a list.

remove_parenth...........................Function : - text : A string variable

                                                    Description:
                                                    This function simply removes parenthesis from a string (for the 
                                                    purposes of entering a valid search query on LyricWikia).

replace_special..........................Function : - text : A string variable

                                                    Description:
                                                    This function removes special characters (for the 
                                                    purposes of entering a valid search query on LyricWikia).

replace_accents..........................Function : - text : A string variable

                                                    Description:
                                                    This function replaces accented characters with their English
                                                    equivalent (for the purposes of entering a valid search query 
                                                    on LyricWikia).

remove_comments..........................Function : - text : A string variable

                                                    Description:
                                                    This function removes text in HTML comment brackets. This was
                                                    used to clean raw website source code to pull lyrics from 
                                                    LyricWikia.

decode_decimal...........................Function : - letters : A list of letter characters in byte format

                                                    Description:
                                                    This function converts text encoded in decimal format based 
                                                    on their integer code. This was needed because the lyrics were
                                                    coded in decimal format on LyricWikia (presumably to make web
                                                    scraping more difficult).

getlyrics................................Function : - track : A string for the song name

                                                    - artist : A string for the artist name

                                                    Description:
                                                    This function scrapes LyricWikia for the lyrics for the song
                                                    name and artist specified.

days_between.............................Function : - start : A date object (starting day)

                                                    - end : A date object (ending day)

                                                    Description:
                                                    This is a helper function that returns a list of dates (as 
                                                    strings) between the specified dates (including the start
                                                    date but not including the end date).

spotify_charts...........................Function : - start : A date object (starting day)

                                                    - end : A date object (ending day)

                                                    Description:
                                                    This function pulls the CSV files from SpotifyCharts.com
                                                    and stacks the song data (top 200 lists per day) for each
                                                    day within the range specified. It produces the file 
                                                    "spotifycharts.csv".

spotify_charts_emotions..................Function : - None

                                                    Description:
                                                    This function imports "spotifycharts.csv", finds only the
                                                    unique songs in the list (since there are duplicates over 
                                                    many days of top 200 songs), collects lyric data from
                                                    LyricWikia, and produces the "spotifychartsemotions.csv"
                                                    file that contains emotion percentages for each song. This
                                                    program automatically excludes songs where lyrics were not
                                                    found.

articles_emotions........................Function : - None

                                                    Description:
                                                    This function imports "getdayarticles.csv" and adds emotion
                                                    percentages as new columns for each article. It produces the 
                                                    file "getdayarticlesemotions.csv".
                                                   
articles_emotions_perday.................Function : - None

                                                    Description:
                                                    This function imports "getdayarticlesemotions.csv" and 
                                                    aggregates the emotion percentages for the articles within
                                                    each date. Emotion percentages are weighted by social media
                                                    score. The result is "getdayarticlesemotionsperday.csv".

kmeans_centroids.........................Function : - df : A pandas dataframe

                                                    - var : A list of variables in the dataframe

                                                    - k : The number of clusters for K-means

                                                    Description: 
                                                    This function fits the variables into a K-means algorithm 
                                                    using k clusters. The output is 'kmeans_centroids.csv' and 
                                                    also return the cluster labels.

kmeans_silhouette........................Function : - df : A pandas dataframe

                                                    - var : A list of variables in the dataframe

                                                    - k : The list of the number of clusters for K-means

                                                    Description: 
                                                    This function fits several K-means algorithms on specified 
                                                    variables as based on the elements of k. A silhouette 
                                                    score is generated each time (with values closer to 
                                                    1 being the best). The output is a text file containing the
                                                    silhouette scores ('kmeans_silhouettescores.txt').

ward_groupmeans..........................Function : - df : A pandas dataframe

                                                    - var : A list of variables in the dataframe

                                                    - k : The list of the number of clusters for Ward

                                                    Description:
                                                    This function runs Ward clustering based with the number of 
                                                    clusters being the stopping criterion. It outputs a CSV file
                                                    containing the means for each variable broken down by the 
                                                    clusters that resulted from Ward. Output is 
                                                    'ward_groupedmeans.csv' and also returns the cluster labels.

ward_silhouette..........................Function : - df : A pandas dataframe

                                                    - var : A list of variables in the dataframe

                                                    - k : The list of the number of clusters for Ward

                                                    Description: 
                                                    This function fits several Ward algorithms on specified 
                                                    variables as based on the elements of k. A silhouette 
                                                    score is generated each time (with values closer to 
                                                    1 being the best). The output is a text file containing the
                                                    silhouette scores ('ward_silhouettescores.txt').

scatterplotclusters......................Function : - df : A pandas dataframe

                                                    - var : A list of variables in the dataframe
                                                   
                                                    - labels : A list of cluster labels

                                                    - title : Title to be applied to the plot (string)

                                                    - savename : The name of the image file (string)

                                                    Description: 
                                                    This function plots a 3D scatterplot and applies colors to 
                                                    points based on the cluster labels. The output is an 
                                                    image file (e.g., 'dbscan_3Dscatterplot.png').


getdayarticles...........................Function : - start : A date object (starting day)
                                                    
                                                    - end : A date object (ending day)
                         
                                                    - directory : String for location of the Firefox executable
  
                                                    - login_email : String login email address
 
                                                    - login_password : String login password

                                                    Description: 
                                                    This function mines EventRegistry.org for news events in 
                                                    the US with 50 or more articles written about them. The events
                                                    are sorted by social media score, and the source code is pulled
                                                    which provides the "prototypical" news article representing the
                                                    news event (header text and first several sentences of the article).
                                                    This function may take several hours to complete based on the 
                                                    starting and end dates, since the program must wait for the web
                                                    page to load properly prior to taking the source code (about 30 
                                                    to 50 seconds per day specified). As the website requires login 
                                                    prior to large-scale searches, this function emulates the login
                                                    process and then proceeds with the article mining. This function
                                                    requires Selenium, geckodriver, and a valid installation of 
                                                    Firefox (with the directory to the executable as the 'directory'
                                                    argument.

nn_tester................................Function : - df : A pandas dataframe
        
                                                    - predictor : String specifying the predictor variable in df
 
                                                    - predicted : String specifying the predicted variable in df
 
                                                    - lag : Integer for the number of lag days to use as the input

                                                    - name : String for the name of the image file to be generated

                                                    Description: 
                                                    This is a convenience function created to quickly apply 
                                                    multi-layer perceptron models for longitudinal data. It is 
                                                    based on the premise that the lagged values of the predictor
                                                    are predicting the values of the predicted variable. The function
                                                    uses 100 hidden layers, trains using the first 100 available
                                                    days and tests using the remaining days. It will print the 
                                                    test R-squared (with 1 indicating perfect fit). R-squared can be
                                                    negative since the model can be arbitrarily worse than just the
                                                    mean. The solver is 'lbfgs' which uses Newton steps, and the 
                                                    logistic activation function. The penalty hyperparameter is set
                                                    to 0.01. The function returns the predicted values (for both
                                                    training and test cases -- as a single list) and also produces
                                                    a plot with the name specified.

------------------------------------------------------------------------------------------------------------------

