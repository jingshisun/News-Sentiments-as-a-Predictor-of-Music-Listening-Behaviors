ANLY 501 Project: Part 2
November 8, 2016

This README will explain the contents of the project ZIP file and explain the functions used in the main 
Python script.

##################################################################################################################
IMPORTANT!!:
Prior to running the Python script, there are several prerequites that must be fulfilled.

1) Install the Selenium module:
- This module is not pre-installed in Anaconda, but it can be installed fairly easily through pip. The module
  is used to pull source code from websites through the Firefox browser.

2) Install Firefox:
- It can be downloaded from their website: https://www.mozilla.org/en-US/firefox/new/. The directory to the 
  executable (e.g., C:\Program Files (x86)\Mozilla Firefox\firefox.exe) should also be known, since it is
  required for the use of Selenium.

3) Place the 'geckodriver' file in your Python working directory
- The 'geckodriver' file is necessary to use Firefox in the Selenium module.
- We have provided this file for Windows (64-bit) and Mac OSX in the ZIP folder, but if you are using a 
  different operating system, please download the appropriate file from
  https://github.com/mozilla/geckodriver/releases.

4) Download relevant files for the NLTK module:
- The NLTK module is pre-installed in Anaconda, but several supplemental files need to be downloaded. The 
  three files (corpora) are: stopwords, wordnet, and words.
- To download these files, run the script "import nltk; nltk.download()" which will prompt a second window 
  to open and allow selection and download of the appropriate files.


*Please note that the functions "getarticles()" (which utilizes the Selenium package) and "getinfo()" (which 
collects additional data from Last.fm) are CURRENTLY COMMENTED-OUT (i.e., these are not necessary for data 
cleaning, exploratory analyses, and predictive analyses). These functions are near the top of the main function.
The data these functions collect are also time-sensitive meaning that data collected today will be different 
from the data collected yesterday, etc. However, I do recommend testing these functions out (although they can 
take a while to run).

Please contact me at jmc511@georgetown.edu if anything goes wrong.

##################################################################################################################



I.  DESCRIPTION OF FILES IN ZIP FOLDER:

FILENAME                                                DESCRIPTION
------------------------------------------------------------------------------------------------------------------
article_emotions.txt....................................Description of emotion words pulled from news article 
                                                        headers from eventregistry.org. This is generated from 
                                                        the function 'getarticles'.

associationrules1.txt...................................These provide the results of association rules based on
                                                        minimum support = 0.10. The class method
                                                        'associationcombtest' from class 'associationrules' 
                                                        produces this output.

associationrules2.txt...................................This provides the results of association rules based on
                                                        minimum support = 0.10 (with maximum number of precedents
                                                        in the association rule being 3). The class method
                                                        'associationcombtest' from class 'associationrules' 
                                                        produces this output. The variable 'tagslist' from data
                                                        set 'cleaneddata.csv' was used for association rules.

associationrules3.txt...................................This provides the results of association rules based on
                                                        minimum support = 0.15 (with maximum number of precedents
                                                        in the association rule being 3). The class method
                                                        'associationcombtest' from class 'associationrules' 
                                                        produces this output. The variable 'tagslist' from data
                                                        set 'cleaneddata.csv' was used for association rules.

binning.txt.............................................This provides the results of association rules based on
                                                        minimum support = 0.20 (with maximum number of precedents
                                                        in the association rule being 3). The class method
                                                        'associationcombtest' from class 'associationrules' 
                                                        produces this output. The variable 'tagslist' from data
                                                        set 'cleaneddata.csv' was used for association rules.

cleaneddata.csv.........................................This is the main analysis file that is used to produce 
                                                        all exploratory and predictive analyses. This file is a
                                                        combination of 'getlyrics_output_newfeatures.csv' and 
                                                        'gettracks_output_newfeatures.csv' which were generated in
                                                        Part 1 of the project. In addition, new song data pulled
                                                        through the function 'getinfo' (which is contained in 
                                                        'getinfo_output.csv'). During data cleaning phase, news
                                                        article data pulled from eventregistry.org using the
                                                        'getarticles' function (which is contained in
                                                        'getarticles_output.csv'), and the NRC emotion-association
                                                        lexicon were incorporated into this file in the formation 
                                                        of the emotion word counts (and percentages) in addition
                                                        to the cohesion distance score (which is the Euclidean
                                                        distance between the emotion percentage composition of the
                                                        news articles and the emotion percentage of a given song).

contractions.txt........................................This is a text file that contains common contractions
                                                        (e.g., 'couldn't' is contraction of 'could not'). This was
                                                        created so that the data cleaning functions will have
                                                        access to this dictionary and break up contractions when
                                                        necessary during the determination of word emotion 
                                                        associations.
                                                        
correlations.csv........................................This file contains the Pearson correlation matrix of three
                                                        of the variables contained in 'cleaneddata.csv'. This is 
                                                        generated using the 'getcorrelations' function.

dbscan_3Dscatterplot.png................................This is a 3D scatterplot generated for three variables 
                                                        ("anger_percent", "sadness_percent", "joy_percent") which
                                                        were examined for clustering using DBSCAN. A cluster label
                                                        of "-1" signifies an outlier point (i.e., neither a core
                                                        nor a border point). The remaining numbers are cluster
                                                        labels.

dbscan_groupedmeans.csv.................................This is table showing grouped means for three variables
                                                        ("anger_percent", "sadness_percent", "joy_percent") and 
                                                        each cluster label examined using DBSCAN. Note that the 
                                                        vast majority of points are in the first cluster ("1").

dbscan_kdistance.png....................................This is a plot showing sorted K-distances using the K
                                                        nearest neigbhors algorithm (using Euclidean distances). 
                                                        This generates what is known as a "knee plot" where the 
                                                        distance of steepest incline is the theoretical eps 
                                                        (minimum distance) to be used in DBSCAN. In this case, the
                                                        4th nearest neighbor distances were plotted, and the
                                                        incline point suggests an eps of 0.08.

descriptives.txt........................................This text file presents descriptive statistics for all
                                                        variables that could be potentially used during analyses.
                                                        Quantitative variables are shown with mean, median, and 
                                                        standard deviation, while the two categorical variables
                                                        show the frequencies for the 10 most common occurrences.

geckodriver.............................................This is the geckodriver to be used for Mac OSX for the
                                                        use of the Selenium module. See important notes above.

geckodriver.exe.........................................This is the geckodriver to be used for Windows 64 for the
                                                        use of the Selenium module. See important notes above.

getarticles_output.csv..................................This file is produced from 'getarticles' function, and
                                                        pulls the 25 most shared news articles on social media
                                                        for a given time period (2016-09-19 to 2016-09-26 in our
                                                        case).

getinfo_output.csv......................................This file is produced from 'getinfo' function, and
                                                        pulls more song information (e.g., tags and playcount)
                                                        for each of the songs present in the original 
                                                        'gettracks_output_newfeatures.csv' dataset

getlyrics_output_newfeatures.csv........................This is the song lyrics data pulled from Part 1 of the
                                                        project.

gettracks_output_newfeatures.csv........................This is the song data pulled from Part 1 of the project.

histogram_anger_percent.png.............................This is a histogram for the variable 'anger_percent'
                                                        from 'cleaneddata.csv' using the 'gethistograms'
                                                        function.

histogram_joy_percent.png...............................This is a histogram for the variable 'joy_percent' from 
                                                        'cleaneddata.csv' using the 'gethistograms' function.

histogram_sadness_percent.png...........................This is a histogram for the variable 'sadness_percent' from 
                                                        'cleaneddata.csv' using the 'gethistograms' function.

Hypothesis_Testing.txt..................................This contains all relevant outputs from predictive analyses
                                                        including results of the independent samples t-test, 
                                                        ordinary least squares regression, support vector machines,
                                                        Naive Bayes, Random Forest, Decision Tree, and K nearest
                                                        neighbors (accuracy, confusion matrix, and results of 
                                                        chi-square goodness-of-fit tests based on expected and 
                                                        observed class frequencies are shown).

kmeans_3Dscatterplot.png................................This is a 3D scatterplot generated for three variables 
                                                        ("anger_percent", "sadness_percent", "joy_percent") which
                                                        were examined for clustering using K-means. The three 
                                                        numbers on the legend are cluster labels.

kmeans_centroids.csv....................................This file contains the centroid positions after running
                                                        K-means algorithm with 3 clusters.

kmeans_silhouettescores.txt.............................This file contains the silhouette scores for the K-means
                                                        algorithm for cluster counts varying from 2 to 8. A
                                                        score close to 1 indicates better fit (so K=3 was selected
                                                        in this particular case).

missingdata.txt.........................................This text file contains the number of missing values that
                                                        were found in the dataset, which were then removed.

NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt....This text file contains the NRC word emotion association
                                                        database, which associates a given word with a particular
                                                        emotion. This is used in the class 'word_associations'
                                                        in order to give a count of the number of occurrences 
                                                        of each emotion for a given song or news article. Note
                                                        that documents are first cleaned by breaking contractions,
                                                        words are lemmatized, negations are marked, and stop words
                                                        removed.

project2.py.............................................This file contains all relevant Python code to perform the
                                                        additional data collection, data cleaning, descriptive
                                                        statistics, plots, and predictive analyses. The description
                                                        of each of the functions are shown below.

Project 2 Writeup.pdf...................................This is the report document for Project Part 2.

scatterplotmatrix.png...................................This is a scatterplot matrix provided for three variables
                                                        from 'cleaneddata.csv' using the 'getscatterplots'
                                                        function.

ward_3Dscatterplot.png..................................This is a 3D scatterplot generated for three variables 
                                                        ("anger_percent", "sadness_percent", "joy_percent") which
                                                        were examined for clustering using Ward. The three 
                                                        numbers on the legend are cluster labels.

ward_groupedmeans.csv...................................This is table showing grouped means for three variables
                                                        ("anger_percent", "sadness_percent", "joy_percent") and 
                                                        each cluster label examined using Ward.

ward_silhouettescores.txt...............................This file contains the silhouette scores for the Ward
                                                        algorithm for cluster counts varying from 2 to 8. A
                                                        score close to 1 indicates better fit (so K=3 was selected
                                                        in this particular case).
------------------------------------------------------------------------------------------------------------------





II. DESCRIPTION OF FUNCTIONS/CLASSES/METHODS:

FUNCTION NAME                            DESCRIPTION
------------------------------------------------------------------------------------------------------------------
associationrules.........................Class    : - listoflists : A list containing elements that are also lists
                                                        in the same fashion as a series of transactions with 
                                                        with each transaction containing several items. 
                                                    
                                                    - minsup : The minimum support (between 0 to 1).

                                                    Description:
                                                    Constructor produces 'associationrules' objects, which takes
                                                    a list of lists (i.e., a list of items for each row) and the
                                                    minimum support (0 to 1), and produces a list of valid items
                                                    that pass the apriori principle (as based on minimum support).
                                                    A dataframe of occurrences of each valid item per row is 
                                                    generated (that contains a Boolean value) which can then be
                                                    used to generate confidence values for association rules.

associationrules.confidence........ ......Method  : - precedent : A list of items that comprise the precedent in 
                                                        an association rule.
                                                     
                                                    - antecedent : A list of items that comprise the antecedent in
                                                        an association rule. The method is currently limited to
                                                        lists with only a single element.
                                                    
                                                    Description:
                                                    This method calculates the confidence of a given association
                                                    rule taking in the precedent items (as a list) and the 
                                                    antecedent items (as a list).

associationrules.associationcombtest......Method  : - maxitems : The maximum number of items to be used in all
                                                        combinations of precedents.

                                                    - minconf : The minimum confidence of the association rules to 
                                                        be displayed.

                                                    - filename : The name of the text output.

                                                    Description:
                                                    This method loops the confidence method based on all possible
                                                    combinations of precedents and a single antecedent, and a 
                                                    given number of maximum items in the precedent list. As one 
                                                    can probably imagine, this method is quite slow in most cases.
                                                    The method prints out which association rules are being
                                                    examined, so that the user can track its progress. Output is
                                                    'associationrules1.txt', 'associationrules2.txt', and 
                                                    'associationrules3.txt'.

binning..................................Function : - df: The pandas dataframe being used.

                                                    - variable: The variable in the dataframe to be binned.

                                                    - levels : The number of resulting levels from the binning. It
                                                        divides the variable by quantile (equal density).

                                                    Description:
                                                    This function creates a new variable in the specified 
                                                    dataframe that divides the variable into the specified number 
                                                    (as specified by levels).

break_contractions.......................Function : - text : A string -- usually in the form of an entire document
                                                        or sentence.

                                                    Description: This function uses the file 'contractions.txt' 
                                                    which contains a dictionary of contraction words (e.g., can't,
                                                    wouldn't) and their deconstructed words. Function searches
                                                    the text and replaces instances of contractions with the
                                                    comprised words.

cleandata................................Function : - None

                                                    Description: 
                                                    This function merges three dataframes: 
                                                    'gettracks_output_newfeatures.csv',                       
                                                    'getlyrics_output_newfeatures.csv', and 'getinfo_output.csv', 
                                                    removes duplicate instances of songs, removes cases without 
                                                    any lyrics, cases with zero duration, and adds several new 
                                                    variables (count of emotional words in each song, and a
                                                    cohesion score for a song's similarity to the same week's news
                                                    events). Output is 'cleaneddata.csv'.

dbscan_groupmeans........................Function : - df : A pandas dataframe
                     
                                                    - var : A list of variables in the dataframe
                             
                                                    - minpts : Minimum number of points for DBSCAN
  
                                                    - eps : The minimum distance to produce densities for DBSCAN

                                                    Description: 
                                                    This function runs DBSCAN based on minpts and eps on the 
                                                    specified variables. It outputs a CSV file containing the 
                                                    means for each variable broken down by the clusters that 
                                                    resulted from DBSCAN. Output is 'dbscan_groupedmeans.csv' and
                                                    also returns the cluster labels.

dbscan_kdistanceplot.....................Function : - df : A pandas dataframe

                                                    - var : A list of variables in the dataframe

                                                    - k : The number of neighbors for K nearest neighbors

                                                    Description: 
                                                    This function is used to estimate an appropriate eps for
                                                    DBSCAN. A K nearest neighbors is run, and a plot showing 
                                                    the distance to the Kth neighbor sorted from smallest to 
                                                    largest. The distance where the large incline is shown (also
                                                    known as the 'knee') is the suggested eps. Output is 
                                                    'dbscan_kdistance.png'.

Decision_Tree............................Function : - Y : A pandas dataframe (1-dimensional)

                                                    - X : A pandas dataframe (k-dimensional)

                                                    Description: 
                                                    This function fits a decision tree model using the sklearn 
                                                    package. X is a dataframe containing all predictors (1 per 
                                                    column). The return value is a list containing the accuracy 
                                                    score, confusion matrix, and the p-value of a chi-square 
                                                    goodness-of-fit test.

descriptives.............................Function : df : A pandas dataframe.

                                                    Description:
                                                    This function produces descriptive statistics for several
                                                    predetermined variables. Means, medians, and standard 
                                                    deviations are produced for quantitative variables and the top
                                                    10 most common occurrences are shown for categorical
                                                    variables. The output is 'descriptives.txt'.

getarticles..............................Function : - datestart : Starting date of news articles (e.g., 
                                                        '2016-09-19')

                                                    - dateend : Ending data of news articles (e.g., '2016-09-26')

                                                    - directory : Directory that specifies the location of the
                                                        Firefox executable in the current machine (e.g., 
                                                        r'C:\Program Files (x86)\Mozilla Firefox\firefox.exe')

                                                    Description:
                                                    This function uses Selenium and Firefox to scrape news article
                                                    data from eventregistry.org. The function includes several
                                                    intentional pauses in order to give the browser enough time
                                                    to load the webpage, click a button to sort by social media
                                                    popularity, and then scrape the source code from the site. In
                                                    total, the function will take roughly one minute to complete
                                                    (please do not close the window, it will close on its own). 
                                                    Upon completion, an error will show ('NoneType' object has no
                                                    attribute 'path') but the function will still work properly. 
                                                    Output is 'getarticles_output.csv'.

getcorrelations..........................Function : - df : A pandas dataframe

                                                    - var : A list of variables in the dataframe

                                                    Description: 
                                                    This function will generate a correlation matrix using 
                                                    the specified variables in the specified dataframe. The
                                                    output is 'correlations.csv'.

gethistograms............................Function : - df : A pandas dataframe

                                                    - var : A list of variables in the dataframe

                                                    Description: 
                                                    This function will generate histograms for each of the 
                                                    variables specified. The output is a single histogram
                                                    per variable in PNG format (with suffix "histogram_").

getinfo..................................Function : - api_key : A Last.fm API key (string)

                                                    Description: 
                                                    This function will use the Last.fm API in order to collect 
                                                    more information for each of the songs in the dataset 
                                                    "gettracks_output.csv". New variables include top 5
                                                    tags per song, the number of listeners per song, the 
                                                    playcount for the song, and a more detailed duration variable.
                                                    All song information is based on the past 7 days.

getscatterplots..........................Function : - df : A pandas dataframe

                                                    - var : A list of variables in the dataframe

                                                    Description: 
                                                    This function produces a scatterplot matrix using the 
                                                    specified variables in the specified dataframe. The diagonal 
                                                    portions of the scatterplot matrix are histograms. Output 
                                                    is 'scatterplotmatrix.png'.

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

KNN......................................Function : - Y : A pandas dataframe (1-dimensional)

                                                    - X : A pandas dataframe (k-dimensional)

                                                    Description: 
                                                    This function fits a K nearest neighbors model using the 
                                                    sklearn package. X is a dataframe containing all predictors (1 
                                                    per column). The return value is a list containing the accuracy 
                                                    score, confusion matrix, and the p-value of a chi-square 
                                                    goodness-of-fit test.
                                                    
lemmatize_words..........................Function : - text : A list of strings

                                                    Description:
                                                    This function reduces words to their root word (e.g., 'words'
                                                    becomes 'word') which can then be used for emotion 
                                                    association. The function uses the NLTK Wordnet lemmatizer.
                                                    The function returns a list of lemmatized words.

Linear_Regression........................Function : - outcome : A numeric list for predicted variable

                                                    - predictors : A list of arrays (1 column per predictor)

                                                    Description: 
                                                    This function completes OLS regression using the ols()
                                                    function. The output is a string based on the ols() function
                                                    output.

Naive_Bayes..............................Function : - Y : A pandas dataframe (1-dimensional)

                                                    - X : A pandas dataframe (k-dimensional)

                                                    Description: 
                                                    This function fits a naive Bayes model using the sklearn 
                                                    package. X is a dataframe containing all predictors (1 per 
                                                    column). The return value is a list containing the accuracy 
                                                    score, confusion matrix, and the p-value of a chi-square 
                                                    goodness-of-fit test.

Random_Forest............................Function : - Y : A pandas dataframe (1-dimensional)

                                                    - X : A pandas dataframe (k-dimensional)

                                                    Description: 
                                                    This function fits a random forest model using the sklearn 
                                                    package. X is a dataframe containing all predictors (1 per 
                                                    column). The return value is a list containing the accuracy 
                                                    score, confusion matrix, and the p-value of a chi-square 
                                                    goodness-of-fit test.

remove_stopwords.........................Function : - text : A list of strings

                                                    Description:
                                                    This function uses the NLTK stopword corpus to remove elements
                                                    that are stopwords (e.g., and, if, then) which are basically
                                                    meaningless for the purposes of emotion association. Output is
                                                    list of words that are not stopwords.

scatterplotclusters......................Function : - df : A pandas dataframe

                                                    - var : A list of variables in the dataframe
                                                   
                                                    - labels : A list of cluster labels

                                                    - title : Title to be applied to the plot (string)

                                                    - savename : The name of the image file (string)

                                                    Description: 
                                                    This function plots a 3D scatterplot and applies colors to 
                                                    points based on the cluster labels. The output is an 
                                                    image file (e.g., 'dbscan_3Dscatterplot.png').

SVM......................................Function : - Y : A pandas dataframe (1-dimensional)

                                                    - X : A pandas dataframe (k-dimensional)

                                                    Description: 
                                                    This function fits a SVM model using the sklearn package. X 
                                                    is a dataframe containing all predictors (1 per column). The 
                                                    return value is a list containing the accuracy score, confusion
                                                    matrix, and the p-value of a chi-square goodness-of-fit test.

T_test...................................Function : - var1 : A numeric list
                                                   
                                                    - var2 : A numeric list

                                                    Description: 
                                                    This function performs an independent samples t-test based on
                                                    the assumption of unequal variances (i.e., Welch's t-test). 
                                                    The output is a string of the ttest_ind function output.
                                                    
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
------------------------------------------------------------------------------------------------------------------

