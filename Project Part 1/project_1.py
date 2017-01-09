# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:42:00 2016

@author: tomec
"""

import requests
import csv
import urllib
import re
import unicodedata
import pandas as pd
import string

################################## GETTRACKS ##################################

## This program will use the last.fm API in order to collect data on the most
## popular tracks in the US during the previous week.

def gettracks(api_key, perpage, pages):
    
    # Clear contents of output file (if it exists) and create header row.
    
    with open("gettracks_output.csv", "w", newline = '', encoding='utf-8') as f:
        fieldnames = ['page', 'name', 'songmbid', 'artistname', 'artistmbid', 
                      'artisturl', 'duration', 'listeners', 'url', 'rank', 
                      'streamstat', 'streamtext', 'imagesmallurl', 'imagemedurl',
                      'imagelargeurl', 'imagexlargeurl']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
    
    # Loop through 'pages' number of pages and grab 'perpage' number
    # of tracks per page.        
    
    for i in range(1, pages + 1):
    # get json data from API
        url = {'method': 'geo.getTopTracks',
               'country': 'United States',
               'page': i,
               'limit': perpage,
               'api_key': api_key,
               'format': 'json'}
        response = requests.get("http://ws.audioscrobbler.com/2.0/", url)
        txt = response.json()
        
        for song in txt['tracks']['track']:
            name = song['name']
            artistname = song['artist']['name']
            artistmbid = song['artist']['mbid']
            artisturl = song['artist']['url']
            duration = song['duration']
            url = song['url']
            rank = song['@attr']['rank']
            songmbid = song['mbid']
            streamstat = song['streamable']['fulltrack']
            streamtext = song['streamable']['#text']
            listeners = song['listeners']
            imagesmallurl = song['image'][0]['#text']
            imagemedurl = song['image'][1]['#text']
            imagelargeurl = song['image'][2]['#text']
            imagexlargeurl = song['image'][3]['#text']
            with open("gettracks_output.csv", "a", newline = '', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames = fieldnames)
                writer.writerow({'page': i,
                                 'name': name,
                                 'songmbid': songmbid,
                                 'artistname': artistname,
                                 'artistmbid': artistmbid,
                                 'artisturl': artisturl,
                                 'duration': duration,
                                 'listeners': listeners,
                                 'url': url,
                                 'rank': rank,
                                 'streamstat': streamstat,
                                 'streamtext': streamtext,
                                 'imagesmallurl': imagesmallurl,
                                 'imagemedurl': imagemedurl,
                                 'imagelargeurl': imagelargeurl,
                                 'imagexlargeurl': imagexlargeurl})


################################## GETLYRICS ##################################

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


def getlyrics():
    
    # Get track and artist info from csv that was collected earlier
    with open("gettracks_output.csv", "r", newline = '', encoding = "utf-8") as file:
        input_file = csv.DictReader(file)
        tracks = []
        for row in input_file:
            tracks.append(row)  
    
    # Create header for csv output file
    with open("getlyrics_output.csv", "w", newline = '', encoding='utf-8') as f:
        fieldnames = ['url', 'lyrics']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
    
    # Main regex search pattern
    Pattern = re.compile('lyricbox..>(.+?)<div class=..lyricsbreak')
    
    for song in tracks:
        # Attempt initial search using the raw song and artist name
        url = "http://lyrics.wikia.com/wiki/" + song['artistname'] + ":" + song['name']
        url = remove_parenth(url)             # url: remove parentheses and its contents
        url = url.strip().replace(" ", "_")   # url: replace spaces with underscores
        url = replace_special(url)            # url: replace non-convertible special characters
        url = replace_accents(url)            # url: remove accents on characters

        req = urllib.request.Request(url)     # create Request object      
        print("Getting lyrics from: " + req.get_full_url()) # print full url passed to urlopen
        
        try:
            data = urllib.request.urlopen(req)    # open site and pull html
            getdata = str(data.read())            # convert html to byte string
            output = re.findall(Pattern, getdata) # search using main regex pattern 
            
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
                print("Getting lyrics from: " + req.get_full_url()) # print full url passed to urlopen
                
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
            with open("getlyrics_output.csv", "a", newline = '', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames = fieldnames)
                writer.writerow({'url': song['url'], 'lyrics': lyrics})
           
        # This is the last-resort case where there are no reasonable matches   
        except Exception:
            with open("getlyrics_output.csv", "a", newline = '', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames = fieldnames)
                writer.writerow({'url': song['url'], 'lyrics': 'Not found'})
            pass


################################## DATASTATS ##################################

#This function find the percentage of missing values for each attribute from a data frame.
#and takes the average of them to get a quality score.   
def quality_score(dataframe, outfile, nameofData):
    
    list = dataframe.columns.values.tolist() #List to record all the attributes in a dataframe.
    sum = 0
    numbersOfAttributes = len(list)
    for attribute in list:
        #If the attribute is "rank" which is a special case, only blanks will count as missing value.
        if(attribute =="rank"):
            count = 0
            for i in range(len(dataframe)): 
                    if (pd.isnull(dataframe.ix[i,attribute])):
                        count=count+1
        else:
            count = 0
            for i in range(len(dataframe)): 
                    #The types of missing values for this function are string "Not found", blanks and zeros.
                    if (pd.isnull(dataframe.ix[i,attribute]) or dataframe.ix[i,attribute] == "Not found" or dataframe.ix[i,attribute] == 0):
                        count=count+1
        sum = sum + count/len(dataframe)
    #return '{percent:.2%}'.format(percent=sum/numbersOfAttributes)
    outfile.write("\n"+"The quality score for "+ nameofData +" is "'{percent:.2%}'.format(percent=sum/numbersOfAttributes) + " which is calculated by averaging the fractions (of missing values for all the other attributes) in the data set" +"\n")
 

#This function find the percentage of missing values for each attribute from a data frame.   
def fraction_of_missing(dataframe, outfile, nameofData):
    
    list = dataframe.columns.values.tolist()#List to record all the attributes in a dataframe.
    for attribute in list:
        #If the attribute is "rank" which is a special case, only blanks will count as missing value.
        if(attribute == "rank"):
            count = 0
            for i in range(len(dataframe)): 
                    if (pd.isnull(dataframe.ix[i,attribute])):
                        count=count+1
        else:
            count = 0
            for i in range(len(dataframe)):
                    #The types of missing values for this function are string "Not found", blanks and zeros.
                    if (pd.isnull(dataframe.ix[i,attribute]) or dataframe.ix[i,attribute] == "Not found" or dataframe.ix[i,attribute] == 0):
                        count=count+1
        #return ("The fraction of missing values for "+attribute+" is :"+'{percent:.2%}'.format(percent=count/len(dataframe)))
        outfile.write("The fraction of missing values for "+ attribute +" in "+ nameofData +" is: "+'{percent:.2%}'.format(percent=count/len(dataframe))+"\n")
 
#This function checks the number of duplicates within a list    
def number_unique(dataframe, alist, outfile, nameofData):
    
    length = len(dataframe[alist])
    ulength = len(set(dataframe[alist]))
    outfile.write("There are "+ str(length-ulength) +" duplicates in "+ nameofData + " from a total of " + str(length) + "\n")


def datastats():
     
    tracksDataFrame = pd.read_csv('gettracks_output.csv', sep=',', encoding='utf-8') 
    lyricsDataFrame = pd.read_csv('getlyrics_output.csv', sep=',', encoding='utf-8')
    file = open("Data Quality.txt", "w")
    number_unique(tracksDataFrame, 'url', file, "tracks data set")
    number_unique(lyricsDataFrame, 'url', file, "lyrics data set")
    fraction_of_missing(tracksDataFrame, file, "tracks data set")
    fraction_of_missing(lyricsDataFrame, file, "lyrics data set")
    quality_score(tracksDataFrame, file, "tracks data set")
    quality_score(lyricsDataFrame, file, "lyrics data set")
    file.close()


################################# ADDFEATURES #################################

def addfeatures():
    
    ####### SECTION I: ADD FEATURES TO THE TRACKS OUTPUT FILE ('gettracks_output.csv') #######    
    
    
    ### This section will create a feature that flags duplicate songs from 'gettracks_output.csv'
    
    with open("gettracks_output.csv", "r", newline = '', encoding = 'utf-8') as f:
        reader = csv.reader(f)
        uid = []
        artists = []
        for row in reader:
            # Column 9 contains variable 'url' which is the unique song url
            uid.append(row[8]) 
            # Column 4 contains variable 'artistname' to be used later
            artists.append(row[3])
    
    # Delete first element from each (since it is the header)
    del uid[0]
    del artists[0]
    
    bank = []       # Create a bank of unique uid's
    duplicates = [] # This will become a Boolean list: 0=First instance, 1=Duplicate
    for i in uid:
        if i not in bank:
            duplicates.append(0)
            bank.append(i)
        else:
            duplicates.append(1)
    
    ### This section will create a feature that counts the number of times the artist appears
    ### in the list of top 5000
   
    # Create a list of lists: first list is artist name, second list is frequency
    artistlist = [[artists[0]], [1]] # artists[0] is header
    counter = 2
    for i, k in zip(artists[1:], duplicates[1:]):
        if counter == 5000: # Limit to 5000 valid rows
            break
        elif i in artistlist[0] and k == 0: # If in artistlist already and is not a duplicate
            artistlist[1][artistlist[0].index(i)] = artistlist[1][artistlist[0].index(i)] + 1
            counter = counter + 1
        elif k == 0:
           # If artist is new to artist list, add them
           artistlist[0].append(i)
           artistlist[1].append(1)
           counter = counter + 1

    # Create new variable that provides frequency in top 5000 for each track in 'gettracks_output.csv'
    artistfreq = []
    for i, k in zip(artists, duplicates):
        if i in artistlist[0]:
            artistfreq.append(artistlist[1][artistlist[0].index(i)])
        else:
            artistfreq.append(0) # If artist is not found on top 5000, append zero

    ### This section will create a new global ranking variable (original data had ranks
    ### restart from zero on each page pulled from the API)
    
    counter = 1
    fullrank = []
    for i in duplicates:
        # We want to create a rank only for those that are not duplicates
        if i == 0:
            fullrank.append(counter) # Rank becomes whatever is in the counter
            counter = counter + 1     # Iterate to next rank
        else:
            # Duplicates get 'None' instead of a rank
            fullrank.append(None)

    ### At this stage, we want to add these two new columns to the 'gettracks_output.csv'
    ### file. The new csv file will be called 'gettracks_output_newfeatures.csv'
    
    # Clear contents of output file (if it exists) and create header row with two extra
    # variables we want to add
    with open("gettracks_output_newfeatures.csv", "w", newline = '', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        header = ['page', 'name', 'songmbid', 'artistname', 'artistmbid', 
                  'artisturl', 'duration', 'listeners', 'url', 'rank', 
                  'streamstat', 'streamtext', 'imagesmallurl', 'imagemedurl',
                  'imagelargeurl', 'imagexlargeurl','duplicate', 'fullrank', 'artistfreq']
        writer.writerow(header)
    
    # Open the original output file in read mode
    with open("gettracks_output.csv", "r", newline = '', encoding = 'utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Iterate to second row (since header row is already written in new output)
        # Open the new output file in append mode
        with open("gettracks_output_newfeatures.csv", "a", newline = '', encoding = 'utf-8') as g:    
            writer = csv.writer(g)
            i = 0 # Use for iterating the duplicates, fullrank, and artistfreq lists
            for j in reader:
                j.append(duplicates[i]) # Append ith element of duplicates
                j.append(fullrank[i])   # Append ith element of fullrank
                j.append(artistfreq[i]) # Append ith element of artistfreq
                writer.writerow(j)
                i = i + 1

    ####### SECTION II: ADD FEATURES TO THE TRACKS OUTPUT FILE ('getlyrics_output.csv') #######   

    ### This section will create a variable for the 'getlyrics_output' dataset for the 
    ### number of characters in the lyrics variable
    
    # Clear contents of output file (if it exists) and create header row with extra variable
    with open("getlyrics_output_newfeatures.csv", "w", newline = '', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        header = ['url', 'lyrics', 'lyricslength']
        writer.writerow(header)
    
    # Create a tuple of punctuation characters (to be used later)
    exclude = set(string.punctuation)
    
    with open("getlyrics_output.csv", "r", newline = '', encoding = 'utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Iterate to second row (since header row is already written in new output)
        with open("getlyrics_output_newfeatures.csv", "a", newline = '', encoding = 'utf-8') as g:
            writer = csv.writer(g)
            for row in reader:
                # Second column contains lyrics
                out = row[1]
                # If the lyrics were not found, do not find character length
                if row[1] == 'Not found':
                    row.append(None)
                else:
                    # Strip punctuation characters
                    x = []
                    for i in out:
                        if i not in exclude:
                            x.append(i)
                    out = "".join(x)
                    # Strip spaces
                    out = row[1].replace(" ", "")
                    # Append number of characters after stripping the lyric string
                    row.append(len(out))
                writer.writerow(row)

  
if __name__ == "__main__":
    
    # For testing purposes, we recommend using a smaller number for 'perpage' and 'pages'
    # like perpage = 5, pages = 10 for a total of 50 sonngs. Due to the way the API works,
    # there is a high chance of duplicates appearing, though you will still get 50 unique
    # songs.
    gettracks(api_key = '21b84b7984893188a7dd09863171b052', perpage = 101, pages = 56)
    getlyrics()
    datastats()
    addfeatures()