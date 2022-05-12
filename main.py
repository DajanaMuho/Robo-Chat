# used a dictionary to represent an intents JSON file
data = {"intents": [
             {"tag": "greeting",
              "patterns": ["Hello", "How are you?", "Hi there", "Hi", "Whats up"],
              "responses": ["Howdy Partner!", "Hello", "How are you doing?", "Greetings!", "How do you do?"],
             },
             {"tag": "age",
              "patterns": ["how old are you?", "when is your birthday?", "when was you born?"],
              "responses": ["I am 24 years old", "I was born in 1996", "My birthday is July 3rd and I was born in 1996", "03/07/1996"]
             },
             {"tag": "date",
              "patterns": ["what are you doing this weekend?",
"do you want to hang out some time?", "what are your plans for this week"],
              "responses": ["I am available all week", "I don't have any plans", "I am not busy"]
             },
             {"tag": "name",
              "patterns": ["what's your name?", "what are you called?", "who are you?"],
              "responses": ["My name is Kippi", "I'm Kippi", "Kippi"]
             },
             {"tag": "goodbye",
              "patterns": [ "bye", "g2g", "see ya", "adios", "cya"],
              "responses": ["It was nice speaking to you", "See you later", "Speak soon!"]
             }
]}
print("DUMB DATA", data)

# ADDED - E. Miller
# Testing to see if changes showing

print("TEST _ EMILLER")



###################################################################################
# E. Miller May 11, 2020 9:43 am

import numpy as np
import pandas as pd
import nltk

# basic libraries

import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Text pre-processing library

nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('omw-1.4')

# downloading text preprocessing data

WikiQA1 = pd.read_csv("WikiQACorpus/WikiQA.tsv", sep = '\t' )

# reading in main tsv file

WikiQA1.head()

# looking at first few entries




###################################################################################
# E. Miller May 12, 2020 9:15 am


# ADDING A PRE-PROCESSING FUNCTION FROM ASSIGNMENT 2
# MIGHT NEED TO ADD A FEW MORE STEPS HERE

        
def text_preprocessor(text_df):
    
    # this function takes in a data frame of text only columns
    # and goes through each column and processes the text data
    # into a format that is more easily analyzed for prediction
    # It then returns the converted text data frame
    
    num_rows = len(text_df)
    num_cols = len(text_df.columns)
    
    # getting number of rows and columns in data frame
    
    TXT = pd.DataFrame(columns = text_df.columns, index = text_df.index)
    
    # setting up a new data frame to fill with processed text
    
    punctuation = ",!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    
    # Saving a list of commonly used punctuation to a string in order to remove
    # them from the text columns
    
    stop_words = stopwords.words("english")
    
    # getting list of common stopwords to remove from the text entries
    
    lemma = WordNetLemmatizer()
    
    # setting up lemmatizing function
    
    stemm =  PorterStemmer()
        
    for i in range(num_cols):
        
        # loop through the columns in the data frame
        
        converted_text = []
        
        # reset the converted text list so we can use it
        # in the column
    
        for j in range(num_rows):
            
            # go through each row in the column
            
            new_text = text_df.iloc[j][i].lower()
                        
            # makes text lower case
            
            # NOTE: i and j are reversed for this section,
            # since we are going through each row in the column
            
            for char in new_text:
                
                if char in punctuation:
                    
                    new_text = new_text.replace(char, " ")
            
            # removing punctuation
                        
            new_text = new_text.split()
            
            # splitting up the text into individual words
            
            for word in new_text:
    
                if word in stop_words:
        
                    new_text.remove(word)
            
            # removing stop words
            
            for word in new_text:
                
                lemma.lemmatize(word)
                
            # lemmatizing words
            
            for word in new_text:
                
                stemm.stem(word)
            
            # stemming words
            
            new_text = " ".join(new_text)
            
            converted_text.append(new_text)
        
            # add the converted text value to the list
        
        column_name = text_df.columns[i]
        
        # get the name of the column that has been converted
        
        TXT[ column_name ] = converted_text
        
        # fill the rows of the corresponding column in TXT
        # with the list of processed text and then return TXT after all
        # columns are filled
    
    return TXT
