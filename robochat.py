import pandas as pd
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import colorama
colorama.init()
from colorama import Fore, Style, Back

nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

#####################################################

# Read the dataset

QA_url = "https://raw.githubusercontent.com/DajanaMuho/Robo-Chat/main/Q%26A.csv"

# Downloading chatbot csv with only a few responses

dataset = pd.read_csv(QA_url)

#####################################################

words = []
classes = []
doc_X = []
doc_y = []

# Organize tags and patterns
for index, row in dataset.iterrows():
    pattern = row["Pattern"]
    tokens = nltk.word_tokenize(pattern)
    words.extend(tokens)
    doc_X.append(pattern)
    doc_y.append(row["Tag"])
    if row["Tag"] not in classes:
        classes.append(row["Tag"])

# Lemmatize the words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# sort the set to remove duplicated words
words = sorted(set(words))
classes = sorted(set(classes))

print(words)
print(classes)
print(doc_X)
print(doc_y)

# convert data to numerical values
training = []
out_empty = [0] * len(classes)

for idx, doc in enumerate(doc_X):  # Bag of Words Model
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Feed training data into a Neural Network Model

input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200
# the deep learning model
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation="softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)


# CREATE THE CHATBOT
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list


def get_response(intents_list, data):
    tag = intents_list[0]
    for index_i, row_i in data.iterrows():
        if row_i["Tag"] == tag:
            result = random.choice(data[data['Tag'] == tag]['Responses'].values)
            break
    return result



QUIT = list(["stop", "quit", "end", "Stop", "Quit","End", "STOP", "QUIT", "END"])

# List of words to end the chatbot conversation

def RUN_CHATBOT():

    # running the chatbot
    print('START CHATTING WITH THE ROBO_CHAT, \n\nTYPE any of the words to stop the conversation:\n',QUIT,'\n')

    while True:
        print(Fore.YELLOW + "HUMAN:" + Style.RESET_ALL, end="")
        message = input("")
        if message in QUIT:
            print(Fore.RED + "Ending the chat\n " + Style.RESET_ALL, end="")
            break
        intents = pred_class(message, words, classes)
        result = get_response(intents, dataset)
        print(Fore.GREEN + "ROBO-CHAT:" + Style.RESET_ALL, result)


RUN_CHATBOT()   # TESTING FUNCTION - BASIC CHATBOT

def text_preprocessor(text_df):
    
    # this function takes in a data frame of text only columns
    # and goes through each column and processes the text data
    # into a format that is more easily analyzed for prediction
    # It then returns the converted text data frame
    
    num_rows = len(text_df)
    num_cols = len(text_df.columns)
    
    # getting number of rows and columns in data frame
    
%%time
#WikiQA1_CONVERTED = text_preprocessor(WikiQA1)

QUESTION_COL = WikiQA1.loc[:, ["Question"]]

QUESTIONS_REDUCED = text_preprocessor(QUESTION_COL)
    
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

QUESTION_COL = WikiQA1.loc[:, ["Question"]]

QUESTIONS_REDUCED = text_preprocessor(QUESTION_COL)

display(QUESTIONS_REDUCED)

dataset_WIKI = pd.DataFrame(columns = dataset.columns)

dataset_WIKI["Pattern"] = QUESTIONS_REDUCED

dataset_WIKI["Responses"] =  WikiQA1.loc[:, ["Sentence"]]

#dataset_WIKI["Tag"] = str("keywords")

dataset_WIKI = dataset_WIKI.assign(Tag='keywords')

display(dataset_WIKI)

dataset_FULL = pd.concat([dataset_WIKI,dataset], axis=0)

# combining the two datasets

display(dataset_FULL)


#TEST UPLOAD FROM PYCHARM - EMILLER