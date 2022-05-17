%%time

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


QUIT =  list(["stop","quit", "end", "Stop","Quit","End", "STOP", "QUIT", "END"])


# running the chatbot
print('START CHATTING WITH THE ROBO-CHAT')
while True:
    message = input("")
    
    if message in QUIT:
        
        break
    
    intents = pred_class(message, words, classes)
    result = get_response(intents, dataset)
    print(result)

print("Okay, have a great day!")
