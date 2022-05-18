import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import pandas as pd

from robochat import lemmatizer

nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

# ADDING STOPWORDS - E MILLER
nltk.download("stopwords")

# ADDING CONTRACTION EXPANDER - E MILLER
import contractions

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


class PreProcessing:
    def __init__(self, dataset):
        self.dataset = dataset
        self.words = []
        self.classes = []
        self.doc_X = []
        self.doc_y = []

    def pre_process_words(self):

        # Organize tags and patterns
        for index, row in self.dataset.iterrows():
            pattern = row["Pattern"]

            ########## ADDED CODE - E_Miller 5_18

            pattern = text_preprocessor(pattern)

            # Calling text_preprocessor function to add processing to pattern

            ########## ADDED CODE - E_Miller 5_18


            tokens = nltk.word_tokenize(pattern)
            self.words.extend(tokens)
            self.doc_X.append(pattern)
            self.doc_y.append(row["Tag"])
            if row["Tag"] not in self.classes:
                self.classes.append(row["Tag"])

        # Lemmatize the words
        self.words = [lemmatizer.lemmatize(word.lower()) for word in self.words if word not in string.punctuation]

        # sort the set to remove duplicated words
        words = sorted(set(self.words))
        classes = sorted(set(self.classes))

        # print(self.words)
        # print(self.classes)
        # print(self.doc_X)
        # print(self.doc_y)

    def bag_of_words_model(self):
        # convert data to numerical values
        training = []
        out_empty = [0] * len(self.classes)

        for idx, doc in enumerate(self.doc_X):
            bow = []
            text = lemmatizer.lemmatize(doc.lower())
            for word in self.words:
                bow.append(1) if word in text else bow.append(0)
            output_row = list(out_empty)
            output_row[self.classes.index(self.doc_y[idx])] = 1
            training.append([bow, output_row])

        return training

    # EDWARDS CODE TO MERGE CSVs
    """
        QUESTION_COL = WikiQA1.loc[:, ["Question"]]
        QUESTIONS_REDUCED = text_preprocessor(QUESTION_COL)
        display(QUESTIONS_REDUCED)
        dataset_WIKI = pd.DataFrame(columns=dataset.columns)
        dataset_WIKI["Pattern"] = QUESTIONS_REDUCED
        dataset_WIKI["Responses"] = WikiQA1.loc[:, ["Sentence"]]
        # dataset_WIKI["Tag"] = str("keywords")
        dataset_WIKI = dataset_WIKI.assign(Tag='keywords')
        display(dataset_WIKI)
        dataset_FULL = pd.concat([dataset_WIKI, dataset], axis=0)
        # combining the two datasets
        display(dataset_FULL)
    """
def text_preprocessor(text):

    # this function takes in a string of text and processes it
    # so that it can more easily fit into the neural network

    punctuation = ",!\#$%&()*+-./:;<=>?@[\]^_`{|}~'"
    # Saving a list of commonly used punctuation to a string in order to remove
    # them from the text columns
    stop_words = stopwords.words("english")
    # getting list of common stopwords to remove from the text entries
    lemma = WordNetLemmatizer()
    # setting up lemmatizing function
    stemm = PorterStemmer()

    # SETTING UP NEEDED FUNCTIONS TO PROCESS TEXT

    expanded_words = []
    # creating an empty list for contractions
    for word in text.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))

    expanded_text = ' '.join(expanded_words)

    # EXPANDING CONTRACTIONS

    new_text = expanded_text.lower()

    # MAKE TEXT LOWER CASE

    for char in new_text:
        if char in punctuation:
            new_text = new_text.replace(char, " ")

    # REMOVING PUNCTUATION

    new_text = new_text.split()

    # SPLITTING UP TEXT INTO INDIVIDUAL WORDS

    for word in new_text:
        if word in stop_words:
            new_text.remove(word)

    # REMOVING STOP WORDS

    for word in new_text:
        lemma.lemmatize(word)

    # LEMMATIZING WORDS

    for word in new_text:
        stemm.stem(word)

    # STEMMING WORDS

    new_text = " ".join(new_text)

    # PUTTING REDUCED BACK TOGETHER TO RETURN

    return new_text