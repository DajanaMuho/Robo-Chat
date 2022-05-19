import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import pandas as pd
from nltk.stem import WordNetLemmatizer
# ADDING CONTRACTION EXPANDER - E MILLER
import contractions

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure

# Added imports E - MILLER

lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

# ADDING STOPWORDS - E MILLER
nltk.download("stopwords")


def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def text_preprocessor(text):
    # this function takes in a string of text and processes it
    # so that it can more easily fit into the neural network
    punctuation = ",!#$%&()*+-./\:;<=>?@[]^_`{|}~'"
    # Saving a list of commonly used punctuation to a string in order to remove
    # them from the text columns
    stop_words = stopwords.words("english")
    # getting list of common stopwords to remove from the text entries
    lemma = WordNetLemmatizer()
    # setting up lemmatizing function
    stem = PorterStemmer()
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
        stem.stem(word)

    # STEMMING WORDS
    new_text = " ".join(new_text)

    # PUTTING REDUCED BACK TOGETHER TO RETURN
    return new_text


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
            # ADDED CODE - E_Miller 5_18
            pattern = text_preprocessor(pattern)
            # Calling text_preprocessor function to add processing to pattern
            # ADDED CODE - E_Miller 5_18
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

    # VISUALIZE CODE
    def Stop_Word_Plotter(self):
        # this function pre-processes the text data in a single text column and then finds out
        # how many stop words were removed and plots them in a bar graph
        """####### BASIC SECTION ########
        #Q_A_BASIC = pd.read_csv("https://raw.githubusercontent.com/DajanaMuho/Robo-Chat/main/Q%26A.csv")
        # read in the CSV from the Github site
        #text_df = Q_A_BASIC["Pattern"]
        # save the "Pattern" column into text_df
        """
        text_df = self.dataset["Pattern"]
        # save the "Pattern" column into text_df
        ###############################
        num_rows = len(text_df)
        # get num rows in text_df
        punctuation = ",!#$%&()*+-./\:;<=>?@[]^_`{|}~'"
        # Saving a list of commonly used punctuation to a string in order to remove
        # them from the text columns
        stop_words = stopwords.words("english")
        # getting list of common stopwords to remove from the text entries
        stop_words_removed = []
        # initializing empty list of stop words being removed
        for row in range(num_rows):
            text = text_df.iloc[row].lower()
            # getting the text to process from the row of text_df
            expanded_words = []
            # creating an empty list for contractions
            for word in text.split():
                # using contractions.fix to expand the shortened words
                expanded_words.append(contractions.fix(word))
            expanded_text = ' '.join(expanded_words)
            # expanding contractions
            new_text = expanded_text.lower()
            # make lower case text
            for char in new_text:
                if char in punctuation:
                    new_text = new_text.replace(char, " ")
            # get rid of punctuation
            for word in new_text.split():
                if word in stop_words:
                    # see if a word in new_text is in the list of stop_words
                    stop_words_removed.append(word)
                    # adding to list of stop words removed if its in the list
        stop_words_removed_df = pd.DataFrame(stop_words_removed, columns=['stop words'])
        # Turning list into a dataframe so it can be plotted
        plt.figure(figsize=(9, 6))
        # Needs to be set ahead of time before specifying the plot
        #TODO: Return the 10 value back to 500 <- FIXED 5-19-22 E_MILLER - SHOWS TOP 25 NOW
        STOP_WORDS_COUNT = stop_words_removed_df['stop words'].value_counts().nlargest(25).plot(kind='barh').invert_yaxis()
        # creating a horizontal bar plot from the stop words frequency
        # NOTE .nlargest(25) added here to constrain the plot to show
        # only the 25 highest bars in the list

        plt.xticks(rotation=0)
        # An extra part of the figure size above
        plt.title("Bar Plot Showing Frequency of Stop Words Removed (Top 25)")
        # title of the plot
        plt.show()
        # show the plot

    def Tags_Plotter(self):
        # Makes a plot of the tags from the basic Q&A files
        # Showing horizontal bar charts of the number of occurrences of tags
        plt.figure(figsize=(9, 6))
        # Needs to be set ahead of time before specifying the plot
        TAGS = self.dataset["Tag"].value_counts().plot(kind='barh').invert_yaxis()
        plt.xticks(rotation=0)
        # An extra part of the figure size above
        plt.title("Bar Plot Showing Frequency of Tags")
        plt.show()
        # Show the plot



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
