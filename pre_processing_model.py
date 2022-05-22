import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
from nltk.stem import WordNetLemmatizer
import contractions
import matplotlib.pyplot as plt

nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()


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
            tokens = nltk.word_tokenize(pattern)
            self.words.extend(tokens)
            self.doc_X.append(pattern)
            self.doc_y.append(row["Tag"])
            if row["Tag"] not in self.classes:
                self.classes.append(row["Tag"])

        # Lemmatize the words
        self.words = [lemmatizer.lemmatize(word.lower()) for word in self.words if word not in string.punctuation]
        # sort the set to remove duplicated words
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))

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
        # TODO: Return the 10 value back to 500 <- FIXED 5-19-22 E_MILLER - SHOWS TOP 25 NOW
        STOP_WORDS_COUNT = stop_words_removed_df['stop words'].value_counts().nlargest(25).plot(
            kind='barh').invert_yaxis()
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

        # E.Miller - May 22 - changing Q tags to "question"

        for index, row in self.dataset.iterrows():

            string1 = row["Tag"]

            First_char = ord(string1[0])

            # ASCII TABLE FOR 'Q' is 81 - check to see if this is a Q

            Second_char = ord(string1[1])

            # ASCII TABLE FOR DIGITS 48 - 57 - check to see if this is a digit

            if First_char == 81 and Second_char > 47 and Second_char < 58:
                row["Tag"] = "question"
                # changes to question tag

        # Now all tags with format Q and number show question

        # E.Miller - May 22 - changing Q tags to "question"

        # Showing horizontal bar charts of the number of occurrences of tags
        plt.figure(figsize=(9, 6))
        # Needs to be set ahead of time before specifying the plot
        TAGS = self.dataset["Tag"].value_counts().nlargest(25).plot(kind='barh').invert_yaxis()
        plt.xticks(rotation=0)
        # An extra part of the figure size above
        plt.xscale('symlog')
        # log scale since question is so large
        plt.title("Bar Plot Showing Frequency of Tags")
        plt.show()
        # Show the plot
