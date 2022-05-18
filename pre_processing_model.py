import nltk
from nltk.stem import WordNetLemmatizer
import string
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")
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

    #EDWARDS CODE TO MERGE CSVs
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

    #EDWARD PREPROCESSING CODE - CAUSES ERRORS, PLEASE CHECK

    def text_preprocessor(text_df):
        # this function takes in a data frame of text only columns
        # and goes through each column and processes the text data
        # into a format that is more easily analyzed for prediction
        # It then returns the converted text data frame
        
        num_rows = len(text_df)
        num_cols = len(text_df.columns)
        
        # getting number of rows and columns in data frame
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

