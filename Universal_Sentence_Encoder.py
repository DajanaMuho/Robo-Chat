import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# Getting the universal sentence encoder for use in the chatbot

def embed_training_data(df):
    # embeds the training data into vectors ahead of time
    # to make searches faster
    return embed(df["Pattern"].values)


def embed_input(user_input):
    # change user input_query into a vector
    return embed([user_input])


def get_cos_similarity(embedded_training_data, input_query):
    embedded_input = embed_input(input_query)
    # seeing how similar the input_query and the question are using
    # cosine_similarity
    return cosine_similarity(embedded_training_data, embedded_input)


def get_response(dataset, embedded_training_data, input_query):
    cos_similarity = get_cos_similarity(embedded_training_data, input_query)
    similarity_plot(cos_similarity, input_query, dataset['Responses'])
    # gets all similarities between the user input and the Q&A dataset
    similarity_df = pd.DataFrame(cos_similarity)
    # making into a data frame in order to be able to search easier
    MAX_ROW = similarity_df.idxmax()
    MAX_ROW = int(MAX_ROW + 0)
    # getting the row number where the largest similarity is
    # and sending back the matching response from that row
    return dataset.loc[MAX_ROW].at["Responses"]

# E_MILLER May 21, 2022 - 5:17pm
# Adding faster get responses function with more information returned
# Adding function to check whether the user is asking a Wiki question
# or making small talk

def get_closest_answers(user_input, df):
    # Gets a data frame of the
    # closest answers using cosine similarity

    embedded_input = embed([user_input])

    # change user input into a vector

    embedded_questions = embed(df["Pattern"].values)

    # embed the question data

    similarity = cosine_similarity(embedded_input, embedded_questions)

    # get an array of cosine similarities

    similarity_df = pd.DataFrame(np.transpose(similarity), columns=["Similarity"])

    # Make array into a dataframe to make it easier to work with

    similarity_df_sorted = similarity_df.sort_values(by="Similarity", ascending=False)

    Most_Similar = similarity_df_sorted[:5]

    # getting the rows with the 5 highest similarities

    Best_Response_List = df["Responses"].copy()

    Best_Response_List = pd.DataFrame(Best_Response_List, index=Most_Similar.index)

    # Getting the responses fitting with the rows most similar

    Tag_List = df["Tag"].copy()

    Tag_List = pd.DataFrame(Tag_List, index=Most_Similar.index)

    # Getting the tags associated with the responses

    RESPONSE_DF = pd.concat([Most_Similar, Best_Response_List, Tag_List], axis=1, join='outer')

    # putting the information together into one data frame to return
    # has three columns - similarity, response and tag

    return RESPONSE_DF

def is_input_a_question(df):
    # checks the tags of the best_response data frame
    # to see if it's a question to look up or
    # small talk

    basic_url = "https://raw.githubusercontent.com/DajanaMuho/Robo-Chat/main/Q%26A.csv"

    DATA_BASIC = pd.read_csv(basic_url)

    small_talk_tags = DATA_BASIC["Tag"].tolist()

    # Getting a list of small talk tags to check

    for index, row in df.iterrows():

        TAG = row["Tag"]

        if TAG in small_talk_tags:
            # if loop finds matching tag corresponds
            # with small talk tags
            # return False since not a question

            return False

    else:

        # if matching tag not in small talk
        # return False since not a question

        return True


def similarity_plot(similarity, input, responses):
    similarity_df = pd.DataFrame(similarity, columns=["Similarity"])
    # Make array into a dataframe to make it easier to work with
    similarity_df_sorted = similarity_df.sort_values(by="Similarity", ascending=False)

    Ten_Most_Similar = similarity_df_sorted[:10]

    Best_Responses_List = responses.copy()

    Best_Responses_List = pd.DataFrame(Best_Responses_List, index=Ten_Most_Similar.index)

    SIMILAR_DF = pd.concat([Best_Responses_List, Ten_Most_Similar], axis=1, join='outer')

    # put together a full dataframe to use for the plot

    plt.figure(figsize=(15, 10))

    # Needs to be set ahead of time before specifying the plot

    SIMILAR_DF.plot(x="Responses", y="Similarity", kind='barh').invert_yaxis()

    plt.xticks(rotation=0)

    # An extra part of the figure size above

    plt.title("Using Cosine Similarity to find most likely Responses to " + "'" + input + "'")

    plt.show()

    # Show the plot

def QUESTION_SEARCH(user_input, DATA_FULL):
    # takes in a user response and a dataset, gets the best responses
    # and asks the user if that's what they were looking for

    BEST = get_closest_answers(user_input, DATA_FULL)

    # gets the dataframe of the five closest answers

    BEST_ANSWER = BEST.iloc[0]["Responses"]

    # Gets the first best answer

    print(BEST_ANSWER)

    # prints best answer from get closest_answers function

    confirmation_words = list(["yes", "y", "yep", "Yes", "YES", "yeah", "yep", "Yeah", "YEAH", "YEP", "Y"])

    # list of words to break the loop below

    for i in range(5):

        yes_or_no = input("Was that you were looking for? ")

        # Looking for a yes or no from the user

        if (yes_or_no not in confirmation_words) and (i == 4):
            print("Sorry, I couldn't find the answer you were looking for")
            break

            # leaves the loop if it reaches the fifth answer and the user
            # still says that it wasn't the answer they were looking for

        if yes_or_no in confirmation_words:

            break

            # leaves the loop if user says yes

        else:

            print("Here is another answer I've found:")
            print(BEST.iloc[i + 1]["Responses"])

        # Gives the next answer in the data frame