import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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
    # gets all similarities between the user input and the Q&A dataset
    similarity_df = pd.DataFrame(cos_similarity)
    # making into a data frame in order to be able to search easier
    MAX_ROW = similarity_df.idxmax()
    MAX_ROW = int(MAX_ROW + 0)
    # getting the row number where the largest similarity is
    # and sending back the matching response from that row
    return dataset.loc[MAX_ROW].at["Responses"]


# TO BE Removed as isn't being called , keeping it for reference
def get_closest_answer(user_input, df):
    # Gets the closest answer using cosine similarity

    embedded_input = embed([user_input])

    # change user input_query into a vector

    similarity_array = np.zeros((len(df)), dtype=float)

    # initializes array of similarities

    for index, row in df.iterrows():
        test_comparison = row["Pattern"]

        # checking a question in the dataset

        # print(test_comparison)

        embedded_question = embed([test_comparison])

        # vectorizing that question

        similarity = cosine_similarity(embedded_input, embedded_question)

        # seeing how similar the input_query and the question are using
        # cosine_similarity

        similarity_array[index] = similarity

        # saving it into the array

    similarity_df = pd.DataFrame(similarity_array)

    # Make array into a dataframe to make it easier to work with

    MAX_ROW = similarity_df.idxmax()

    # Find the index of the max value.

    MAX_ROW = int(MAX_ROW + 0)

    # mske sure MAX_ROW is an integer

    best_response = df.loc[MAX_ROW].at["Responses"]

    # get the best response from row corresponding to the
    # most similarity

    return best_response


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


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def similarity_plot():

    # Makes a horizontal bar charts of the most likely
    # #responses based on cosine similarity

    Q_A = pd.read_csv("https://raw.githubusercontent.com/DajanaMuho/Robo-Chat/main/WikiQA.csv", sep='\t')

    # reading the dataset

    user_input = str("Hi there, how are you doing today")

    # Gets the closest answer using cosine similiarity

    embedded_input = embed([user_input])

    # change user input into a vector

    embedded_questions = embed(Q_A["Pattern"].values)

    # embed the question data

    similarity = cosine_similarity(embedded_input, embedded_questions)

    # get an array of cosine similarities

    similarity_df = pd.DataFrame(np.transpose(similarity), columns=["Similarity"])

    # Make array into a dataframe to make it easier to work with

    similarity_df_sorted = similarity_df.sort_values(by="Similarity", ascending=False)

    Ten_Most_Similar = similarity_df_sorted[:10]

    Best_Responses_List = Q_A["Responses"].copy()

    Best_Responses_List = pd.DataFrame(Best_Responses_List, index=Ten_Most_Similar.index)

    SIMILAR_DF = pd.concat([Best_Responses_List, Ten_Most_Similar], axis=1, join='outer')

    # put together a full dataframe to use for the plot

    plt.figure(figsize=(15, 10))

    # Needs to be set ahead of time before specifying the plot

    SIMILAR_DF.plot(x="Responses", y="Similarity", kind='barh').invert_yaxis()

    plt.xticks(rotation=0)

    # An extra part of the figure size above

    plt.title \
        ("Using Cosine Similarity to find most likely Responses to 'Hi there, how are you doing today'")

    plt.show()

    # Show the plot
