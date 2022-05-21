import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def embed_training_data(df):
    return embed(df["Pattern"].values)


def embed_input(user_input):
    return embed([user_input])


def get_cos_similarity(embedded_training_data, input_query):
    embedded_input = embed_input(input_query)
    return cosine_similarity(embedded_training_data, embedded_input)


def get_response(dataset, embedded_training_data, input_query):
    cos_similarity = get_cos_similarity(embedded_training_data, input_query)
    similarity_df = pd.DataFrame(cos_similarity)
    MAX_ROW = similarity_df.idxmax()
    MAX_ROW = int(MAX_ROW + 0)
    return dataset.loc[MAX_ROW].at["Responses"]


# TO BE Removed as isn't being called , keeping it for reference
def get_closest_answer(user_input, df):
    # Gets the closest answer using cosine similiarity

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

        # seeing how similar the the input_query and the question are using
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

    # get best response from row corresponding to the
    # most similarity

    return best_response
