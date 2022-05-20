import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

import tensorflow_hub as hub


def get_closest_answer(user_input, df):
    # Gets the closest answer using cosine similiarity

    embedded_input = embed([user_input])

    # change user input into a vector

    similarity_array = np.zeros((len(df)), dtype=float)

    # initializes array of similarities

    for index, row in df.iterrows():
        test_comparison = row["Pattern"]

        # checking a question in the dataset

        # print(test_comparison)

        embedded_question = embed([test_comparison])

        # vectorizing that question

        similarity = cosine_similarity(embedded_input, embedded_question)

        # seeing how similar the the input and the question are using
        # cosine_similarity

        similarity_array[index] = similarity

        # saving it into the array

    similarity_df = pd.DataFrame(similarity_array)

    MAX_ROW = similarity_df.idxmax()

    #print("MAX_ROW = ", MAX_ROW)

    # Find the index of the max value.

    best_response = df.iat[MAX_ROW][2]

    # TODO: Access sentence from the dataframe; right now causing errors

    print(best_response)

    return best_response


