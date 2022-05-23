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
    # gets all similarities between the user input_query and the Q&A dataset
    similarity_df = pd.DataFrame(cos_similarity)
    # making into a data frame in order to be able to search easier
    MAX_ROW = similarity_df.idxmax()
    MAX_ROW = int(MAX_ROW + 0)
    # getting the row number where the largest similarity is
    # and sending back the matching response from that row
    return dataset.loc[MAX_ROW].at["Responses"]


def get_closest_answers(user_input, df):
    # Gets a data frame of the closest answers using cosine similarity
    embedded_input = embed([user_input])
    # change user input_query into a vector
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
    # putting the information together into one data frame to return, has three columns - similarity, response and tag
    return RESPONSE_DF


def similarity_plot(similarity, input_query, responses):
    similarity_df = pd.DataFrame(similarity, columns=["Similarity"])
    # Make array into a dataframe to make it easier to work with
    similarity_df_sorted = similarity_df.sort_values(by="Similarity", ascending=False)
    Ten_Most_Similar = similarity_df_sorted[:10]
    Best_Responses_List = responses.copy()
    # get the responses into its own data frame
    Best_Responses_List = Best_Responses_List.str.slice(0, 50)
    # Shorten the string to 50 chars to fit into the x labels
    Best_Responses_List = pd.DataFrame(Best_Responses_List, index=Ten_Most_Similar.index)
    SIMILAR_DF = pd.concat([Best_Responses_List, Ten_Most_Similar], axis=1, join='outer')
    # put together a full dataframe to use for the plot
    plt.figure(figsize=(15, 10))
    # Needs to be set ahead of time before specifying the plot
    SIMILAR_DF.plot(x="Responses", y="Similarity", kind='barh').invert_yaxis()
    plt.xticks(rotation=0)
    # An extra part of the figure size above
    plt.title("Using Cosine Similarity to find most likely responses to " + "'" + input_query + "'")
    plt.tight_layout()
    plt.show()
