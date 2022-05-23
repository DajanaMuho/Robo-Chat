import numpy as np
import pandas as pd
import colorama
import tensorflow_hub as hub
colorama.init()
from colorama import Fore, Style
import Universal_Sentence_Encoder
from sklearn.metrics.pairwise import cosine_similarity


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# -----CONSTANTS-----#
# List of words to end the chatbot conversation
QUIT = list(["stop", "quit", "end", "Stop", "Quit", "End", "STOP", "QUIT", "END"])
SMALL_TALK_TAG = list(['greeting', 'greeting_question_how', 'chatbot_question_who', 'chatbot_question_chatboot',
                       'greeting_response_good', 'greeting_response_okay', 'greeting_response_bad', 'no_yes',
                       'yes_no', 'Unknown', 'closing', 'ending'])
path = "WikiQA.csv"


# -----METHODS-----#
def is_input_a_question(df):
    # checks the tags of the best_response data frame to see if it's a question to look up or  small talk
    for index, row in df.iterrows():
        TAG = row["Tag"]
        if TAG in SMALL_TALK_TAG:
            # if loop finds matching tag corresponds with small talk tags return False since not a question
            return False
    else:
        # if matching tag not in small talk return False since not a question
        return True


def QUESTION_SEARCH(user_input, df):
    # takes in a user response and a dataset,
    # gets the best responses and asks the user if that's what they were looking for

    # E Miller - Added section starts here
    #  plot cosine-similarity_plot

    embedded_input = embed([user_input])
    # change user input_query into a vector
    embedded_questions = embed(df["Pattern"].values)
    # embed the question data
    similarity = cosine_similarity(embedded_input, embedded_questions)
    similarity = np.transpose(similarity)

    response_df = df["Responses"].copy()

    Universal_Sentence_Encoder.similarity_plot(similarity, user_input, response_df)

    # E Miller - Added section stops here

    BEST = Universal_Sentence_Encoder.get_closest_answers(user_input, df)
    # gets the dataframe of the five closest answers
    BEST_ANSWER = BEST.iloc[0]["Responses"]
    # Gets the first best answer
    print(Fore.GREEN + "ROBO-CHAT:" + Style.RESET_ALL, BEST_ANSWER + '\n' + "Was that you were looking for? ")
    # prints best answer from get closest_answers function
    confirmation_words = list(["yes", "y", "yep", "Yes", "YES", "yeah", "yep", "Yeah", "YEAH", "YEP", "Y"])
    # list of words to break the loop below
    for i in range(5):
        print(Fore.YELLOW + "HUMAN: " + Style.RESET_ALL, end="")
        yes_or_no = input("")
        # Looking for a yes or no from the user
        if (yes_or_no not in confirmation_words) and (i == 4):
            print(Fore.GREEN + "ROBO-CHAT:" + Style.RESET_ALL, "Sorry, I couldn't find the answer you were looking for! ")
            break
            # leaves the loop if it reaches the fifth answer and the user still says that it wasn't the answer they
            # were looking for
        if yes_or_no in confirmation_words:
            print(Fore.GREEN + "ROBO-CHAT:" + Style.RESET_ALL, "Happy I could help you. Do you have another question? ")
            break  # leaves the loop if user says yes
        else:
            print(Fore.GREEN + "ROBO-CHAT:" + Style.RESET_ALL, "Here is another answer I've found:")
            print(BEST.iloc[i + 1]["Responses"] + '\n' + "Was that you were looking for? ")


def RUN_CHATBOT(embedded_training_data, dataset):
    print('START CHATTING WITH THE ROBO-CHAT, \n\nTYPE any of the words to stop the conversation:\n', QUIT, '\n')
    print(Fore.GREEN + "ROBO-CHAT: " + Style.RESET_ALL, "Hello from ROBO-CHAT! How can I help you ?")
    while True:
        print(Fore.YELLOW + "HUMAN: " + Style.RESET_ALL, end="")
        message = input("")
        if message in QUIT:
            print(Fore.RED + "Ending the chat\n " + Style.RESET_ALL, end="")
            break

        answers = Universal_Sentence_Encoder.get_closest_answers(message, dataset)

        if is_input_a_question(answers):
            QUESTION_SEARCH(message, dataset)
        else:
            response = Universal_Sentence_Encoder.get_response(dataset, embedded_training_data, message)
            print(Fore.GREEN + "ROBO-CHAT:" + Style.RESET_ALL, response)


# ----- ROBO-CHAT IMPLEMENTATION -----#

dataset = pd.read_csv(path, sep='\t')  # Read the data

print('Embedding training data...\n')
training_data = Universal_Sentence_Encoder.embed_training_data(dataset)

RUN_CHATBOT(training_data, dataset)
