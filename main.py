import pandas as pd
import colorama
colorama.init()
from colorama import Fore, Style
import Universal_Sentence_Encoder

# -----CONSTANTS-----#
# List of words to end the chatbot conversation
QUIT = list(["stop", "quit", "end", "Stop", "Quit", "End", "STOP", "QUIT", "END"])
path = "WikiQA.csv"


# -----METHODS-----#
def RUN_CHATBOT(embedded_training_data, dataset):
    print('START CHATTING WITH THE ROBO-CHAT, \n\nTYPE any of the words to stop the conversation:\n', QUIT, '\n')
    print(Fore.GREEN + "ROBO-CHAT:" + Style.RESET_ALL, "Hello from ROBO-CHAT! How can I help you ?")
    while True:
        print(Fore.YELLOW + "HUMAN:" + Style.RESET_ALL, end="")
        message = input("")
        if message in QUIT:
            print(Fore.RED + "Ending the chat\n " + Style.RESET_ALL, end="")
            break
        # response = Universal_Sentence_Encoder.get_closest_answer(message, dataset)  # WORKS, but takes time to reply
        response = Universal_Sentence_Encoder.get_response(dataset, embedded_training_data, message)
        print(Fore.GREEN + "ROBO-CHAT:" + Style.RESET_ALL, response)


# ----- ROBO-CHAT IMPLEMENTATION -----#

dataset = pd.read_csv(path, sep='\t')  # Read the data

print('Embedding training data...\n')
training_data = Universal_Sentence_Encoder.embed_training_data(dataset)

RUN_CHATBOT(training_data, dataset)
