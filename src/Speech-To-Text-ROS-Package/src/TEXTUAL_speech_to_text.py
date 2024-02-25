#!/home/Sara/sara_cpu_venv/bin/python

import pyaudio
import wave
import openpyxl
import os
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vosk import Model, KaldiRecognizer

from std_msgs.msg import String
from natural_language_understanding.msg import Utterance, Frame
from datetime import datetime

# For Spectogram Analysis
import parselmouth

# Use seaborn's default style to make attractive graphs
sns.set()



# TODO: introduce language detector



def get_language_id():

    '''

    Asks the user to first provide a value that will be used as an ID, then to enter the value associated 
    with the prefered language in which the user wants to interact with

    Args: -

    Return: 
        dict_lang (dict): Dictionary with value-language association -> {0: 'english', 1: 'french', 2: 'german', 3: 'italian', 4: 'spanish'} 
        value (int): Value associated to the language chosen by the user
        personID (float): The ID provided by the user

    '''

    # TODO: automatically asign an ID to the user
    
    print("Provide an ID for the user")

    # The user provides the value that will be the user ID
    ID = float(input())
    personID = str(ID)

    print('In which language do you want to interact with the robot? \n')
    langs = ['english', 'french', 'german', 'italian', 'spanish']

    print('Write the number associated to the language you want to speak, by considering the following supported languages:')

    # Value-language dictionary creation 
    dict_lang = {}

    # Ordering the languages in alphabetical order
    sort_langs = sorted(langs)

    # Filling the dictionary with value-language correspondance
    for i in range(len(sort_langs)):
        dict_lang[i] = sort_langs[i]
    print(dict_lang)

    # Variabile for tracking the user's language selection
    not_select_lang = True

    # Until the user selects a valid value for the language
    while not_select_lang:
        value = int(input())

        # Check if the value entered by the user is in the rage of supported languages, so between 0 and 4 (included)
        if value > -1 and value < len(sort_langs):
            print('\nGreat! You chose to interact in '+ dict_lang[value] +', is it right?')
            print('(YES to continue with the selected language, NO to choose another language)')
            
            # The user confirms his/her language choice
            confirm = str(input())
            if confirm == "YES" or confirm == "Yes" or confirm == "yes":
                not_select_lang = False
            else:
                print("Select the language in which you want to interact")
                print(dict_lang)
        else:
            print("The inserted value is not valid, select a value between 0 and " + str(len(sort_langs)-1) + " (included)")

    return dict_lang, value, personID



def create_user_folder(personID):

    '''

    Create user's folder and files to save the vocal utterances and spectograms

    Args: 
        personID (float): The ID provided by the user

    Return: 
        path (string): The path to the user's directory

    '''

    # Parent Directory path
    parent_dir = "./"
    folder = "User Interactions"
    path_users = os.path.join(parent_dir, folder)

    # Check if the main folder containing users' interactions exists
    if not os.path.exists(path_users):
        os.mkdir(path_users)
        print("Folder '% s' created" % folder)

    directory = "Person_" + personID
    excel_file = "Utterances_"+ personID +".xlsx"

    wb = openpyxl.Workbook()

    # Path
    path = os.path.join(path_users, directory)

    # Check if the user's directory exists
    if not os.path.exists(path):

        # Create the user's directory
        os.mkdir(path)
        print("Directory '% s' created" % directory)

        out_path = os.path.join(path, excel_file)
        wb.save(out_path)
        print("Excel saved correctly!")

    return path



"""

    SPEECH RECOGNITION

"""

class SpeechRecognizer:

    def __init__(self, lang, ID, path):

        # Variables
        self.lang = lang
        self.ID = ID
        self.path = path
        self.voice_data = String()

        # Create ROS message
        self.utterance_msg = Utterance()

        # Publish textual representation of the spoken utterance
        self.pub_speech = rospy.Publisher('/recognition_result', Utterance, queue_size=10)



    def convert_speech_to_text(self):

        '''

        Acquires user's vocal utterances and converts them into their textual representations (sentences)

        '''
    
        #data_list = []
        sent_id = 0

        # While the node is not shutted down
        while not rospy.is_shutdown():

                # Extract only the textual data
                print("\nProvide the textual sentence:")
                self.voice_data = str(input())

                if self.voice_data == "exit":
                    break

                # Process the current vocal utterance
                if self.voice_data != "" and self.voice_data != "exit":
                    adj_voice_data = self.voice_data + "."
                    self.voice_data = adj_voice_data.capitalize()
                    print("\nAdjusted sentence to be sent to the NLU module:")
                    print(self.voice_data)

                    # Publish a message containing the user_ID, sentence and language; speech_act field is empty
                    self.utterance_msg.sent_id = str(sent_id)
                    self.utterance_msg.language = self.lang
                    self.utterance_msg.sentence = self.voice_data
                    self.utterance_msg.speech_act = ""
                    self.utterance_msg.frame_roles = [Frame()]
                    self.utterance_msg.frame_words = [Frame()]
                    self.pub_speech.publish(self.utterance_msg)


                    print('Published voice message')
                    print('---------------------')







if __name__ == '__main__':
    #rospy.init_node('speech_rec', anonymous=True)
    rospy.init_node('speech_rec', anonymous=True)

    # Assign an id to the user and ask the 
    dict_lang, value, personID = get_language_id()

    # Create user directories/files to collect vocal utterances
    path = create_user_folder(personID)

    # Initialize an instance of Speech Recognition for the specific language
    speech = SpeechRecognizer(dict_lang[value], personID, path)

    # Start the speech to text recognition
    speech.convert_speech_to_text()

