#!/usr/bin/env python3

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
        self.model = self.load_language_model()
        self.rate = 16000 
        self.rec = KaldiRecognizer(self.model, self.rate)
        self.mic = pyaudio.PyAudio()
        self.voice_data = String()

        # Create ROS message
        self.utterance_msg = Utterance()

        # Publish textual representation of the spoken utterance
        self.pub_speech = rospy.Publisher('/recognition_result', Utterance, queue_size=10)



    def load_language_model(self):

        '''

        Function loading the vosk model associated to the language selected by the user

        Args: -

        Return:
            self.model (Model): The Vosk model associated to the desired language

        '''

        # TODO: change to english big model

        if self.lang == 'english':            # Loading Vosk english model
            self.model = Model("./vosk-models/english/vosk-model-small-en-us-0.15")

        elif self.lang == 'french':           # Loading Vosk french model
            self.model = Model("./vosk-models/french/vosk-model-small-fr-0.22")

        elif self.lang == 'german':           # Loading Vosk german model
            self.model = Model("./vosk-models/german/vosk-model-small-de-0.15")

        elif self.lang == 'italian':          # Loading Vosk italian model
            self.model = Model("./vosk-models/italian/vosk-model-it-0.22")

        elif self.lang == 'spanish':          # Loading Vosk spanish model
            self.model = Model("./vosk-models/spanish/vosk-model-small-es-0.42")

        return self.model


    # Function taken from: https://github.com/YannickJadoul/Parselmouth
    def draw_spectrogram(self, spectrogram, dynamic_range=70):

        '''

        Helper function to draw a spectogram

        '''

        X, Y = spectrogram.x_grid(), spectrogram.y_grid()
        sg_db = 10 * np.log10(spectrogram.values)
        plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='inferno') #cmap='afmhot')
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
        plt.axis('off')
        #plt.xlabel("time [s]")
        #plt.ylabel("frequency [Hz]")


    # Fucntion taken from: https://github.com/YannickJadoul/Parselmouth
    def draw_pitch(self, pitch):

        '''

        Helper function to extract selected pitch contour and replace unvoiced samples by NaN to not plot them
        
        '''

        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values==0] = np.nan
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
        plt.grid(False)
        plt.ylim(0, pitch.ceiling)
        #plt.ylabel("fundamental frequency [Hz]")
        plt.axis('off')


    # Function taken from: https://github.com/YannickJadoul/Parselmouth
    def create_spectogram(self, data, sent_id):

        '''

        Function that generates the spectogram and saves it

        '''

        snd = parselmouth.Sound(data)
        pitch = snd.to_pitch()
        # If desired, pre-emphasize the sound fragment before calculating the spectrogram
        pre_emphasized_snd = snd.copy()
        pre_emphasized_snd.pre_emphasize()
        spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
        plt.figure()
        self.draw_spectrogram(spectrogram)
        plt.twinx()
        self.draw_pitch(pitch)
        plt.xlim([snd.xmin, snd.xmax])
        name = "spectogram_" + str(sent_id) + ".jpg"
        spect_path = os.path.join(self.path, name)
        plt.savefig(spect_path)




    def convert_speech_to_text(self):

        '''

        Acquires user's vocal utterances and converts them into their textual representations (sentences)

        '''
        
        # Start recording the vocal utterances
        stream = self.mic.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=8192) 
        stream.start_stream()
    
        data_list = []
        sent_id = 0

        # While the node is not shutted down
        while not rospy.is_shutdown():

            # Take current time
            #start = time.process_time()

            data = stream.read(1024) #4096
            data_list.append(data)

            # Check if the recognizer is acquiring vocal data
            if self.rec.AcceptWaveform(data):

                # Output of the vocal transcription
                result = self.rec.Result()

                # Extract only the textual data
                self.voice_data = result[14:-3]

                # Process the current vocal utterance
                if self.voice_data != "" and self.voice_data != "huh":
                    adj_voice_data = self.voice_data + "."
                    self.voice_data = adj_voice_data.capitalize()
                    print(self.voice_data)


                    '''print("------------ TIME INITIAL INTERACTION -----------")
                    print(datetime.now())
                    print("---------------------------------------------")'''
                    

                    # -------------------------
                    # datetime object containing current date and time
                    #now = datetime.now()

                    # dd/mm/YY H:M:S
                    #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    #print("date and time =", dt_string)

                    # -------------------------

                    sample_width = pyaudio.PyAudio.get_sample_size(self,format=pyaudio.paInt16)
                    sent_id += 1

                    # Save the wav file 
                    wav_filename = "audio_sentence_" + str(sent_id) + ".wav"
                    wav_path = os.path.join(self.path, wav_filename)


                    # Open the wav file and save the recording
                    wf = wave.open(wav_path, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(self.rate)
                    wf.writeframes(b''.join(data_list))
                    wf.close()

                    #TODO: improve the spectrogram extraction
                    self.create_spectogram(wav_path, sent_id)

                    # Publish a message containing the user_ID, sentence and language; speech_act field is empty
                    self.utterance_msg.sent_id = str(sent_id)
                    self.utterance_msg.language = self.lang
                    self.utterance_msg.sentence = self.voice_data
                    self.utterance_msg.speech_act = ""
                    self.utterance_msg.frame_roles = [Frame()]
                    self.utterance_msg.frame_words = [Frame()]
                    self.pub_speech.publish(self.utterance_msg)

                    '''print("***********************")
                    print('Time required for sentence recognition is:')
                    print(datetime.now() - now)
                    print("***********************")'''

                    print('Published voice message')

                data_list = []

        # Stop recording the vocal utterance
        stream.stop_stream()
        stream.close()
        self.mic.terminate()





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

