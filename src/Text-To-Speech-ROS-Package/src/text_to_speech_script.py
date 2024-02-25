#!/usr/bin/env python3

import os
import time
from datetime import datetime
import rospy
from picotts import PicoTTS
from playsound import playsound
from pathlib import Path

from text_to_speech.msg import TTSMessage


"""

    FOLDERS

"""

sound_assets_dir = os.path.join(os.getenv('HOME'), "audios/")
generated_sounds_dir = os.path.join(sound_assets_dir, "generated/")
recorded_sounds_dir = os.path.join(sound_assets_dir, "recorded/")


"""

    Pico TTS

"""

supported_languages = {
    "it": "it-IT",
    "en": "en-GB",
    "fr": "fr-FR",
    "es": "es-ES",
    "de": "de-DE"
}
"""
Language initials to Pico TTS supported languages
"""


"""

    TTS

"""


class TextToSpeech:
    def __init__(self):
        # Initialize the ROS node
        self.node = rospy.init_node('text_to_speech', anonymous=True)

        # Subscribe to the text topic
        self.sub = rospy.Subscriber('text_to_speech_topic', TTSMessage, self.generate_speech)

        # Spin ROS
        rospy.spin()


    def generate_speech(self, data: TTSMessage):
        """Say the given sentence or play the given audio

        Args:
            data (TTSRequest): Request to say or play
        """

        # Check if the provided language is supported
        print(data.lang)
        if data.lang not in supported_languages:
            #rospy.logerr("Language not supported")
            print("ERROR: Language not supported")
            return
        
        # Check if 'audios' folder exists
        if not os.path.exists(sound_assets_dir):
            # Create 'audios' folder
            audios_fold = os.mkdir(sound_assets_dir)
            print("Directory 'audios' created")

        for sentence in data.sentences:

            # Say requested text
            if data.type == "say":
                #rospy.loginfo(f"Saying \"{sentence}\" ({data.lang})")
                print(f"Saying \"{sentence}\" ({data.lang})")

                # Initialize the PicoTTS speech engine
                picotts = PicoTTS()

                # Changes languages regarding the input parameter
                picotts.voice = supported_languages[data.lang]

                # Transform the text to speech
                wav = picotts.synth_wav(sentence)

                # Get current date time
                date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # Check if 'generated' folder exists
                if not os.path.exists(generated_sounds_dir):
                    # Create 'generated' folder
                    generated_fold = os.mkdir(generated_sounds_dir)
                    print("Directory 'generated' created")

                # Folder path
                folder_path = os.path.join(generated_sounds_dir, data.lang)

                # Check if language folder exists
                if not os.path.exists(folder_path):
                    # Create language folder
                    generated_fold = os.mkdir(folder_path)
                    print("Created language directory")

                # Absolute path of the output file
                output_file_path = os.path.join(folder_path, f"{date_time}.wav")

                # Open/create the wav file
                f = open(output_file_path, "wb")

                # Write the content to it
                f.write(wav)

                # Play the sound
                playsound(output_file_path)

                # Close the file
                f.close()

                # Clear the oldest file
                self.clear_older_files(folder_path)
                
            # Play a recorded sound
            elif data.type == "play":
                #rospy.loginfo(f"Playing \"{sentence}\" ({data.lang})")
                print(f"Playing \"{sentence}\" ({data.lang})")

                # Check if 'recorded' folder exists
                if not os.path.exists(recorded_sounds_dir):
                    # Create 'recorded' folder'
                    recorded_fold = os.mkdir(recorded_sounds_dir)
                    print("Directory 'recorded' created")

                # Folder path
                folder_path = os.path.join(generated_sounds_dir, data.lang)

                # Check if language folder exists
                if not os.path.exists(folder_path):
                    # Create language folder
                    generated_fold = os.mkdir(folder_path)
                    print("Directory '% s' created" % folder_path)

                # Get the file path
                file_path = os.path.join(recorded_sounds_dir, data.lang, f"{sentence}.wav")

                # If the files doesn't exist
                if not os.path.exists(file_path):
                    #rospy.logerr(f"File {file_path} doesn't exist")
                    print(f"ERROR: File {file_path} doesn't exist")
                # Otherwise
                else:
                    # Play the audio
                    playsound(file_path)
                            
            # If another command type was provided
            else:
                #rospy.logerr("Command type not supported")
                print("ERROR: Command type not supported")
                return




    def clear_older_files(self, folder_path: str):
        """Clear the oldest file above the file limit in the given folder

        Args:
            folder_path (str): Folder path to clear the oldest file
        """

        # Return if the path doesn't exist
        if not os.path.exists(folder_path):
            return

        # Get the files in the folder
        files = os.listdir(folder_path)

        # If there is more than 100 files in the folder
        if len(files) >= 100:
            # Sort the files and get the first one (oldest)
            oldest_file = sorted(files)[0]
            
            # Get the oldest file's path
            oldest_file_path = os.path.join(folder_path, oldest_file)

            #rospy.loginfo(f"Too much generated files, deleting the oldest one:\n{oldest_file_path}")
            print(f"Too much generated files, deleting the oldest one:\n{oldest_file_path}")

            # Remove the file
            os.remove(oldest_file_path)


"""

    MAIN

"""

if __name__ == '__main__':
    TextToSpeech()