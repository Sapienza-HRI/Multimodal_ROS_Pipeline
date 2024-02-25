#!/usr/bin/env python3

import rospy
import uuid
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from transformers import pipeline

from canopies_utils.log_utils import create_csv_with_headers_if_does_not_exist, write_to_log_file

from text_to_speech.msg import TTSMessage
from dialog_manager.srv import HandleFrame, HandleFrameRequest, HandleFrameResponse
from dialog_manager.srv import TTSRequest, TTSRequestRequest, TTSRequestResponse
from dialog_manager.srv import HandleSuggestion, HandleSuggestionRequest, HandleSuggestionResponse


"""

    LOGS

"""

class SpeakerType(Enum):
    ROBOT = "robot"
    HUMAN = "human"


class ConversationType(Enum):
    """
    Represent the type of conversation sentence(s)
    """
    INFORMATION = "Information"
    REQUEST = "Request"
    COMMAND = "Command"
    COMPLEX_CI = "Command + Information"
    COMPLEX_RI = "Request + Information"
    ANSWER = "Answer"
    PROBLEM = "Problem"
    ERROR = "Error"
    RESULT = "Result"
    SUGGESTION = "Suggestion"



class ConversationLog():
    """
    Represent a single sentence log
    """

    def __init__(self, speaker_type: SpeakerType, conversation_type: ConversationType, sentences: 'list[str]'):
        self.datetime = datetime.now() #.strftime("%Y/%m/%d %H:%M:%S")
        self.speaker_type = speaker_type
        self.conversation_type = conversation_type
        self.sentences = sentences


"""

    SPEAKERS

"""

supported_translations = [
    "fr",
    "es",
    "de",
    "en",
    "it"
]
"""
List of the supported translation languages
"""

language_to_initials = {
    "english": "en",
    "italian": "it",
    "french": "fr",
    "spanish": "es",
    "german": "de"
}
"""
Language name to its initials
"""

class Speaker:
    """
    Represent a single speaker
    """

    def __init__(self, first_name: str, last_name: str, language: str):
        self.first_name = first_name
        self.last_name = last_name
        self.uuid = str(uuid.uuid4())
        self.language = language_to_initials[language]
        self.conversation: list[ConversationLog] = []


"""

    MANAGER

"""

class DialogManager:
    def __init__(self):

        #time_start = datetime.now()

        # Variables
        self.default_speaker = Speaker("Human", "", "italian")
        self.current_speaker = self.default_speaker
        self.speakers = [self.default_speaker]

        # Create the default speaker conversation log file
        create_csv_with_headers_if_does_not_exist(
            f"dialog_{self.default_speaker.first_name}_{self.default_speaker.last_name}_conversation_logs.csv",
            ("datetime", "speaker_type", "conversation_type", "sentences")
        )

        # Constants
        # Get the frame definitions JSON file
        frame_definitions_file = open("src/Dialog-Manager-ROS-Package/src/hierarchy_knowledge_ONLY_frames.json", "r")
        # Load the frame definitions file to a variable 
        self.frame_definitions = json.load(frame_definitions_file)

        self.handle_function_linker = {
            "handle_introduce": self.generate_introduce_response,
            "handle_go": self.generate_go_response,
            "handle_stop": self.generate_stop_response,
            "handle_turn": self.generate_turn_response,
            "handle_check": self.generate_check_response,
            "handle_harvest": self.generate_harvest_response,
            "handle_prune": self.generate_prune_response,
            "handle_cut": self.generate_cut_response,
            "handle_block": self.generate_block_response,

            "handle_speak": self.generate_speak_response,
            "handle_do": self.generate_do_response,
            "handle_detect": self.generate_detect_response,
            "handle_be_exist": self.generate_be_exist_response,
            "handle_show": self.generate_show_response,
            "handle_remove": self.generate_remove_response,
            "handle_finish": self.generate_finish_response,
            
        }
                

        # Initialize the ROS node
        self.node = rospy.init_node('dialog_manager', anonymous=True)


        # Services
        # Dialog service
        self.dialog_service = rospy.Service('handle_frame_service', HandleFrame, self.manage_dialog)

        # Handle suggestion service
        self.handle_suggestion_service = rospy.Service('handle_suggestion_service', HandleSuggestion, self.handle_suggestion)

        # Create TTS service
        self.tts_service = rospy.Service('tts_service', TTSRequest, self.handle_tts)

        # Publishers
        self.tts = rospy.Publisher('text_to_speech_topic', TTSMessage, queue_size=10)

        # Spin ROS
        rospy.spin()


    """
    
        LOGS
    
    """

    def log_conversation(self, conversation_log: ConversationLog, speaker: Speaker = None):
        """Write the given log to the log file

        Args:
            speaker (Speaker): On which speaker file to log data
            frame_log (FrameLog): Log to write in the log file
        """
        if speaker is None:
            speaker = self.current_speaker

        write_to_log_file(
            f"dialog_{speaker.first_name}_{speaker.last_name}_conversation_logs.csv",
            (conversation_log.datetime, conversation_log.speaker_type.name, conversation_log.conversation_type.name, conversation_log.sentences)
        )


    """
    
        FRAMES

    """

    def manage_dialog(self, request: HandleFrameRequest) -> HandleFrameResponse:
        """Process and log a given frame into a result sentence, and a potential task linked to the speaker

        Args:
            request (HandleFrameRequest): Frame to process

        Returns:
            HandleFrameResponse: Result sentence, potential task linked to the speaker
        """
        #print('---------- Entrato nel manage_dialog -------------')
        request = request.frame

        # Turns the "None" and "" to actual None type by analysing frame_roles information
        frame_roles = [None if frame == "None" or frame == "" else frame for frame in request.frame_roles]
        
        # Turns the "None" and "" to actual None type by analysing frame_words information
        frame_words = [None if frame == "None" or frame == "" else frame for frame in request.frame_words]
        
        # Get the speech act
        speech_act = request.speech_act

        # Get the full speaker's sentence
        full_sentence = request.full_sentence

        # Get the speaker's language
        language = request.language

        self.current_speaker.language = language_to_initials[language]


        print("Frame roles", frame_roles)
        print("Frame words", frame_words)
        print("Speech act", speech_act)


        # Get the response and its associated log
        log, response = self.handle_frame(frame_roles, frame_words, speech_act)

        print('RESPONSE: ', response)

        # Create conversation log
        request_conversation_log = ConversationLog(SpeakerType.HUMAN, ConversationType(speech_act), [full_sentence])

        # Log the frame request into the current speaker conversation
        self.current_speaker.conversation.append(request_conversation_log)

        # Log the conversation to the log file
        self.log_conversation(request_conversation_log)

        # Log the response log into the current speaker conversation
        self.current_speaker.conversation.append(log)

        # Log the conversation to the log file
        self.log_conversation(log)

        # LOG TODO: Remove
        self.print_speaker_conversation(self.current_speaker)

        # Return the response
        return response



    def handle_frame(self, frame_roles: 'list[str, None]', frame_words:'list[str, None]', speech_act: str) -> 'tuple[ConversationLog, HandleFrameResponse]':
        """Handle a single frame and its speech act

        Args:
            frame_roles (list[str, None]): List with a frame name and its role arguments
            frame_words (list[str, None]): List with a frame name and its word arguments
            speech_act (str): The speech act associated to the frame name

        Returns:
            tuple[ConversationLog, HandleFrameResponse]: Translated response and its associated log
        """

        print("Received:", speech_act, frame_roles, frame_words)


        # ---------------------------------
        #           COMBINATION
        # ---------------------------------

        # Managing combination of different information from speech and gesture
        frame_roles_comb = []
        frame_words_comb = []

        if "|" in frame_roles:
            delim = frame_words.index("|")
            frame_roles_S = frame_roles[:delim]
            frame_roles_G = frame_roles[delim+1:]
            if frame_roles_S[0] != frame_roles_G[0] and frame_roles_S[0] == "STOP":
                return self.generate_response(
                    [f"Ho riconosciuto il frame {frame_roles_S[0]} con il parlato e mi sto fermando"],
                    ConversationType.ANSWER,
                    f"Mi sto fermando",
                    f"stop_task",
                    frame_words
                )
            elif frame_roles_S[0] != frame_roles_G[0]:
                return self.generate_response(
                    [f"Ho riconosciuto diversi frame: {frame_roles_S[0]} per il parlato e {frame_roles_G[0]} per i gesti"],
                    ConversationType.ERROR#,
                    #f"Errore",
                    #f"stop_task"#,
                    #frame_arg_words
                )
            elif len(frame_roles_S[1:]) != len(frame_roles_G[1:]):
                return self.generate_response(
                    [f"La lunghezza degli argomenti individuati attraverso il parlato ed i gesti è diversa"],
                    ConversationType.ERROR
                )
            elif set(frame_roles_S[1:]) != set(frame_roles_G[1:]):
                return self.generate_response(
                    [f"Gli argomenti individuati con il parlato e con i gesti sono diversi"],
                    ConversationType.ERROR
                )
            frame_roles_comb.append(frame_roles_S)
            frame_roles_comb.append(frame_roles_G)

        if "|" in frame_words:
            delim = frame_words.index("|")
            frame_words_S = frame_words[:delim]
            frame_words_G = frame_words[delim+1:]
            
            frame_words_comb.append(frame_words_S)
            frame_words_comb.append(frame_words_G)


        print("Trying:", speech_act, frame_roles, frame_words)

        # No multimodality
        if len(frame_roles_comb) == 0:
            # Get the frame name
            frame_name = frame_roles[0]
        else: # multimodality
            frame_name = frame_roles_comb[0][0]

        # Ensure the frame name exists
        if frame_name not in self.frame_definitions:
            return self.generate_response(
                [f"{frame_name.lower()} non è tra i frame che conosco"],
                ConversationType.ERROR
            )

        # Get the current frame case
        frame_case = self.frame_definitions[frame_name]

        print('Frame case: ',frame_case)
        '''
        --- OUTPUT ---
        [{'Arguments': ['Attribute', 'Day', 'Location', 'Location_Num', 'Location_Per', 'Location_Spec', 
        'Negation', 'Quantity', 'Specifier', 'Theme', 'Theme_Determiner', 'Time', 'Time_Spec'], 
        'Function': 'handle_harvest'}]
        '''

        frame_arguments_list = frame_case[0]["Arguments"]

        # No multimodality
        if len(frame_roles_comb) == 0:
            frame_arg_roles = ['_'] * len(frame_arguments_list)
            frame_arg_words = ['_'] * len(frame_arguments_list)

            for arg in frame_roles[1:]:
                if arg not in frame_arguments_list:
                    return self.generate_response(
                    [f"{arg} non è nella lista degli argomenti associati a {frame_name.lower()}"],
                        ConversationType.ERROR
                )


            for idx, ar in enumerate(frame_arguments_list):
                print(ar)
                if ar in frame_roles:
                    frame_arg_roles[idx] = ar
                    index = frame_roles.index(ar)
                    frame_arg_words[idx] = frame_words[index].lower()
            
            print(frame_arg_roles)
            print(frame_arg_words)

            # Get the frame handler
            handle = self.handle_function_linker[frame_case[0]["Function"]]

            print('***********')
            print(speech_act)
            print(frame_arg_roles)
            print(frame_arg_words)
            print('***********')

            # Handle the given arguments and return the response
            return handle(speech_act, frame_arg_roles, [], frame_arg_words, [])


        else: # multimodality
            frame_arg_roles_speech = ['_'] * len(frame_arguments_list)
            frame_arg_words_speech = ['_'] * len(frame_arguments_list)
            frame_arg_roles_gesture = ['_'] * len(frame_arguments_list)
            frame_arg_words_gesture = ['_'] * len(frame_arguments_list)

            for arg in frame_roles_comb[0][1:]:
                if arg not in frame_arguments_list:
                    return self.generate_response(
                    [f"{arg} non è nella lista degli argomenti associati a {frame_name.lower()}"],
                        ConversationType.ERROR
                )


            for idx, ar in enumerate(frame_arguments_list):
                print(ar)
                if ar in frame_roles_comb[0] and ar in frame_roles_comb[1]:
                    frame_arg_roles_speech[idx] = ar
                    frame_arg_roles_gesture[idx] = ar
                    index_s = frame_roles_comb[0].index(ar)
                    index_g = frame_roles_comb[1].index(ar)
                    frame_arg_words_speech[idx] = frame_words_comb[0][index_s].lower()
                    frame_arg_words_gesture[idx] = frame_words_comb[1][index_g].lower()
            
            print(frame_arg_roles_speech)
            print(frame_arg_roles_gesture)
            print(frame_arg_words_speech)
            print(frame_arg_words_gesture)

            # Get the frame handler
            handle = self.handle_function_linker[frame_case[0]["Function"]]

            print('***********')
            print(speech_act)
            print(frame_arg_roles_speech)
            print(frame_arg_roles_gesture)
            print(frame_arg_words_speech)
            print(frame_arg_words_gesture)
            print('***********')

            # Handle the given arguments and return the response
            return handle(speech_act, frame_arg_roles_speech, frame_arg_roles_gesture, frame_arg_words_speech, frame_arg_words_gesture)




    def handle_suggestion(self, request: HandleSuggestionRequest) -> HandleSuggestionResponse:
        """Handle a suggestion frame

        Args:
            request (HandleSuggestionRequest): Frame request

        Returns:
            HandleSuggestionResponse: _description_
        """        
        frame_message = request.frame

        # Get frame
        frame = frame_message.frame
        # Get the speech act
        speech_act = frame_message.speech_act
        # Get the frame name
        frame_name = frame[0]

        print("Trying:", speech_act, frame)

        # Create conversation log
        conversation_log = ConversationLog(SpeakerType.HUMAN, ConversationType(speech_act), [frame_message.full_sentence])

        # Log the frame request into the current speaker conversation
        self.current_speaker.conversation.append(conversation_log)

        # Log the conversation to the log file
        self.log_conversation(conversation_log)

        # Create response
        response = HandleSuggestionResponse()

        # If the frame name is not SUGGESTION
        if frame_name != "SUGGEST" or speech_act != ConversationType.REQUEST.value:
            response.suggestion_result = "None"
            return response
        
        if frame[1] == "si":
            response.suggestion_result = "True"
        elif frame[1] == "no":
            response.suggestion_result = "False"
        else:
            response.suggestion_result = "None"

        return response
    

    """
    
        SPEAKERS

    """

    def translate(self, text: str, lang: str = None) -> str :
        """Translate a text from english to the current speaker language if needed

        Args:
            text (str): Text to translate
            lang (str): Language to use for destination translation. Defaults to None.

        Returns:
            str: Translated text
        """

        # If no lang was provided
        if lang is None:
            # Set lang as the current speaker language
            lang = self.current_speaker.language

        # If the destination language is different from english and italian
        if lang != "en" and lang != "it":
            # Get appropriate model
            model_checkpoint = f"Helsinki-NLP/opus-mt-en-{lang}"  

            # Translate the text
            translator = pipeline("translation", model=model_checkpoint)
            result = translator(text)
            print(result)
            text = result[0]['translation_text']
    
        # Return the text
        return text



    def find_speaker_by_name(self, first_name: str, last_name:  str) -> Speaker:
        """Find speaker by its name among speakers

        Args:
        first_name (str): Speaker first name
            last_name (str): Speaker last name

        Returns:
            Speaker: The speaker found
        """

        # For each speaker
        for speaker in self.speakers:
            # If the first and last name compared are the same
            if speaker.first_name == first_name and speaker.last_name == last_name:
                # Speaker found
                return speaker
            
        # Speaker not found
        return None
    



    def find_speaker_by_uuid(self, uuid: str) -> Speaker:
        """Find speaker by its UUID among speakers

        Args:
            uuid (str): Speaker UUID

        Returns:
            Speaker: The speaker found
        """

        # For each speaker
        for speaker in self.speakers:
            # If the compared UUID is the same
            if speaker.uuid == uuid:
                # Speaker found
                return speaker
            
        # Speaker not found
        return None
    


    def print_speaker_conversation(self, speaker: Speaker):
        """Print the given speaker conversation history

        Args:
            speaker (Speaker): The speaker to print conversation history
        """

        print()
        # Print the speaker first name
        print(speaker.first_name, speaker.last_name)

        # For each log in the conversation
        for log in speaker.conversation:
            # Print the data
            print(log.speaker_type.name, log.conversation_type.name, log.datetime, log.sentences)



    def handle_tts(self, request: TTSRequestRequest) -> TTSRequestResponse:
        """Handle a TTS request by passing translating the sentences according the speaker context and passes it to the TTS node

        Args:
            request (TTSRequestRequest): Request to process

        Returns:
            TTSRequestResponse: Acknowledgment
        """

        # If a speaker UUID was provided
        if request.speaker_uuid is not None:
            # Find the speaker
            speaker = self.find_speaker_by_uuid(request.speaker_uuid)

            # If the speaker was not found
            if speaker is None:
                # Set the speaker as the current one
                speaker = self.current_speaker
            
            # Get the speaker language
            lang = speaker.language
        else:
            # Set the speaker as the current one
            speaker = self.current_speaker

            # Get the current speaker language
            lang = self.current_speaker.language

        # Translate the request sentences
        sentences = [self.translate(sentence, lang) for sentence in request.sentences]


        # If the conversation has more than one log
        if len(speaker.conversation) > 0:
            # Last log
            last_log = speaker.conversation[-1]

            # Current datetime minus the last log datetime
            time_difference = datetime.now() - last_log.datetime #datetime.strptime(last_log.datetime, "%Y/%m/%d %H:%M:%S")

            # If the last log was from the Robot and more than 2 minutes ago
            if last_log.speaker_type is SpeakerType.ROBOT and time_difference > timedelta(minutes=2):
                # Insert a "start conversation" sentence before other sentences
                sentences.insert(0, self.translate(f"{speaker.first_name}, I hope that you are doing great", lang))

        # If the nothing was ever said
        '''else:
            sentences.insert(0, self.translate(f"Hello, I hope that you are doing great", lang))
        '''

        # Create the conversation log
        conversation_log = ConversationLog(SpeakerType.ROBOT, ConversationType(request.type), sentences)

        # Log the response
        speaker.conversation.append(conversation_log)
        
        # Log the conversation to the log file
        self.log_conversation(conversation_log, speaker)


        # TODO: remove
        self.print_speaker_conversation(speaker)

        # Send the TTS Request
        self.send_tts_request(
            sentences,
            lang,
            request.command
        )

        # Return acknowledgment
        return TTSRequestResponse()

    
    def send_tts_request(self, sentences: 'list[str]', lang: str, command: str = "say"):
        """Send a TTS request to the TTS topic

        Args:
            sentences (list[str]): Sentences to send
            lang (str): Language to use
            command (str, optional): Type of TTS command to use. Defaults to "say".
        """       

        # Create message
        tts_message = TTSMessage()
        tts_message.sentences = sentences
        tts_message.type = command
        tts_message.lang = lang

        # Publish message to the topic
        self.tts.publish(tts_message)


    """

        RESPONSES

    """

    def generate_response(self, sentences: 'list[str]', conversation_type: ConversationType, result_sentence:str = "", task_name: str = "", args = []) -> 'tuple[ConversationLog, HandleFrameResponse]':
        """Generate the response to pass back to the decision manager, can possibly contain a task and its arguments

        Args:
            sentences (list[str]): Sentences to say with TTS
            conversation_type (ConversationType): Type conversation to log the response
            result_sentence (str, optional): Task result sentence. Empty for None. Defaults to "".
            task_name (str, optional): Task name to pass to the task manager. Empty for None. Defaults to "".
            args (list, optional): Task arguments. Empty for None. Defaults to [].

        Returns:
            tuple[ConversationLog, HandleFrameResponse]: Conversation log and response
        """
        
        # Translate the sentences
        if self.current_speaker.language == "es" or self.current_speaker.language == "fr" or self.current_speaker.language == "de":
            sentences = [self.translate(sentence) for sentence in sentences]

        # Send the TTS Request
        self.send_tts_request(sentences, self.current_speaker.language)

        # Log the response
        log = ConversationLog(SpeakerType.ROBOT, conversation_type, sentences)

        # Create response message
        response = HandleFrameResponse()
        response.result_sentence = result_sentence
        response.task_name = task_name
        response.speaker_uuid = self.current_speaker.uuid
        response.args = args

        # Return log and response
        return log, response


    def generate_go_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the GO frame 
        """

        if speech_act == "Command":
            if "Direction" in frame_arg_roles:
                idx = frame_arg_roles.index("Direction")
                direction = frame_arg_words[idx]
                if self.current_speaker.language == "it":

                    if direction == "avanti" or direction == "dritto":  
                        return self.generate_response(
                            [f"Sto procedendo {direction}"],
                            ConversationType.ANSWER,
                            f"Non riesco a raggiungerti. Ci sono molti rami spezzati nel filare.",
                            #f"Ho terminato di andare {direction}",
                            f"go_forward",
                            frame_arg_words
                        )

                    elif direction == "indietro":  
                        return self.generate_response(
                            [f"Sto andando {direction}. Dimmi quando fermarmi."],
                            ConversationType.ANSWER,
                            f"Ho terminato di andare {direction}",
                            f"go_backward",
                            frame_arg_words
                        )
                    
                    
                    


                '''else
                    return self.generate_response(
                        [f"I am going {direction}"],
                        ConversationType.ANSWER,
                        f"I finished moving {direction}",
                        f"go_task",
                        frame_arg_words
                    )'''
                

    def generate_stop_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the STOP frame 
        """

        if speech_act == "Command":
            if self.current_speaker.language == "it":
                return self.generate_response(
                        [f"Comando di arresto ricevuto."],
                        ConversationType.ANSWER,
                        f"Sono fermo.",
                        f"stop_task",
                        frame_arg_words
                )
            

    def generate_block_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the BLOCK frame 
        """

        if speech_act == "Command" or speech_act == "Information":
            if self.current_speaker.language == "it":
                return self.generate_response(
                            [f"Non riesco a venire da te. Ci sono rami spezzati nel filare."],
                            ConversationType.ANSWER,
                            f"",
                            f"block_task",
                            frame_arg_words
                )
            

    def generate_turn_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the TURN frame 
        """

        if speech_act == "Command":
            if frame_arg_roles_1 == [] and frame_arg_words_1 == []: # only 1 modality
                if "Direction" in frame_arg_roles:
                    idx = frame_arg_roles.index("Direction")
                    direction = frame_arg_words[idx]
                    
                    if self.current_speaker.language == "it":
                        if direction == "destra" or direction == "dritto":  
                            return self.generate_response(
                                [f"Giro a {direction} di 90 gradi"],
                                ConversationType.ANSWER,
                                f"Ho terminato di andare {direction}",
                                f"turn_right",
                                frame_arg_words
                            )

                        elif direction == "sinistra":  
                            return self.generate_response(
                                [f"Sto girando a {direction} di 90 gradi"],
                                ConversationType.ANSWER,
                                f"Ho terminato di andare {direction}",
                                f"turn_left",
                                frame_arg_words
                            )
                            
                        elif direction == "indietro" or direction == "contrario" or direction == "mi" or direction == "me":  
                            return self.generate_response(
                                [f"Sto ruotando di 180 gradi"],
                                ConversationType.ANSWER,
                                f"Ho terminato di ruotare",
                                f"turn_180",
                                frame_arg_words
                            )
                            
                '''else
                    return self.generate_response(
                        [f"I am going {direction}"],
                        ConversationType.ANSWER,
                        f"I finished moving {direction}",
                        f"go_task",
                        frame_arg_words
                    )'''
                
            else:
                if "Direction" in frame_arg_roles and "Direction" in frame_arg_roles_1:
                    idx = frame_arg_roles.index("Direction")
                    idx_1 = frame_arg_roles_1.index("Direction")
                    direction = frame_arg_words[idx]
                    direction_1 = frame_arg_words_1[idx_1]
                    if direction != direction_1:
                        return self.generate_response(
                            [f"Ho ricevuto diverse direzioni: {direction} attraverso il parlato e {direction_1} attraverso i gesti"],
                                ConversationType.ERROR,
                                f"Errore nella direzione",
                                f"stop_task",
                                frame_arg_words
                            )
                    # In case the direction in the same
                    if self.current_speaker.language == "it":
                        if direction == "destra" or direction == "dritto":  
                            return self.generate_response(
                                [f"Sto girando a {direction} di 90 gradi"],
                                ConversationType.ANSWER,
                                f"Ho terminato di andare {direction}",
                                f"turn_right",
                                frame_arg_words
                            )

                        elif direction == "sinistra":  
                            return self.generate_response(
                                [f"Sto girando a {direction} di 90 gradi"],
                                ConversationType.ANSWER,
                                f"Ho terminato di andare {direction}",
                                f"turn_left",
                                frame_arg_words
                            )
                        
                        elif direction == "indietro" or direction == "contrario":  
                            return self.generate_response(
                                [f"Sto girando di 180 gradi"],
                                ConversationType.ANSWER,
                                f"Ho terminato di ruotare",
                                f"turn_180",
                                frame_arg_words
                            )



    def generate_check_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the CHECK frame 
        """

        if speech_act == "Command":
            if "Theme" and "Possessor" and "Location" and "Location_Per" in frame_arg_roles:
                idx_theme = frame_arg_roles.index("Theme")
                idx_poss = frame_arg_roles.index("Possessor")
                idx_loc = frame_arg_roles.index("Location")
                idx_loc_per = frame_arg_roles.index("Location_Per")
                theme = frame_arg_words[idx_theme]
                poss = frame_arg_words[idx_poss]
                loc = frame_arg_words[idx_loc]
                loc_per = frame_arg_words[idx_loc_per]
                if self.current_speaker.language == "it" and theme == "qualità" and poss == "grappoli" and loc == "sopra" and loc_per == "te":
                    # TODO
                    return self.generate_response(
                        [f"Guarda lo schermo"],
                        ConversationType.ANSWER,
                        f"",
                        f"check",
                        frame_arg_words
                    )
                
            elif "Time" and "Theme" and "Attribute" in frame_arg_roles:
                idx_time = frame_arg_roles.index("Time")
                idx_theme = frame_arg_roles.index("Theme")
                idx_attr = frame_arg_roles.index("Attribute")
                time = frame_arg_words[idx_time]
                theme = frame_arg_words[idx_theme]
                attr = frame_arg_words[idx_attr]
                if self.current_speaker.language == "it" and time == "anche" and theme == "livello" and attr == "zuccherino":
                    # TODO
                    return self.generate_response(
                        [f"E la dimensione dei gràppoli?"],
                        ConversationType.ANSWER,
                        f"",
                        f"check",
                        frame_arg_words
                    )
                    
            elif "Theme" in frame_arg_roles:
                idx = frame_arg_roles.index("Theme")
                theme = frame_arg_words[idx]
                if self.current_speaker.language == "it":

                    if theme == "grappolo" or theme == "livello":  
                        return self.generate_response(
                            [f"Sto controllando il {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di controllare il {theme}",
                            f"check",
                            frame_arg_words
                        )
                    

                    elif theme == "grappoli":
                        return self.generate_response(
                            [f"Sto controllando i {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di controllare i {theme}",
                            f"check",
                            frame_arg_words
                        )
                    

                    elif theme == "foglie":
                        return self.generate_response(
                            [f"Sto controllando le {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di controllare le {theme}",
                            f"check",
                            frame_arg_words
                        )
                    


                '''else
                    return self.generate_response(
                        [f"I am going {direction}"],
                        ConversationType.ANSWER,
                        f"I finished moving {direction}",
                        f"go_task",
                        frame_arg_words
                    )'''
                



    def generate_harvest_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the HARVEST frame 
        """

        if speech_act == "Command":
            if "Theme" in frame_arg_roles:
                idx = frame_arg_roles.index("Theme")
                theme = frame_arg_words[idx]
                if self.current_speaker.language == "it":

                    if theme == "grappolo":  
                        return self.generate_response(
                            [f"Sto raccogliendo il {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di raccogliere il {theme}",
                            f"harvest",
                            frame_arg_words
                        )
                    

                    elif theme == "grappoli":
                        return self.generate_response(
                            [f"Sto raccogliendo i {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di raccogliere i {theme}",
                            f"harvest",
                            frame_arg_words
                        )
                    

                    elif theme == "uva":
                        return self.generate_response(
                            [f"Sto raccogliendo i gràppoli maturi"],
                            ConversationType.ANSWER,
                            f"Ho terminato di raccogliere l'{theme}",
                            f"harvest",
                            frame_arg_words
                        )
                    
                    elif theme == "rami":
                        return self.generate_response(
                            [f"Sto tagliando i {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di tagliare i {theme}",
                            f"prune",
                            frame_arg_words
                        )
                    
                    elif theme == "foglie":
                        return self.generate_response(
                            [f"Sto tagliando le {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di tagliare le {theme}",
                            f"prune",
                            frame_arg_words
                        )

        elif speech_act == "Information":
            if "Theme" and "Attribute" in frame_arg_roles:
                idx_theme = frame_arg_roles.index("Theme")
                idx_attr = frame_arg_roles.index("Attribute")
                theme = frame_arg_words[idx_theme]
                attr = frame_arg_words[idx_attr]
                if self.current_speaker.language == "it" and theme == "grappoli" and attr == "id":
                    
                    # TODO
                    return self.generate_response(
                        [f"Il nuovo piano di raccolta è gràppolo: X, Y e poi Z"],
                        ConversationType.ANSWER,
                        f"",
                        f"",
                        frame_arg_words
                    )

        

    def generate_prune_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the PRUNE frame 
        """

        if speech_act == "Command":
            if "Theme" in frame_arg_roles:
                idx = frame_arg_roles.index("Theme")
                theme = frame_arg_words[idx]
                if self.current_speaker.language == "it":

                    if theme == "ramo":  
                        return self.generate_response(
                            [f"Sto potando il {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di potare il {theme}",
                            f"prune",
                            frame_arg_words
                        )
                    

                    elif theme == "rami":
                        return self.generate_response(
                            [f"Sto potando i {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di potare i {theme}",
                            f"prune",
                            frame_arg_words
                        )
                    
                    


                '''else
                    return self.generate_response(
                        [f"I am going {direction}"],
                        ConversationType.ANSWER,
                        f"I finished moving {direction}",
                        f"go_task",
                        frame_arg_words
                    )'''
                

    def generate_cut_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the CUT frame 
        """

        if speech_act == "Command":
            if "Theme" in frame_arg_roles:
                idx = frame_arg_roles.index("Theme")
                theme = frame_arg_words[idx]
                if self.current_speaker.language == "it":

                    if theme == "grappolo":  
                        return self.generate_response(
                            [f"Sto tagliando il {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di tagliare il {theme}",
                            f"harvest",
                            frame_arg_words
                        )
                    

                    elif theme == "ramo":
                        return self.generate_response(
                            [f"Sto tagliando il {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di tagliare il {theme}",
                            f"prune",
                            frame_arg_words
                        )
                    

                    elif theme == "foglie":
                        return self.generate_response(
                            [f"Sto tagliando le {theme}"],
                            ConversationType.ANSWER,
                            f"Ho terminato di tagliare le {theme}",
                            f"prune",
                            frame_arg_words
                        )
                    
                    


                '''else
                    return self.generate_response(
                        [f"I am going {direction}"],
                        ConversationType.ANSWER,
                        f"I finished moving {direction}",
                        f"go_task",
                        frame_arg_words
                    )'''
                

    def generate_speak_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the SPEAK frame 
        """

        if speech_act == "Command + Information" or speech_act == "Information + Information":
            if "Theme" in frame_arg_roles:
                idx = frame_arg_roles.index("Theme")
                theme = frame_arg_words[idx]
                if self.current_speaker.language == "it":

                    return self.generate_response(
                            [f"Ciao Sara, piacere di conòscerti e benvenuta in azienda"],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )
    
    def generate_do_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the DO frame 
        """

        if speech_act == "Request":
            if "Cosa" or "cosa" in frame_arg_words:
                if self.current_speaker.language == "it":

                    return self.generate_response(
                            [f"Posso rilevare i gràppoli, dirti la loro dimensione, il colore, la posizione, la maturazione, contarli e pianificare la loro raccolta."],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )
                
    def generate_detect_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the DETECT frame 
        """

        if speech_act == "Request":
            if "Specifier" and "Theme" in frame_arg_roles:
                idx_spec = frame_arg_roles.index("Specifier")
                idx_theme = frame_arg_roles.index("Theme")
                spec = frame_arg_words[idx_spec]
                theme = frame_arg_words[idx_theme]
                if self.current_speaker.language == "it" and spec == "quanti" and theme == "grappoli":

                    return self.generate_response(
                            [f"Dove?"],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )
                
    def generate_be_exist_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the BE_EXIST frame 
        """

        if speech_act == "Request":
            if "Specifier" and "Theme" and "Location" and "Location_Per" in frame_arg_roles and ("quanti" and "grappoli" and "fronte" and "te" in frame_arg_words):
                idx_spec = frame_arg_roles.index("Specifier")
                idx_theme = frame_arg_roles.index("Theme")
                idx_loc = frame_arg_roles.index("Location")
                idx_loc_per = frame_arg_roles.index("Location_Per")
                spec = frame_arg_words[idx_spec]
                theme = frame_arg_words[idx_theme]
                loc = frame_arg_words[idx_loc]
                loc_per = frame_arg_words[idx_loc_per]
                if self.current_speaker.language == "it" and spec == "quanti" and theme == "grappoli" and loc == "fronte" and loc_per == "te":
                    # TODO
                    return self.generate_response(
                            [f"Ci sono 8 gràppóli"],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )
            
            elif "Specifier" and "Quantity" and "Attribute" in frame_arg_roles and ("quanti" and "quelli" and "maturi" in frame_arg_words):
                idx_spec = frame_arg_roles.index("Specifier")
                idx_quant = frame_arg_roles.index("Quantity")
                idx_attr = frame_arg_roles.index("Attribute")
                spec = frame_arg_words[idx_spec]
                quant = frame_arg_words[idx_quant]
                attr = frame_arg_words[idx_attr]
                
                if self.current_speaker.language == "it" and spec == "quanti" and quant == "quelli" and attr == "maturi":
                    # TODO
                    return self.generate_response(
                            [f"3 sono maturi e 5 non sono ancora pronti per essere raccolti"],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )
            
            elif "Specifier" and "Theme" and "Possessor" and "Possessor_Attribute" in frame_arg_roles and ("qual" and "colore" and "grappoli" and "maturi" in frame_arg_words):
                idx_spec = frame_arg_roles.index("Specifier")
                idx_theme = frame_arg_roles.index("Theme")
                idx_poss = frame_arg_roles.index("Possessor")
                idx_poss_att = frame_arg_roles.index("Possessor_Attribute")
                spec = frame_arg_words[idx_spec]
                theme = frame_arg_words[idx_theme]
                poss = frame_arg_words[idx_poss]
                poss_att = frame_arg_words[idx_poss_att]
                
                if self.current_speaker.language == "it" and spec == "qual" and theme == "colore" and poss == "grappoli" and poss_att == "maturi":
                    # TODO
                    return self.generate_response(
                            [f"I gràppoli maturi sono di colore viola"],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )
            
            elif "Specifier" and "Theme" and "Possessor" and "Attribute" in frame_arg_roles and ("qual" and "piano" and "raccolta" and "pronti" in frame_arg_words):
                idx_spec = frame_arg_roles.index("Specifier")
                idx_theme = frame_arg_roles.index("Theme")
                idx_poss = frame_arg_roles.index("Possessor")
                idx_att = frame_arg_roles.index("Attribute")
                spec = frame_arg_words[idx_spec]
                theme = frame_arg_words[idx_theme]
                poss = frame_arg_words[idx_poss]
                att = frame_arg_words[idx_att]
                
                if self.current_speaker.language == "it" and (spec == "qual" or spec == "quale") and theme == "piano" and poss == "raccolta" and att == "pronti":
                    # TODO
                    return self.generate_response(
                            [f"Raccògliere prima il gràppolo con ID 1, poi 0 e alla fine 2."],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )
            
        elif speech_act == "Information":
            if "Time" and "Attribute" in frame_arg_roles:
                idx_time = frame_arg_roles.index("Time")
                idx_att = frame_arg_roles.index("Attribute")
                time = frame_arg_words[idx_time]
                att = frame_arg_words[idx_att]
                
                if self.current_speaker.language == "it" and time == "anche" and att == "importante":
                    # TODO
                    return self.generate_response(
                            [f"Ecco il risultato dell'anàlisi."],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )

                


    def generate_show_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the SHOW frame 
        """

        if speech_act == "Command":
            if "Location_Spec" and "Attribute" in frame_arg_roles:
                idx_loc_spec = frame_arg_roles.index("Location_Spec")
                idx_attr = frame_arg_roles.index("Attribute")
                loc_spec = frame_arg_words[idx_loc_spec]
                attr = frame_arg_words[idx_attr]
                if self.current_speaker.language == "it" and loc_spec == "quelli" and attr == "maturi":

                    return self.generate_response(
                            [f"Sono sullo schermo. Quelli nei rettangoli verdi."],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )
                

    def generate_remove_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the REMOVE frame 
        """

        if speech_act == "Command":
            if "Theme" and "Attribute" in frame_arg_roles:
                idx_theme = frame_arg_roles.index("Theme")
                idx_attr = frame_arg_roles.index("Attribute")
                theme = frame_arg_words[idx_theme]
                attr = frame_arg_words[idx_attr]
                if self.current_speaker.language == "it" and theme == "rami" and attr == "secchi":

                    return self.generate_response(
                            [f"Nel filare non ci sono rami con problemi"],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )
                

    def generate_finish_response(self, speech_act: str, frame_arg_roles: list, frame_arg_roles_1: list, frame_arg_words: list, frame_arg_words_1: list) -> HandleFrameResponse:
        """
        Generate the response for the FINISH frame 
        """

        if speech_act == "Request + Information":
            if "Theme" and "Time" and "Location" in frame_arg_roles:
                idx_theme = frame_arg_roles.index("Theme")
                idx_time = frame_arg_roles.index("Time")
                idx_loc = frame_arg_roles.index("Location")
                theme = frame_arg_words[idx_theme]
                time = frame_arg_words[idx_time]
                loc = frame_arg_words[idx_loc]
                if self.current_speaker.language == "it" and theme == "lavoro" and time == "oggi" and loc == "robot":

                    return self.generate_response(
                            [f"Ciao Sara"],
                            ConversationType.ANSWER,
                            f"",
                            f"",
                            frame_arg_words
                    )


    def generate_introduce_response(self, type: str, first_name: str, last_name: str, language: str)  -> HandleFrameResponse:
        """
        Generate the response for the INTRODUCE frame name
        """

        # Find if speaker with the given first and last name exists
        existing_speaker = self.find_speaker_by_name(first_name, last_name)

        # If the user has not been met yet
        if existing_speaker is None:
            # Create the new speaker
            speaker = Speaker(first_name, last_name, language)

            # Add the speaker to the list of speakers
            self.speakers.append(speaker)

            # Create the speaker conversation log file
            create_csv_with_headers_if_does_not_exist(f"dialog_{speaker.first_name}_{speaker.last_name}_conversation_logs.csv", ("datetime", "speaker_type", "conversation_type", "sentences"))

            # Set the current speaker as the added one
            self.current_speaker = self.speakers[-1]

            # Return the generated and translated response
            return self.generate_response(
                [f"Piacere di conoscerti {self.current_speaker.first_name}"],
                ConversationType.ANSWER
            )
            
        else:
            # Set the current speaker as the provided one
            self.current_speaker = existing_speaker

            # Return the generated and translated response
            return self.generate_response(
                [f"Bentornato {self.current_speaker.first_name}"],
                ConversationType.ANSWER
            )
    
    

"""

    MAIN

"""

if __name__ == '__main__':
    DialogManager()