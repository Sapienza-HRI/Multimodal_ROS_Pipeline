#!/usr/bin/env python3

import rospy
from enum import Enum
from datetime import datetime, timedelta
import time
from time import sleep

from canopies_utils.log_utils import create_csv_with_headers_if_does_not_exist, write_to_log_file

from dialog_manager.srv import HandleFrame, HandleFrameResponse
from dialog_manager.srv import HandleSuggestion, HandleSuggestionResponse
from dialog_manager.srv import TTSRequest
from dialog_manager.msg import FrameMessage
from task_manager.srv import AddTask, AddTaskResponse
from task_manager.srv import KillAllTasks
from decision_manager.msg import FrameGroupMessage
from decision_manager.msg import IssueMessage
from decision_manager.srv import TaskResult, TaskResultRequest, TaskResultResponse


class OngoingTask:
    """
    Represent an ongoing task
    """
    def __init__(self, result_sentence: str, speaker_uuid: str):
        self.result_sentence = result_sentence
        self.speaker_uuid = speaker_uuid


"""

    FRAMES

"""

class Frame:
    """
    Represent a single frame
    """
    def __init__(self, frame_roles: 'list[str]', frame_words: 'list[str]', speech_act: str, full_sentence: str, language: str):
        self.frame_roles = frame_roles
        self.frame_words = frame_words
        self.speech_act = speech_act
        self.full_sentence = full_sentence
        self.language = language



"""

    ISSUE

"""

class IssueSeverity(Enum):
    MINOR_ERROR = 1 # not used in that scope
    MINOR_PROBLEM = 2
    MAJOR_PROBLEM = 3
    MAJOR_ERROR = 4



class Issue:
    """
    Represent an issue
    """
    def __init__(self, severity: IssueSeverity, name: str, sentence: str, task_uuid: str = None):
        self.severity = severity
        self.name = name
        self.sentence = sentence
        self.task_uuid = task_uuid


issue_to_suggestion = {
    "Grape disappeared": "Are you able to fix the problem ?"
}

"""
Issue name to user suggestion
"""




"""

    LOGS

"""

class FrameLog:
    """
    Represent a single frame log
    """

    def __init__(self, frame: Frame):
        self.datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.frame_roles = frame.frame_roles
        self.frame_words = frame.frame_words
        self.speech_act = frame.speech_act
        self.full_sentence = frame.full_sentence
        self.language = frame.language



"""

    MANAGER

"""

class DecisionManager:
    def __init__(self):
        rospy.loginfo("Starting")

        # Variables
        self.frames_groups: list[list[Frame]] = []
        self.frames_groups_gesture: list[list[Frame]] = []
        self.combination: list[list[Frame]] = []
        self.ongoing_tasks: dict[str, OngoingTask] = {}
        self.issues: list[Issue] = []
        self.system_is_blocked = False
        self.start_time = ""
        self.end_time = ""

        # Init node
        rospy.init_node("decision_manager_node", anonymous=True)

        # Services
        # Dialog manager service
        rospy.wait_for_service('handle_frame_service')
        self.handle_frame_service = rospy.ServiceProxy('handle_frame_service', HandleFrame)

        # Handle suggestion service
        rospy.wait_for_service('handle_suggestion_service')
        self.handle_suggestion_service = rospy.ServiceProxy('handle_suggestion_service', HandleSuggestion)

        # TTS service
        rospy.wait_for_service('tts_service')
        self.tts_service = rospy.ServiceProxy('tts_service', TTSRequest)

        # Task services
        rospy.wait_for_service('add_task')
        self.add_task = rospy.ServiceProxy('add_task', AddTask)
        rospy.wait_for_service('kill_all_tasks')
        self.kill_all_tasks = rospy.ServiceProxy('kill_all_tasks', KillAllTasks)

        
        # Emit light signals service
        #rospy.wait_for_service('/canopies_simulator/robot00/light_signal')
        #self.light_signals_service = rospy.ServiceProxy('/canopies_simulator/robot00/light_signal', light_signals)

        # Create finish task service
        self.finish_task_service = rospy.Service('finish_task', TaskResult, self.finish_ongoing_task)

        # Publishers
        #self.play_audio = rospy.Publisher('/canopies_simulator/sound_play', SoundPlay, queue_size=10)


        # Subscribers
        self.new_frames_gest = rospy.Subscriber('/decision_manager_topic_gesture', FrameGroupMessage, self.manage_new_frame_group_gesture, queue_size=10)
        self.new_frames = rospy.Subscriber('/decision_manager_topic_speech', FrameGroupMessage, self.manage_new_frame_group_speech, queue_size=10)
        self.new_issue = rospy.Subscriber('issue_handler_topic', IssueMessage, self.manage_new_issue, queue_size=10)

        # Main runtime
        self.main_runtime()




    def log_frame(self, frame_log: FrameLog):
        """Write the given log to the log file

        Args:
            frame_log (FrameLog): Log to write in the log file
        """
        write_to_log_file(log_file_name, (frame_log.datetime, frame_log.frame_roles, frame_log.frame_words, frame_log.speech_act, frame_log.full_sentence, frame_log.language))



    def main_runtime(self):
        rospy.loginfo("Started")
        
        while not rospy.is_shutdown():

            # For each issue in the issue list
            for issue in self.issues:
                # MINOR PROBLEM - 2
                if issue.severity == IssueSeverity.MINOR_PROBLEM:
                    # Get speaker UUID
                    speaker_uuid = self.ongoing_tasks[issue.task_uuid].speaker_uuid

                    # Tell the user about the problem
                    self.tts_service(
                        "say",
                        "Problem",
                        [issue.sentence],
                        speaker_uuid
                    )
                
                # MAJOR PROBLEM - 3
                elif issue.severity == IssueSeverity.MAJOR_PROBLEM:
                    # Get speaker UUID
                    speaker_uuid = self.ongoing_tasks[issue.task_uuid].speaker_uuid

                    # Get a suggestion sentence linked with the issue
                    suggestion_sentence = issue_to_suggestion[issue.name]

                    # Emit light signals
                    #self.light_signals_service(ledstrip_id: int, pattern: int, frequency: float)

                    # Tell the user about the problem
                    self.tts_service(
                        "say",
                        "Problem",
                        [issue.sentence],
                        speaker_uuid
                    )

                    # Suggestion to correct the encountered problem
                    self.tts_service(
                        "say",
                        "Suggestion",
                        [suggestion_sentence, 'Please respond only by yes or no'],
                        speaker_uuid
                    )

                    # Wait for suggestion result
                    suggestion_result = self.wait_for_suggestion(speaker_uuid)

                    # If the user responded by Yes                
                    if suggestion_result == True:
                        self.tts_service(
                            "say",
                            "Answer",
                            ["Great"],
                            speaker_uuid
                        )
                    # If the user responded by No
                    elif suggestion_result == False:
                        self.tts_service(
                            "say",
                            "Answer",
                            ["No problem"],
                            speaker_uuid
                        )
                    # The user didn't responded
                    else:
                        self.tts_service(
                            "say",
                            "Answer",
                            ["I did not get your response, so I am ending the current task"],
                            speaker_uuid
                        )

                    # Delete ongoing task
                    del self.ongoing_tasks[issue.task_uuid]

                # MAJOR ERROR - 4
                elif issue.severity == IssueSeverity.MAJOR_ERROR:
                    # Kill all the tasks
                    self.kill_all_tasks()

                    # Delete all local ongoing tasks
                    self.ongoing_tasks = {}

                    # Emit light signals
                    #self.light_signals_service(ledstrip_id: int, pattern: int, frequency: float)

                    # Tell the user about the issue
                    self.tts_service(
                        "say",
                        "Error",
                        [issue.sentence],
                        None
                    )

                    # The runtime will now no more allow new tasks and has to be rebooted in order to come back to it's initial state
                    self.system_is_blocked = True

                # Remove the issue from the issue list because it was processed
                self.issues.remove(issue)
            
            # If there is no frame
            if len(self.frames_groups) == 0 and len(self.frames_groups_gesture) == 0:
                continue
            
            # If there are frames
            else:
                # If the system is blocked
                if self.system_is_blocked:
                    # Tell the user that the system is blocked
                    self.tts_service(
                        "say",
                        "Information",
                        [
                            "The system is blocked due to an internal problem",
                            "I cannot handle any further speech or gesture request as an input",
                            "Please contact a technician to fix the problem"
                        ],
                        None
                    )

                    # Empty the frames groups
                    self.frames_groups = []
                    continue
            


            # -------------------------
            #   COMBINATION (SPEECH + GESTURE)
            #             &
            #        ONLY SPEECH
            # -------------------------

            # Combining speech and gesture information
            if self.frames_groups != []:

                # Wait 2 seconds to check if any gestural information was provided in input
                time.sleep(2)

                # -------------------------
                #   COMBINATION (SPEECH + GESTURE)
                # -------------------------
                if self.frames_groups_gesture != [] and "standing" not in self.frames_groups_gesture[0]:
                    print("COMBINATION!!!!!")

                    # For each frame in the first frame group
                    for i, frame_ in enumerate(self.frames_groups[0]):
                        frame: Frame = frame_
                        frame_gest: Frame = self.frames_groups_gesture[0][i]
                        
                        print("Frame roles and frame words from SPEECH")
                        print(frame.frame_roles)
                        print(frame.frame_words)
                        print('------------------')
                        print("Frame roles and frame words from GESTURE")
                        print(frame_gest.frame_roles)
                        print(frame_gest.frame_words)
                        print('------------------')

                        # Log the received frame
                        self.log_frame(FrameLog(frame))
                        self.log_frame(FrameLog(frame_gest))

                        # Create frame message
                        frame_message = FrameMessage()
                        frame_message.frame_roles = frame.frame_roles + ["|"] + frame_gest.frame_roles
                        frame_message.frame_words = frame.frame_words + ["|"] + frame_gest.frame_words
                        
                        if frame.speech_act == frame_gest.speech_act:
                            frame_message.speech_act = frame.speech_act
                        else:
                            self.tts_service(
                                "say",
                                "Error",
                                [
                                    f"Ho identificato diversi atti linguistici: {frame.speech_act.lower()} attraverso il parlato e {frame_gest.speech_act.lower()} attraverso i gesti"
                                ],
                                None
                            ) 
                            continue
                        
                        frame_message.full_sentence = frame.full_sentence
                        frame_message.language = frame.language

                        print('*******************')
                        print(frame_message.frame_roles)
                        print(frame_message.frame_words)
                        print(frame_message.speech_act)
                        print('*******************')


                        # Get from DIM a Response and a potential Task to execute 
                        response: HandleFrameResponse = self.handle_frame_service(frame_message)

                        # If there is a task in the dialog manager response
                        if response.task_name != "":
                            # Add task to Task Manager
                            task_request_response: AddTaskResponse = self.add_task(response.task_name, response.args)

                            # Add local ongoing task with the new task identified with its UUID
                            self.ongoing_tasks[task_request_response.uuid] = OngoingTask(response.result_sentence, response.speaker_uuid)
                        
                    # Delete the processed frame group
                    del self.frames_groups[0]
                    del self.frames_groups_gesture[0][0]


            
                # -------------------------
                #       SPEECH ONLY
                # -------------------------

                # Analyse information coming only from SPEECH
                #elif self.frames_groups != [] and self.frames_groups_gesture == []:
                else:
                    print('ONLY SPEECH')

                    # For each frame in the first frame group
                    for frame_ in self.frames_groups[0]:
                        frame: Frame = frame_

                        print(frame.frame_roles)
                        print(frame.frame_words)
                        print('------------------')

                        # Log the received frame
                        self.log_frame(FrameLog(frame))

                        # Create frame message
                        frame_message = FrameMessage()
                        frame_message.frame_roles = frame.frame_roles
                        frame_message.frame_words = frame.frame_words
                        frame_message.speech_act = frame.speech_act
                        frame_message.full_sentence = frame.full_sentence
                        frame_message.language = frame.language

                        # Get from DIM a Response and a potential Task to execute 
                        response: HandleFrameResponse = self.handle_frame_service(frame_message)

                        # If there is a task in the dialog manager response
                        if response.task_name != "":
                            # Add task to Task Manager
                            task_request_response: AddTaskResponse = self.add_task(response.task_name, response.args)

                            # Add local ongoing task with the new task identified with its UUID
                            self.ongoing_tasks[task_request_response.uuid] = OngoingTask(response.result_sentence, response.speaker_uuid)
                    
                    # Delete the processed frame group
                    del self.frames_groups[0]

        

            # -------------------------
            #       GESTURE ONLY
            # -------------------------

            # Analyse information coming only from GESTURE
            elif self.frames_groups == [] and self.frames_groups_gesture != []:
                print('ONLY GESTURE')
                # For each frame in the first frame group
                for frame_ in self.frames_groups_gesture[0]:
                    frame: Frame = frame_

                    print(frame.frame_roles)
                    print(frame.frame_words)
                    print('------------------')

                    # Log the received frame
                    self.log_frame(FrameLog(frame))

                    # Create frame message
                    frame_message = FrameMessage()
                    frame_message.frame_roles = frame.frame_roles
                    frame_message.frame_words = frame.frame_words
                    frame_message.speech_act = frame.speech_act
                    frame_message.full_sentence = frame.full_sentence
                    frame_message.language = frame.language

                    # Get from DIM a Response and a potential Task to execute 
                    response: HandleFrameResponse = self.handle_frame_service(frame_message)

                    # If there is a task in the dialog manager response
                    if response.task_name != "":
                        # Add task to Task Manager
                        task_request_response: AddTaskResponse = self.add_task(response.task_name, response.args)

                        # Add local ongoing task with the new task identified with its UUID
                        self.ongoing_tasks[task_request_response.uuid] = OngoingTask(response.result_sentence, response.speaker_uuid)
            

                # Delete the processed frame group
                del self.frames_groups_gesture[0]

            

            '''print("***********************")
            print('Time required to DEM is:')
            print(datetime.now() - time_start)
            print("***********************")'''


    def wait_for_suggestion(self, speaker_uuid = None) -> bool:
        """Wait for suggestion reply

        Returns:
            bool: the suggestion reply
        """

        # Initial suggestion waiting time
        initial_time = datetime.now()

        # While no break, repeat wait for the suggestion reply
        while True:
            # Take initial group length
            initial_frame_groups_len = len(self.frames_groups)

            # If the initial length has not changed, sleep
            while initial_frame_groups_len == len(self.frames_groups):
                sleep(0.5)

                # Time difference is initial time minus now
                time_difference = datetime.now() - initial_time

                # If elapsed time is more than 1 minute
                #if time_difference > timedelta(minutes=1):
                if time_difference > timedelta(seconds=10):
                    return None

            # Get the last frame group
            frame_group = self.frames_groups[-1]

            # Remove the last frame (suggestion frame)
            del self.frames_groups[-1]

            # If the group has exactly one frame
            if len(frame_group) == 1:
                # Get the frame
                frame = frame_group[0]

                # Create frame message
                frame_message = FrameMessage()
                frame_message.frame_roles = frame.frame_roles
                frame_message.frame_words = frame.frame_words
                frame_message.speech_act = frame.speech_act
                frame_message.full_sentence = frame.full_sentence
                frame_message.language = frame.language

                # Process frame to get the result
                suggestion_result: HandleSuggestionResponse = self.handle_suggestion_service(frame_message)

                # If the result is either True or False, otherwise start over the loop
                if suggestion_result.suggestion_result in ["True", "False"]:

                    # Return the result as a boolean
                    return suggestion_result.suggestion_result == "True"
                
            # Suggestion to correct the encountered problem
            self.tts_service(
                "say",
                "Answer",
                ["I did not understood your reply", "Please respond only by yes or no"],
                speaker_uuid
            )
               

    
    def manage_new_frame_group_gesture(self, data: FrameGroupMessage):
        """Manage received frames
        
        Args:
            data (Frames): The list of received frames
        """
        
        print("Accessed manage_new_frame_group function GESTURE")

        # Create empty frame list
        frames_to_append: list[Frame] = []

        # For each frame in the message
        for frame_msg in data.frames:
            # Append the frame to the local frame list
            frames_to_append.append(Frame(frame_msg.frame_roles, frame_msg.frame_words, frame_msg.speech_act, frame_msg.full_sentence, frame_msg.language))

        # Append the local frame list to the global frame list
        self.frames_groups_gesture.append(frames_to_append)

        print("Ended manage_new_frame_group_gesture function")



    def manage_new_frame_group_speech(self, data: FrameGroupMessage):
        """Manage received frames
        
        Args:
            data (Frames): The list of received frames
        """
        
        print("Accessed manage_new_frame_group function SPEECH")

        # Create empty frame list
        frames_to_append: list[Frame] = []

        # For each frame in the message
        for frame_msg in data.frames:
            # Append the frame to the local frame list
            frames_to_append.append(Frame(frame_msg.frame_roles, frame_msg.frame_words, frame_msg.speech_act, frame_msg.full_sentence, frame_msg.language))

        # Append the local frame list to the global frame list
        self.frames_groups.append(frames_to_append)

        print("Ended manage_new_frame_group_speech function")
        


    
    def manage_new_issue(self, issue_message: IssueMessage):
        """Handle received issue

        Args:
            issue_message (IssueMessage): Issue to handle
        """
        print("Received new issue:", issue_message.severity, issue_message.issue_name)

        # Create issue
        issue = Issue(IssueSeverity(issue_message.severity), issue_message.issue_name, issue_message.issue_sentence, issue_message.task_uuid)

        # Add the new issue to the issue list
        self.issues.append(issue)




    def finish_ongoing_task(self, task: TaskResultRequest):
        """Say the result and delete local ongoing task

        Args:
            task (TaskResultRequest): The task to finish

        Returns:
            TaskResultResponse: Just an acknowledgment
        """        

        # Get ongoing task
        local_ongoing_task = self.ongoing_tasks[task.task_uuid]

        # Use the TTS service to say the given results
        self.tts_service(
            "say",
            "Result",
            [local_ongoing_task.result_sentence.format(*task.results)],
            local_ongoing_task.speaker_uuid
        )
        
        # Delete the task in the ongoing task list
        del self.ongoing_tasks[task.task_uuid]

        # Return acknowledgment
        return TaskResultResponse()
    

"""

    MAIN

"""

if __name__ == "__main__":
    # Name of the log file
    log_file_name = "frame_logs.csv"
    
    create_csv_with_headers_if_does_not_exist(log_file_name, ("datetime", "frame_roles", "frame_words", "speech_act", "full_sentence", "language"))

    DecisionManager()
