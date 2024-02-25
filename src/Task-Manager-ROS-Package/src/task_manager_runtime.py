#!/usr/bin/env python3

import rospy
import uuid
import time
from time import sleep
from datetime import datetime
from enum import Enum
from multiprocessing import Process, Manager

from canopies_utils.log_utils import create_csv_with_headers_if_does_not_exist, write_to_log_file

from task_manager.srv import AddTask, AddTaskRequest, AddTaskResponse
from task_manager.srv import KillAllTasks, KillAllTasksRequest, KillAllTasksResponse
from decision_manager.srv import TaskResult
from issue_harvester.msg import ProblemMessage
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import ColorRGBA

from canopies_simulator.msg import SoundPlay
from canopies_simulator.srv import light_signals, light_signalsRequest, led_segment, led_segmentRequest


class TaskThread(Process):
    def __init__(self, name, target, args):
        self.results = Manager().Value("results", None)
        self.name = name

        super().__init__(name=name, target=target, args=(self.results, *args))

    def execute(self):
        self.start()
        self.join()

        if isinstance(self.results.value, list):
            return self.results.value
        
        elif isinstance(self.results.value, ProblemMessage):
            problem_topic.publish(self.results.value)

        elif isinstance(self.results.value, Twist):
            print(self.name)
            if self.name == "turn_right": 
                response = light_signals_service(0, 0, 1)
                response = light_signals_service(1, 0, 1)
                response = light_signals_service(2, 0, 1)
                response = light_signals_service(3, 0, 1)

                col = ColorRGBA()
                col.r = 1
                col.g = 0.6
                col.b = 0
                col.a = 1

                t_end = time.time() + 5
                while time.time() < t_end:

                    # ACTIVATE TURN LIGHTS
                    response = led_segment_service(0, 0, col, 3)
                    response = led_segment_service(0, 1, col, 3)
                    response = led_segment_service(3, 6, col, 3)
                    response = led_segment_service(3, 7, col, 3)
                    response = led_segment_service(2, 14, col, 3)
                    response = led_segment_service(2, 15, col, 3)
                    response = led_segment_service(2, 0, col, 3)
                    response = led_segment_service(2, 1, col, 3)

                    pub_vel.publish(self.results.value)
                pub_vel.publish(Twist())

                # DEACTIVATE OTHER LIGHTS
                response = light_signals_service(0, 2, 1)
                response = light_signals_service(1, 2, 1)
                response = light_signals_service(2, 2, 1)
                response = light_signals_service(3, 2, 1)
            
            elif self.name == "turn_left":
                response = light_signals_service(0, 0, 1)
                response = light_signals_service(1, 0, 1)
                response = light_signals_service(2, 0, 1)
                response = light_signals_service(3, 0, 1)

                col = ColorRGBA()
                col.r = 1
                col.g = 0.6
                col.b = 0
                col.a = 1

                t_end = time.time() + 5
                while time.time() < t_end:

                    # ACTIVATE TURN LIGHTS
                    response = led_segment_service(0, 6, col, 3)
                    response = led_segment_service(0, 7, col, 3)
                    response = led_segment_service(3, 0, col, 3)
                    response = led_segment_service(3, 1, col, 3)
                    response = led_segment_service(1, 14, col, 3)
                    response = led_segment_service(1, 15, col, 3)
                    response = led_segment_service(1, 0, col, 3)
                    response = led_segment_service(1, 1, col, 3)

                    pub_vel.publish(self.results.value)
                pub_vel.publish(Twist())

                # DEACTIVATE OTHER LIGHTS
                response = light_signals_service(0, 2, 1)
                response = light_signals_service(1, 2, 1)
                response = light_signals_service(2, 2, 1)
                response = light_signals_service(3, 2, 1)

            elif self.name == "turn_180":
                response = light_signals_service(0, 0, 1)
                response = light_signals_service(1, 0, 1)
                response = light_signals_service(2, 0, 1)
                response = light_signals_service(3, 0, 1)

                col = ColorRGBA()
                col.r = 1
                col.g = 0.6
                col.b = 0
                col.a = 1
                
                t_end = time.time() + 8.5
                while time.time() < t_end:
                    
                    # ACTIVATE TURN LIGHTS
                    response = led_segment_service(0, 0, col, 3)
                    response = led_segment_service(0, 1, col, 3)
                    response = led_segment_service(3, 6, col, 3)
                    response = led_segment_service(3, 7, col, 3)
                    response = led_segment_service(2, 14, col, 3)
                    response = led_segment_service(2, 15, col, 3)
                    response = led_segment_service(2, 0, col, 3)
                    response = led_segment_service(2, 1, col, 3)


                    pub_vel.publish(self.results.value)
                pub_vel.publish(Twist())
                
                # DEACTIVATE OTHER LIGHTS
                response = light_signals_service(0, 2, 1)
                response = light_signals_service(1, 2, 1)
                response = light_signals_service(2, 2, 1)
                response = light_signals_service(3, 2, 1)

            elif self.name == "stop_task":

                # DEACTIVATE OTHER LIGHTS
                #response = light_signals_service(0, 2, 1)
                response = light_signals_service(1, 2, 1)
                response = light_signals_service(2, 2, 1)
                #response = light_signals_service(3, 2, 1)


                response = light_signals_service(0, 0, 1)
                response = light_signals_service(1, 0, 1)
                response = light_signals_service(2, 0, 1)
                response = light_signals_service(3, 0, 1)

                col = ColorRGBA()
                col.r = 1
                col.g = 0
                col.b = 0
                col.a = 1

                # DEACTIVATE SOUND (if on)
                pub_sound.publish(SoundPlay())

                # ACTIVATE STOP LIGHTS
                response = led_segment_service(3, 0, col, 1)
                response = led_segment_service(3, 1, col, 1)
                response = led_segment_service(3, 2, col, 1)
                response = led_segment_service(3, 3, col, 1)
                response = led_segment_service(3, 4, col, 1)
                response = led_segment_service(3, 5, col, 1)
                response = led_segment_service(3, 6, col, 1)
                response = led_segment_service(3, 7, col, 1)

                pub_vel.publish(self.results.value)
                sleep(3)
                response = light_signals_service(3, 2, 1)


            elif self.name == "block_task":

                # DEACTIVATE OTHER LIGHTS
                #response = light_signals_service(0, 2, 1)
                response = light_signals_service(1, 2, 1)
                response = light_signals_service(2, 2, 1)
                #response = light_signals_service(3, 2, 1)


                response = light_signals_service(0, 0, 1)
                response = light_signals_service(1, 0, 1)
                response = light_signals_service(2, 0, 1)
                response = light_signals_service(3, 0, 1)

                col = ColorRGBA()
                col.r = 1
                col.g = 0
                col.b = 0
                col.a = 1

                # DEACTIVATE SOUND (if on)
                pub_sound.publish(SoundPlay())

                # ACTIVATE STOP LIGHTS
                response = led_segment_service(3, 0, col, 1)
                response = led_segment_service(3, 1, col, 1)
                response = led_segment_service(3, 2, col, 1)
                response = led_segment_service(3, 3, col, 1)
                response = led_segment_service(3, 4, col, 1)
                response = led_segment_service(3, 5, col, 1)
                response = led_segment_service(3, 6, col, 1)
                response = led_segment_service(3, 7, col, 1)

                pub_vel.publish(self.results.value)
                sleep(3)
                response = light_signals_service(3, 2, 1)


            elif self.name == "go_forward":
                response = light_signals_service(0, 0, 1)
                response = light_signals_service(1, 0, 1)
                response = light_signals_service(2, 0, 1)
                response = light_signals_service(3, 0, 1)

                # ACTIVATE FORWARD LIGHTS
                col = ColorRGBA()
                col.r = 0
                col.g = 1
                col.b = 0
                col.a = 1

                # right side
                response = led_segment_service(2, 0, col, 4)
                response = led_segment_service(2, 1, col, 4)
                response = led_segment_service(2, 2, col, 4)
                response = led_segment_service(2, 3, col, 4)
                response = led_segment_service(2, 6, col, 4)
                response = led_segment_service(2, 7, col, 4)
                response = led_segment_service(2, 8, col, 4)
                response = led_segment_service(2, 9, col, 4)
                response = led_segment_service(2, 12, col, 4)
                response = led_segment_service(2, 13, col, 4)
                response = led_segment_service(2, 14, col, 4)
                response = led_segment_service(2, 15, col, 4)

                # left side
                response = led_segment_service(1, 0, col, 4)
                response = led_segment_service(1, 1, col, 4)
                response = led_segment_service(1, 2, col, 4)
                response = led_segment_service(1, 3, col, 4)
                response = led_segment_service(1, 6, col, 4)
                response = led_segment_service(1, 7, col, 4)
                response = led_segment_service(1, 8, col, 4)
                response = led_segment_service(1, 9, col, 4)
                response = led_segment_service(1, 12, col, 4)
                response = led_segment_service(1, 13, col, 4)
                response = led_segment_service(1, 14, col, 4)
                response = led_segment_service(1, 15, col, 4)

                # MOVE
                pub_vel.publish(self.results.value)
               

            elif self.name == "go_backward":
                response = light_signals_service(0, 0, 1)
                response = light_signals_service(1, 0, 1)
                response = light_signals_service(2, 0, 1)
                response = light_signals_service(3, 0, 1)

                # ACTIVATE FORWARD LIGHTS
                col = ColorRGBA()
                col.r = 1
                col.g = 0
                col.b = 0
                col.a = 1

                # right side
                response = led_segment_service(2, 0, col, 4)
                response = led_segment_service(2, 1, col, 4)
                response = led_segment_service(2, 2, col, 4)
                response = led_segment_service(2, 3, col, 4)
                response = led_segment_service(2, 6, col, 4)
                response = led_segment_service(2, 7, col, 4)
                response = led_segment_service(2, 8, col, 4)
                response = led_segment_service(2, 9, col, 4)
                response = led_segment_service(2, 12, col, 4)
                response = led_segment_service(2, 13, col, 4)
                response = led_segment_service(2, 14, col, 4)
                response = led_segment_service(2, 15, col, 4)

                # left side
                response = led_segment_service(1, 0, col, 4)
                response = led_segment_service(1, 1, col, 4)
                response = led_segment_service(1, 2, col, 4)
                response = led_segment_service(1, 3, col, 4)
                response = led_segment_service(1, 6, col, 4)
                response = led_segment_service(1, 7, col, 4)
                response = led_segment_service(1, 8, col, 4)
                response = led_segment_service(1, 9, col, 4)
                response = led_segment_service(1, 12, col, 4)
                response = led_segment_service(1, 13, col, 4)
                response = led_segment_service(1, 14, col, 4)
                response = led_segment_service(1, 15, col, 4)

                # MOVE
                pub_vel.publish(self.results.value)   

                # ACTIVATE SOUND
                sound_msg = SoundPlay()
                sound_msg.sound_index = 2
                sound_msg.loops = 0
                pub_sound.publish(sound_msg)         
                
            #else:
            #    pub_vel.publish(self.results.value)

        elif isinstance(self.results.value, JointTrajectory):

            if self.name == "check":
                # TORSO
                print(self.name)
                t_end = time.time() + 2
                print(t_end)
                while time.time() < t_end:
                    pub_torso.publish(self.results.value)

                torso_msg = JointTrajectory()
                torso_pos_msg = JointTrajectoryPoint()
                torso_msg.joint_names = ['torso_yaw_joint', 'torso_lift_joint', 'head_1_joint', 'head_2_joint']

                torso_pos_msg.positions = [0.0, 0.0, 0.0, 0.0]  
                torso_pos_msg.velocities = [0]
                torso_pos_msg.accelerations = [0]
                torso_pos_msg.effort = [0]
                torso_pos_msg.time_from_start = rospy.Duration.from_sec(10)

                torso_msg.points = [torso_pos_msg]

                pub_torso.publish(torso_msg)

            elif self.name == "harvest":
                # TORSO
                print(self.name)
                t_end = time.time() + 2
                print(t_end)
                while time.time() < t_end:
                    pub_torso.publish(self.results.value)

                torso_msg = JointTrajectory()
                torso_pos_msg = JointTrajectoryPoint()
                torso_msg.joint_names = ['torso_yaw_joint', 'torso_lift_joint', 'head_1_joint', 'head_2_joint']

                torso_pos_msg.positions = [0.0, 0.0, 0.0, 0.0]  
                torso_pos_msg.velocities = [0]
                torso_pos_msg.accelerations = [0]
                torso_pos_msg.effort = [0]
                torso_pos_msg.time_from_start = rospy.Duration.from_sec(10)

                torso_msg.points = [torso_pos_msg]

                pub_torso.publish(torso_msg)

            if self.name == "prune":
                # TORSO
                print(self.name)
                t_end = time.time() + 2
                print(t_end)
                while time.time() < t_end:
                    pub_torso.publish(self.results.value)

                torso_msg = JointTrajectory()
                torso_pos_msg = JointTrajectoryPoint()
                torso_msg.joint_names = ['torso_yaw_joint', 'torso_lift_joint', 'head_1_joint', 'head_2_joint']

                torso_pos_msg.positions = [0.0, 0.0, 0.0, 0.0]  
                torso_pos_msg.velocities = [0]
                torso_pos_msg.accelerations = [0]
                torso_pos_msg.effort = [0]
                torso_pos_msg.time_from_start = rospy.Duration.from_sec(10)

                torso_msg.points = [torso_pos_msg]

                pub_torso.publish(torso_msg)


        return None

"""

    TASKS

"""

##### CHECK ##### 

def check_grape(results, task_uuid: str, args: 'list[str]'):
    """Check grape and it's given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    print(task_uuid)

    rospy.loginfo("Doing stuff")

    sleep(10)

    rospy.loginfo("Doing some other stuff, OOPS PROBLEM")

    if True:
        problem_message = ProblemMessage()
        problem_message.severity = 3
        problem_message.problem_name = "Grape disappeared"
        problem_message.problem_sentence = "The grape I was trying to check disappeared"
        problem_message.task_uuid = task_uuid

        results.value = problem_message
        return

    sleep(10)
    rospy.loginfo("Done stuff!")

    results.value = ["ripe"]
    return

def check_temperature(results, task_uuid: str, args: 'list[str]'):
    """Check temperature and it's given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    print(task_uuid)

    rospy.loginfo("Doing stuff")

    sleep(10)
    rospy.loginfo("Done stuff!")

    results.value = ["32"]
    return


def check_ground(results, task_uuid: str, args: 'list[str]'):
    """Check ground and it's given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    print(task_uuid)

    rospy.loginfo("Doing stuff")

    sleep(10)
    rospy.loginfo("Done stuff!")

    results.value = ["wet"]
    return


##### STOP #####
def stop(results, task_uuid: str, args: 'list[str]'):
    """Stop and its given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    twist_msg = Twist()

    rospy.loginfo("Robot stopping!")

    results.value = twist_msg
    return


##### BLOCK #####
def block(results, task_uuid: str, args: 'list[str]'):
    """Block and its given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    twist_msg = Twist()

    rospy.loginfo("Robot stopping!")

    results.value = twist_msg
    return


##### GO #####

def go_forward(results, task_uuid: str, args: 'list[str]'):
    """Go forward and its given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    print('***********')
    print(args)
    print('***********')

    twist_msg = Twist()
    twist_msg.linear.x = 0.4
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = 0.0

    rospy.loginfo("Robot moving forward!")

    results.value = twist_msg

    return




def go_backward(results, task_uuid: str, args: 'list[str]'):
    """Go backward and its given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    print('***********')
    print(args)
    print('***********')

    twist_msg = Twist()
    twist_msg.linear.x = -0.4
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = 0.0

    rospy.loginfo("Robot moving backward!")

    results.value = twist_msg

    return


def turn_right(results, task_uuid: str, args: 'list[str]'):
    """Turn right and its given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    print('***********')
    print(args)
    print('***********')

    twist_msg = Twist()
    twist_msg.linear.x = 0.0
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = -0.32

    rospy.loginfo("Turning right!")

    results.value = twist_msg

    return


def turn_left(results, task_uuid: str, args: 'list[str]'):
    """Turn left and its given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    print('***********')
    print(args)
    print('***********')

    twist_msg = Twist()
    twist_msg.linear.x = 0.0
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = 0.32

    rospy.loginfo("Turning left!")

    results.value = twist_msg

    return


def turn_180(results, task_uuid: str, args: 'list[str]'):
    """Turn 180 and its given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    print('***********')
    print(args)
    print('***********')

    twist_msg = Twist()
    twist_msg.linear.x = 0.0
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = -0.36

    rospy.loginfo("Turning around!")

    results.value = twist_msg

    return


def check(results, task_uuid: str, args: 'list[str]'):
    """Check and its given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    # open robot's camera

    print('***********')
    print(args)
    print('***********')

    torso_msg = JointTrajectory()
    torso_pos_msg = JointTrajectoryPoint()
    torso_msg.joint_names = ['torso_yaw_joint', 'torso_lift_joint', 'head_1_joint', 'head_2_joint']

    torso_pos_msg.positions = [0.0, 0.27, 0.0, 0.0]  #[0.0,0.27,0.0,0.6]
    torso_pos_msg.velocities = [0]
    torso_pos_msg.accelerations = [0]
    torso_pos_msg.effort = [0]
    torso_pos_msg.time_from_start = rospy.Duration.from_sec(7)

    torso_msg.points = [torso_pos_msg]

    rospy.loginfo("Doing the check!")

    results.value = torso_msg

    return


def harvest(results, task_uuid: str, args: 'list[str]'):
    """Check and its given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    # open robot's camera

    print('***********')
    print(args)
    print('***********')

    torso_msg = JointTrajectory()
    torso_pos_msg = JointTrajectoryPoint()
    torso_msg.joint_names = ['torso_yaw_joint', 'torso_lift_joint', 'head_1_joint', 'head_2_joint']

    torso_pos_msg.positions = [0.0, 0.27, 0.0, 0.0]  #[0.0,0.27,0.0,0.6]
    torso_pos_msg.velocities = [0]
    torso_pos_msg.accelerations = [0]
    torso_pos_msg.effort = [0]
    torso_pos_msg.time_from_start = rospy.Duration.from_sec(7)

    torso_msg.points = [torso_pos_msg]

    rospy.loginfo("Doing the harvesting!")

    results.value = torso_msg

    return



def prune(results, task_uuid: str, args: 'list[str]'):
    """Check and its given parameters

    Args:
        args (list[str]): List of arguments for this function

    Returns:
        list[str]: The task results
    """

    # open robot's camera

    print('***********')
    print(args)
    print('***********')

    torso_msg = JointTrajectory()
    torso_pos_msg = JointTrajectoryPoint()
    torso_msg.joint_names = ['torso_yaw_joint', 'torso_lift_joint', 'head_1_joint', 'head_2_joint']

    torso_pos_msg.positions = [0.0, 0.27, 0.0, 0.0]  #[0.0,0.27,0.0,0.6]
    torso_pos_msg.velocities = [0]
    torso_pos_msg.accelerations = [0]
    torso_pos_msg.effort = [0]
    torso_pos_msg.time_from_start = rospy.Duration.from_sec(7)

    torso_msg.points = [torso_pos_msg]

    rospy.loginfo("Doing the pruning!")

    results.value = torso_msg

    return



tasks_definitions = {
    #"check_grape": check_grape,
    #"check_temperature": check_temperature,
    #"check_ground": check_ground,

    #"go_left": go_left,
    #"go_right": go_right,
    "stop_task": stop,
    "block_task": block,
    "go_forward": go_forward,
    "go_backward": go_backward,
    "turn_right": turn_right,
    "turn_left": turn_left,
    "turn_180": turn_180,
    "check": check,
    "harvest": harvest,
    "prune": prune
}
"""
Associates a given string to a local function
"""


"""

    LOGS

"""

class LogType(Enum):
    """
    Represent the type for a single task log
    """

    ADD_TASK = "add_task"
    REMOVE_TASK = "remove_task"
    LIST_TASKS = "list_tasks"
    TASK_START= "task_start"
    TASK_END = "task_end"
    TASK_PROBLEM = "task_problem"


class TaskLog():
    """
    Represent a single task log
    """
    def __init__(self, log_type: LogType, task_uuid: str, info = None, additional = None):
        self.log_type = log_type
        self.datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.info = info
        self.additional = additional
        self.task_uuid = task_uuid



class Task:
    """
    Represent a single task
    """
    def __init__(self, task_name: str, args: 'list[str]'):
        self.name = task_name
        self.args = args
        self.method = tasks_definitions[task_name]
        self.uuid = str(uuid.uuid4())


"""

    MANAGER

"""

class TaskManager:
    def __init__(self):
        # Variables
        self.tasks: list[Task] = []
        self.current_task: TaskThread = None

        # Rate
        self.rate = rospy.Rate(2)

        # Create service
        self.add_task_service = rospy.Service('add_task', AddTask, self.add_task)
        self.kill_all_tasks_service = rospy.Service('kill_all_tasks', KillAllTasks, self.kill_all_tasks)

        # Finish task service
        self.finish_task = rospy.ServiceProxy('finish_task', TaskResult)

        # TODO: call light and sound topic
        #rospy.wait_for_service('/canopies_simulator/robot00/light_signal')
        #self.light_signals_service = rospy.ServiceProxy('/canopies_simulator/robot00/light_signal', light_signals)


        # Main runtime
        self.manage_tasks()



    def log_task(self, task_log: TaskLog):
        """Write the given log to the log file

        Args:
            task_log (TaskLog): Log to write in the log file
        """
        write_to_log_file(log_file_name, (task_log.datetime, task_log.log_type.name, task_log.task_uuid, task_log.info, task_log.additional))



    def add_task(self, request: AddTaskRequest) -> str:
        """Adds the given task to the task list

        Args:
            request (AddTaskRequest): Task to add to the task list

        Returns:
            str: New task UUID
        """

        rospy.loginfo(f"New task: {request.task_name}")

        # Create the new task
        task = Task(request.task_name, request.args)

        # Append the task to the task list
        self.tasks.append(task)

        # Log the adding of the new task
        self.log_task(TaskLog(LogType.ADD_TASK, task.uuid, task.name, task.args))
        
        response = AddTaskResponse()
        response.uuid = task.uuid

        return response



    def kill_all_tasks(self, _: KillAllTasksRequest):
        print("Killing current task")

        # If there is a current task
        if self.current_task is not None:
            # Kill the current task
            self.current_task.kill()

        # If the task list contains more than one task
        if len(self.tasks) > 1:
            # For each task after the first one (because the first one is gonna be deleted anyway in the )
            for task in self.tasks[1:]:
                # Remove the task from the task list
                self.tasks.remove(task)

        return KillAllTasksResponse()




    def manage_tasks(self):
        """
        Main runtime to manage the task list
        """

        rospy.loginfo("Started")

        # While the node isn't shuted down
        while not rospy.is_shutdown():


            # If there is a least one task ine the task list
            if len(self.tasks) > 0:
                # Get the first task
                task = self.tasks[0]

                # Log the task list
                self.log_task(TaskLog(LogType.LIST_TASKS, task.uuid))

                # Log the task start
                # task name - args - start
                self.log_task(TaskLog(LogType.TASK_START, task.uuid, task.name, task.args))

                # Create threaded task
                self.current_task = TaskThread(name=task.name, target=task.method, args=(task.uuid, task.args,))

                # Execute and get task result
                results = self.current_task.execute()

                # If there is no results (In case a problem happened)
                if results is None:
                    # Log the task problem
                    # task name - results - problem
                    self.log_task(TaskLog(LogType.TASK_PROBLEM, task.uuid, task.name))

                # Otherwise, if there is results
                else:
                    # Tell the decision manager that the task is finished
                    self.finish_task(task.uuid, results)

                    # Log the task end
                    # task name - results - end
                    self.log_task(TaskLog(LogType.TASK_END, task.uuid, task.name, results))

                # Log the task which is being removed from the list
                self.log_task(TaskLog(LogType.REMOVE_TASK, task.uuid, task.name))

                # Remove the task from the list of tasks
                self.tasks.remove(task)


            # Sleep at node rate
            self.rate.sleep()


"""

    MAIN

"""

if __name__ == "__main__":
    # Name of the log file
    log_file_name = "task_logs.csv"

    create_csv_with_headers_if_does_not_exist(log_file_name, ("datetime", "log_type", "task_uuid", "info", "additional"))

    # Init node
    rospy.init_node("task_manager_node", anonymous=True)

    # Publishers
    problem_topic = rospy.Publisher('problem_topic', ProblemMessage, queue_size=10)
    pub_vel = rospy.Publisher('/canopies_simulator/moving_base/twist', Twist, queue_size=10)
    #pub_left_joint_traj = rospy.Publisher('/canopies_simulator/arm_left_controller/command', JointTrajectory, queue_size=10)
    #pub_right_joint_traj = rospy.Publisher('/canopies_simulator/arm_right_controller/command', JointTrajectory, queue_size=10)
    pub_torso = rospy.Publisher('/canopies_simulator/torso_controller/command', JointTrajectory, queue_size=10)
    pub_sound = rospy.Publisher('/canopies_simulator/sound_play', SoundPlay, queue_size=10) 

    rospy.wait_for_service('/canopies_simulator/robot00/light_signal')
    light_signals_service = rospy.ServiceProxy('/canopies_simulator/robot00/light_signal', light_signals)

    rospy.wait_for_service('/canopies_simulator/robot00/led_segment')
    led_segment_service = rospy.ServiceProxy('/canopies_simulator/robot00/led_segment', led_segment)


    TaskManager()