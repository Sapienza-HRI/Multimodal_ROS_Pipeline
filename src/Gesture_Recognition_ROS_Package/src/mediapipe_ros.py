#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, Point
import rospkg


import mediapipe as mp
import numpy as np
import tensorflow as tf
from time import time
import os
import sys

from grmodule.msg import Pose
from dialog_manager.msg import FrameMessage
from decision_manager.msg import FrameGroupMessage


# Constants for model paths and data class map
CNN_PATH = 'src/Gesture_module_ROS_Package/gesture_models/cnn_models/mediapipe_100vd_100rd_pose_classifier.tflite'
DATA_CLASS_MAP = {
    0: 'attention', 
    1: 'cut_2', 
    2: 'move_left', 
    3: 'move_right', 
    4: 'pause', 
    5: 'resume', 
    6: 'standing', 
    7: 'start', 
    8: 'stop', 
    9: 'terminate_all', 
    10: 'turn_left', 
    11: 'turn_right', 
    12: 'unknown'
    }

class MediapipePoseEstimationNode:

    def __init__(self):
        self.flag_map = {
            'attention': False,
            'cut': False,
            'move_left': False,
            'move_right': False,
            'pause' : False, 
            'resume' : False,          
            'start': False, 
            'stop':False, 
            'terminate_all': False, 
            'turn_left': False, 
            'turn_right': False, 
        }
        self.flag_map_flush = self.flag_map
        #self.image_sub = rospy.Subscriber('/canopies_simulator/realsense_right/color/image_raw', Image, self.camera_callback)
        self.image_sub = rospy.Subscriber('/webcam/video_frame', Image, self.camera_callback)
        #self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.camera_callback)
        self.pose_pub = rospy.Publisher('/pose_estimation/mediapipe_pose', Pose, queue_size=10)
        self.motion_pub = rospy.Publisher('/canopies_simulator/moving_base/twist', Twist, queue_size=10)
        self.decision_manager_pub = rospy.Publisher('/decision_manager_topic_gesture', FrameGroupMessage, queue_size=10)

        self.bridge = CvBridge()
        self.motion_msg = Twist()
        self.frame_msg = FrameMessage()
        self.frame_group_msg = FrameGroupMessage()
        # Initializing mediapipe pose and holistic class.
        self.mp_pose = mp.solutions.pose

        # Initializing mediapipe drawing class, useful for annotation.
        self.mp_drawing = mp.solutions.drawing_utils 
        self.mp_drawing_styles = mp.solutions.drawing_styles

     # Define a function to show the image in an OpenCV Window
    def show_image(self, img):
        cv2.imshow("Mediapipe Pose Detection", img)
        cv2.waitKey(3)
    
    def set_motion_params(self, x, y, z, ax,ay,az):
        self.motion_msg.linear.x = x
        self.motion_msg.linear.y = y
        self.motion_msg.linear.z = z
        self.motion_msg.angular.x = ax
        self.motion_msg.angular.y = ay
        self.motion_msg.angular.z = az
        self.motion_pub.publish(self.motion_msg)
        


    def set_dialogue_params(self, sentence, roles, words):
        gest_frames: list[FrameMessage] = []

        self.frame_msg = FrameMessage()

        self.frame_msg.full_sentence = sentence
        self.frame_msg.language = "italian"
        self.frame_msg.frame_roles = roles
        self.frame_msg.frame_words = words
        self.frame_msg.speech_act = "Command"

        gest_frames.append(self.frame_msg)
        '''print('-----------')
        print(self.frame_msg.frame_roles)
        print('-----------')
        print(self.frame_msg.frame_words)
        print('-----------')
        print(gest_frames)
        print('?????')'''
                
        self.frame_group_msg = FrameGroupMessage()
        self.frame_group_msg.frames = gest_frames

        self.decision_manager_pub.publish(self.frame_group_msg)



    def detect(self, input_tensor, inference_count=3):
        """Runs detection on an input image.

        Args:
            input_tensor: A [height, width, 3] Tensor of type tf.float32.
            Note that height and width can be anything since the image will be
            immediately resized according to the needs of the model within this
            function.
            inference_count: Number of times the model should run repeatedly on the
            same input image to improve detection accuracy.

        Returns:
            A Person entity detected by the MoveNet.SinglePose.
        """
        image_height, image_width, channel = input_tensor.shape
        output_image = input_tensor.copy()
        # Detect pose using the full input image
        pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

        # Detect pose using the full input image
        pose.process(input_tensor)

        # Repeatedly using previous detection result to identify the region of
        # interest and only croping that region to improve detection accuracy
        for _ in range(inference_count - 1):
            result = pose.process(input_tensor)

        self.mp_drawing.draw_landmarks(output_image,result.pose_landmarks,self.mp_pose.POSE_CONNECTIONS)

        return output_image, result
    
    
    def mediapipe_evaluate_model(self, landmarks, class_index_to_label):
        interpreter = tf.lite.Interpreter(model_path=CNN_PATH)

        # Allocate tensors for the model's input and output
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Print input and output details (optional)
        # print("Input details:", input_details)
        # print("Output details:", output_details)

        # Run inference (you'll need to prepare input data)
        input_data = np.array(landmarks, dtype=np.float32)  # Prepare your input data as a NumPy array
        input_data = input_data.reshape(1,132)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        class_index = np.argmax(output_data)
        # Process the output data (e.g., post-processing)
        label = DATA_CLASS_MAP[class_index]
        prediction_accuracy = float(output_data[0][class_index]*100)

        # Print the results (optional)
        # print("Inference result:", output_data)
        return label, prediction_accuracy
    


    def process_gesture_label(self, final_label):
        if final_label == "start" and final_label not in ["", 'standing', 'unknown']:
            print(final_label)
            self.set_dialogue_params("Muoviti in avanti.", ["GO", "Direction"], ["GO", "avanti"])
            if not self.flag_map['start']:
                self.set_motion_params(0.4, 0.0, 0.0, 0.0, 0.0, 0.0)
                self.flag_map['start'] = True

        elif final_label == "terminate_all" and final_label not in ["", 'standing', 'unknown']:
            print(final_label)
            self.set_dialogue_params("Blocca esecuzione.", ["STOP", "Theme"], ["STOP", "movimento"])

            if not self.flag_map['terminate_all']:
                self.set_motion_params(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                self.flag_map['terminate_all'] = True

        elif final_label == "move_left" and final_label not in ["", 'standing', 'unknown']:
            print(final_label)
            self.set_dialogue_params("Gira a sinistra.", ['TURN', 'Direction'], ['TURN', 'sinistra'])

            if not self.flag_map['turn_left']:
                self.set_motion_params(0.0, 0.0, 0.0, 0.0, 0.0, 0.3)
                self.flag_map['turn_left'] = True

        elif final_label == "move_right" and final_label not in ["", 'standing', 'unknown']:
            print(final_label)
            self.set_dialogue_params("Gira a destra.", ["TURN", "Direction"], ["TURN", "destra"])
            if not self.flag_map['turn_right']:
                self.set_motion_params(0.0, 0.0, 0.0, 0.0, 0.0, -0.3)
                self.flag_map['turn_right'] = True

        elif final_label == "stop" and final_label not in ["", 'standing', 'unknown']:
            print(final_label)
            self.set_dialogue_params("Fermati.", ["STOP"], ["STOP"])
            if not self.flag_map['stop']:
                self.set_motion_params(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                self.flag_map['stop'] = True

        elif final_label == "standing":
            print(final_label)
            self.set_motion_params(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self.flag_map = dict.fromkeys(self.flag_map, False)


    def visualize(self, image, keypoints):
        # Convert keypoints to a format compatible with cv2
        points = []
        for landmark in keypoints.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            points.append((x, y))

        # Draw circles on the keypoints
        for point in points:
            cv2.circle(image, point, 5, (0, 255, 0), -1)

        # Draw lines between the keypoints
        connections = mp.solutions.pose.POSE_CONNECTIONS
        for connection in connections:
            start_point = points[connection[0]]
            end_point = points[connection[1]]
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

        return image


    def camera_callback(self, image_msg):
        # log some info about the image topic
        rospy.loginfo(image_msg.header)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")

            # Convert the image to RGB format
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Flip the image horizontally (for example, to create a mirror image)
            cv_image_flipped = cv2.flip(cv_image_rgb, 1)
            
            # Get the width and height of the frame
            frame_height, frame_width, _ =  cv_image_flipped.shape

            # # Create named window for resizing purposes
            # cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

            # Perform your pose estimation here 
            # Initialize a variable to store the time of the previous frame.
            time1 = 0

            # Create a Pose message to publish the pose data
            pose_msg = Pose() # perform Pose landmark detection.
            frame, results = self.detect(cv_image_flipped)
            if results.pose_landmarks != None:

                # Get landmarks and scale it to the same size as the input image
                pose_landmarks = np.array(
                            [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width,lmk.visibility]
                                for lmk in results.pose_landmarks.landmark], dtype=np.float32)
                coordinates = pose_landmarks.flatten().astype(np.float64).tolist()
                label, accuracy = self.mediapipe_evaluate_model(coordinates, DATA_CLASS_MAP)
                if label not in ['unknown']:
                    cv2.putText(frame, label, (10, 650),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                    pose_msg.pose_class = label
                    pose_msg.class_accuracy = accuracy
                    self.process_gesture_label(label)
                self.show_image(frame)

            else:
                cv2.putText(frame, 'No Pose detected', (10, 650),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                pose_msg.pose_class = 'No_Pose'
                pose_msg.class_accuracy = 0.0


            # Publish the pose data
            self.pose_pub.publish(pose_msg)

        except Exception as e:
            rospy.logerr("Error in camera_callback: %s" % str(e))

if __name__ == '__main__':
    rospack = rospkg.RosPack()
    path = rospack.get_path("grmodule")
    sys.path.append(path)
    rospy.init_node('mediapipe_node', anonymous=True)
    MediapipePoseEstimationNode()
    rospy.spin()
