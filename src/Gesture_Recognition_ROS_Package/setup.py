#import os
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Add the easy_ViTPose folder path to the PYTHONPATH environment variable
#os.environ['PYTHONPATH'] += ':/home/sara/Speech_Pipeline_CANOPIES/workspace/src/Gesture_module_ROS_Package'

# Use the generate_distutils_setup function to generate the setup arguments
setup_args = generate_distutils_setup(
    packages=['movenet', 'movenet.*','cnn_models', 'cnn_models.*'],   # 'easy_ViTPose', 'easy_ViTPose.*', 
    package_dir={'': 'src'},
)

setup(**setup_args)

