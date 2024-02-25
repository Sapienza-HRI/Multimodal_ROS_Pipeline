CANOPIES Text-To-Speech ROS Package
===

This package provides a TTS system for ROS.

## Topic

The node can be used with the `/text_to_speech_topic` topic.

The topic accepts `TTSMessage` (in the `msg` folder), like follow:

```c
string type
string lang
string[] sentences
```

> Don't forget to build and source your code!

## Run

You can run the node with the following command:

```shell
rosrun text_to_speech text_to_speech_script.py
```

## Additional

This packages also provides the `initialize_library.py` script which will generates the required file system architecture to use the node. Please first check its suitability in your case.