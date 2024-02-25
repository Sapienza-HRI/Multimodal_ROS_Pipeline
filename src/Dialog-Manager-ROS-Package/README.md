CANOPIES Dialog Manager ROS Package
===

This package provides a Dialog Manager system for ROS in the CANOPIES context.
 
## Requirements

The log function called `create_csv_with_headers_if_does_not_exist` and `write_to_log_file` comes from the `decision_manager` package in `/src/canopies_utils`.

## Service

The node can be used with the `handle_frame_service`, `handle_suggestion_service` and `tts_service` services.

The service `handle_frame_service` accepts `HandleFrame` (in the `srv` folder), like follow:

```c
FrameMessage frame
---
string speaker_uuid
string result_sentence
string task_name
string[] args
```

The service `handle_suggestion_service` accepts `HandleSuggestion` (in the `srv` folder), like follow:

```c
FrameMessage frame
---
string suggestion_result
```

The service `tts_service` accepts `TTSRequest` (in the `srv` folder), like follow:

```c
string command
string type
string[] sentences
string speaker_uuid
```

And also the `FrameMessage` (in the `msg` folder), like follow:

```c
string full_sentence
string[5] frame
string speech_act
```

> Don't forget to build and source your code!

## Run

You can run the node with the following command:

```shell
rosrun dialog_manager dialog_manager_script.py
```