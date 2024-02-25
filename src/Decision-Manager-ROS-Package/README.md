CANOPIES Decision Manager ROS Package
===

This package provides a Decision Manager system for ROS in the CANOPIES context.
 
## Requirements

The log function called `create_csv_with_headers_if_does_not_exist` and `write_to_log_file` comes from the package in `/src/canopies_utils`.

## Service

The node can be used with the `finish_task` service and the `decision_manager_topic` and `issue_handler_topic` topics.

The service `finish_task` accepts `TaskResult` (in the `srv` folder), like follow:

```c
string task_uuid
string[] results
```

The service `decision_manager_topic` accepts `FrameGroupMessage` (in the `msg` folder), like follow:

```c
dialog_manager/FrameMessage[] frames
```

The service `issue_handler_topic` accepts `IssueMessage` (in the `msg` folder), like follow:

```c
int16 severity
string issue_name
string issue_sentence
string task_uuid
```

> Don't forget to build and source your code!

## Run

You can run the node with the following command:

```shell
rosrun decision_manager decision_manager_runtime.py
```
