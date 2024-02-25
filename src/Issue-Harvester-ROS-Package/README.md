CANOPIES Issue Harvester ROS Package
===

This package provides an Issue Harvester system for ROS in the CANOPIES context.

## Requirements

The log function called `create_csv_with_headers_if_does_not_exist` and `write_to_log_file` comes from the `decision_manager` package in `/src/canopies_utils`.

## Topic

The node can be used with the `/error_topic` and `/problem_topic` topics.

The `error_topic` topic accepts `ErrorMessage` (in the `msg` folder), like follow:

```c
int16 severity
string error_name
string error_sentence
string long_error
```

The `problem_topic` topic accepts `ProblemMessage` (in the `msg` folder), like follow:

```c
int16 severity
string problem_name
string problem_sentence
string task_uuid
```

> Don't forget to build and source your code!

## Run

You can run the node with the following command:

```shell
rosrun issue_harvester issue_harvester_runtime.py
```
