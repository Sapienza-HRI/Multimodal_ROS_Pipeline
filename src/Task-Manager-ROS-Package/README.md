CANOPIES Task Manager ROS Package
===

This package provides a Task Manager system for ROS in the CANOPIES context.
 
## Requirements

The log function called `create_csv_with_headers_if_does_not_exist` and `write_to_log_file` comes from the `decision_manager` package in `/src/canopies_utils`.

## Service

The node can be used with the `add_task` and `kill_all_tasks` services.

The service `add_task` accepts `AddTask` (in the `srv` folder), like follow:

```c
string task_name
string[] args
---
string uuid
```

The service `kill_all_tasks` accepts `KillAllTasks` (in the `srv` folder), which is empty.

> Don't forget to build and source your code!

## Run

You can run the node with the following command:

```shell
rosrun task_manager task_manager_runtime.py
```