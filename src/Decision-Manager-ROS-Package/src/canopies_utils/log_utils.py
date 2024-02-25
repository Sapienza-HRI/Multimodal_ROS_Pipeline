#!/usr/bin/env python3

import csv
import os

log_folder_path = os.path.join(os.getcwd(), "canopies_logs")

def create_csv_with_headers_if_does_not_exist(file_name: str, headers: tuple):
    # Join folder path and file name 
    file_path = os.path.join(log_folder_path, file_name)

    # If the file doesn't exist
    if not os.path.exists(file_path):
        # Create the file and write the header
        write_to_log_file(file_name, headers, "w+")

def write_to_log_file(file_name: str, data: tuple, open_mode: str = "a"):
    """Write the given data to the log file

    Args:
        file_name (str): log file name
        data (tuple): data to write
        open_mode (str, optional): File open mode. Defaults to "a".
    """
    # Join folder path and file name 
    file_path = os.path.join(log_folder_path, file_name)

    # Open the file
    stream = open(file_path, open_mode)

    log_file = csv.writer(stream, delimiter=";", quoting=csv.QUOTE_NONE, escapechar="\\")

    # Write the data to file
    log_file.writerow(data)

    # Close the file
    stream.close()