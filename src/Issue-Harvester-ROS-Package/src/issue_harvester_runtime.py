#!/usr/bin/env python3

import rospy
import time
from datetime import datetime
from canopies_utils.log_utils import create_csv_with_headers_if_does_not_exist, write_to_log_file
from issue_harvester.msg import ErrorMessage, ProblemMessage
from decision_manager.msg import IssueMessage


"""

    ISSUE

"""

class Issue:
    """
    Represent an issue
    """
    def __init__(self, severity: str, name: str, sentence: str, task_uuid: str = None):
        self.severity = severity
        self.name = name
        self.sentence = sentence
        self.task_uuid = task_uuid

"""

    LOG

"""

class IssueLog():
    def __init__(self, issue: Issue, additional_data: str):
        self.datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.severity = issue.severity
        self.name = issue.name
        self.sentence = issue.sentence
        self.task_uuid = issue.task_uuid
        self.additional_data = additional_data

"""

    MANAGER

"""

class IssueHarvester:
    def __init__(self):
        # Variables
        self.issues: list[Issue] = []

        # Init node
        rospy.init_node("issue_harvester_node", anonymous=True)

        # Rate
        self.rate = rospy.Rate(2)

        # Publishers
        self.handle_issue = rospy.Publisher('issue_handler_topic', IssueMessage, queue_size=10)

        # Subscribers
        self.add_error = rospy.Subscriber('error_topic', ErrorMessage, self.add_error, queue_size=10)
        self.add_problem = rospy.Subscriber('problem_topic', ProblemMessage, self.add_problem, queue_size=10)

        time_start = datetime.now()

        # Main runtime
        self.harvest_issues()

        '''
        print("***********************")
        print('Time required to IH is:')
        print(datetime.now() - time_start)
        print("***********************")
        '''


    def log_issue(self, issue_log: IssueLog):
        """Write the given log to the log file

        Args:
            issue_log (IssueLog): Log to write in the log file
        """
        write_to_log_file(log_file_name, (issue_log.datetime, issue_log.severity, issue_log.name, issue_log.additional_data, issue_log.sentence, issue_log.task_uuid))



    def harvest_issues(self):
        """
        Main runtime to manage the issue list
        """

        rospy.loginfo("Started")

        # While the node isn't shuted down
        while not rospy.is_shutdown():
            for issue in self.issues:
                # Create the issue message
                issue_message = IssueMessage()
                issue_message.severity = issue.severity
                issue_message.issue_name = issue.name
                issue_message.issue_sentence = issue.sentence
                issue_message.task_uuid = "" if issue.task_uuid == None else issue.task_uuid

                # Publish the issue message
                self.handle_issue.publish(issue_message)

                # Remove the processed issue from the issue list
                self.issues.remove(issue)

            #print("Getting battery percentage")
            #print("Getting computed storage capacity")
            #print("Getting box storage capacity")

            # Sleep at node rate
            self.rate.sleep()

    def add_error(self, error: ErrorMessage):
        # Create the issue
        issue = Issue(error.severity, error.error_name, error.error_sentence)

        # Log
        self.log_issue(IssueLog(issue, error.long_error))

        # Added the issue to the issues list
        self.issues.append(issue)

    def add_problem(self, problem: ProblemMessage):
        print("Received problem:", problem.severity, problem.problem_name)
        # Create the issue
        issue = Issue(problem.severity, problem.problem_name, problem.problem_sentence, problem.task_uuid)

        # Log
        self.log_issue(IssueLog(issue, None))

        # Added the issue to the issues list
        self.issues.append(issue)

"""

    MAIN

"""

if __name__ == "__main__":
    # Name of the log file
    log_file_name = "issue_logs.csv"

    create_csv_with_headers_if_does_not_exist(log_file_name, ("datetime", "severity", "name", "additional_data", "sentence", "task_uuid"))

    IssueHarvester()