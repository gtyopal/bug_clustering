# Import Project Source Files
import loginfo


class BugLogInfo:
    """This is a class for a bug and its respected log information.

        Args:
            bug_id (str): The id of the bug.
            list_log_info (list): List of log information related to the bug.
    """
    def __init__(self, bug_id, list_log_info=[loginfo.LogInfo]):
        self.bug_id = bug_id
        self.list_log_info = list_log_info
