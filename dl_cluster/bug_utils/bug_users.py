# Import Project Source Files
from bug_utils import user_for_bug


class BugUsers:
    """This is a class for a bug and its respected users.

        Args:
            bug_id (str): The id of the bug.
            list_of_users (list): List of users related to the bug.
    """
    def __init__(self, bug_id, list_of_users=[user_for_bug.BugUser]):
        self.bug_id = bug_id
        self.list_of_users = list_of_users
