# Import Project Source Files
import rel_bug


class BugRelBugs:
    """This is a class for a bug and its related bugs.

        Args:
            bug_id (str): The id of the bug.
            list_of_rel_bugs (list): List of related bugs to the bug.
    """
    def __init__(self, bug_id, list_of_rel_bugs=[rel_bug.RelBug]):
        self.bug_id = bug_id
        self.list_of_rel_bugs = list_of_rel_bugs
