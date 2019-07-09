class BugReproStep:
    """This is a class for a bug and its respected reproduction steps.

        Args:
            bug_id (str): The id of the bug.
            repro_step (str): The reproduction steps of the bug.
    """
    def __init__(self, bug_id, repro_step):
        self.bug_id = bug_id
        self.repro_step = repro_step
