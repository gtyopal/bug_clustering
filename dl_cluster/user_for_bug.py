class BugUser:
    """This is a class for a user.

        Args:
            user_email (str): The email of the user.
    """
    def __init__(self, user_email):
        self.user_email = user_email

    def __repr__(self):
        return "{'email': '" + self.user_email + "'}"
