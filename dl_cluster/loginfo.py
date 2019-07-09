class LogInfo:
    """This is a class for log information.

        Args:
            log_name (str): The name of the log file.
            err_msg (str): The error message in the log file.
            file_path (str): The path to the source file with the error.
    """
    def __init__(self, log_name, err_msg, file_path):
        self.log_name = log_name
        self.err_msg = err_msg
        self.file_path = file_path

    def __repr__(self):
        return "{'log_name': '" + self.log_name + \
               "', 'error_msg': '" + self.err_msg + \
               "', 'file_path': '" + self.file_path + "'}"
