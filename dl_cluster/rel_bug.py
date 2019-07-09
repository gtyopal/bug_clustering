class RelBug:
    """This is a class for a related bug.

        Args:
            bug_id (str): The id of the bug.
            keywords (str): The keywords for the related bug.
            de_mgr (str): The Data Engineer's Manager.
            de (str): The Developer
    """
    def __init__(self, bug_id, keywords, bug_distance, key_feature, key_phrase, description, severity, de_mgr, de):
        self.bug_id = bug_id
        self.keywords = keywords
        self.bug_distance = bug_distance
        self.key_feature = key_feature
        self.key_phrase = key_phrase
        self.description = description
        self.severity = severity
        self.de_mgr = de_mgr
        self.de = de

    def __repr__(self):
        return "{'bug_id': " + self.bug_id + \
               ", 'keywords': " + self.keywords + \
               ", 'bug_distance': " + self.bug_distance + \
               ", 'key_feature': " + self.key_feature + \
               ", 'key_phrase': " + self.key_phrase + \
               ", 'description': " + self.description + \
               ", 'severity': " + self.severity + \
               ", 'de_mgr': " + self.de_mgr + \
               ", 'de': " + self.de + "}"
