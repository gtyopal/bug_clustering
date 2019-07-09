# Import Standard Python Libraries
import json
import ast

# Import Project Source Files
from bug_utils import bug_reprostep
from bug_utils import bug_loginfo
from bug_utils import bug_rel_bugs
from bug_utils import rel_bug
from bug_utils import loginfo
from bug_utils import user_for_bug
from bug_utils import bug_users


def write_json_to_file(json_list, file_name):
    """Function writes a list of Objects to a file in JSON format.

        Args:
            json_list (list): The list of objects that will be converted into JSON.
            file_name (str): The name of the JSON file the list will be written to.
    """
    json_data = json.dumps(json_list, default=lambda o: o.__dict__)
    # lit_json_data = ast.literal_eval(json_data)
    with open(file_name, 'w') as outfile:
        json.dump(json_data, outfile)


def read_json_from_file(file_name):
    """Function reads a JSON file and returns a list of Objects based on the file given.

        Args:
            file_name (str): The name of the JSON file the list will be read from.
    """
    with open(file_name) as infile:
        in_data = json.load(infile)
    decoded_list = []

    # Bug and Reproduce Steps
    if file_name == 'repro_steps.json':
        for i in in_data:
            decoded = bug_reprostep.BugReproStep(i['bug_id'], i['repro_step'])
            decoded_list.append(decoded)

    # Bug and List of Log Information
    elif file_name == 'bug_and_log_info.json':
        for i in in_data:
            new_log_list = []
            log_list = i['list_log_info']
            for j in log_list:
                new_log_list.append(loginfo.LogInfo(j['log_name'], j['err_msg'], j['file_path']))
            decoded = bug_loginfo.BugLogInfo(i['bug_id'], new_log_list)
            decoded_list.append(decoded)

    # Bug and List of Related Bugs
    elif file_name == 'bug_and_related_bugs.json':
        for i in in_data:
            new_rel_bugs_list = []
            related_bugs = i['list_of_rel_bugs']
            for j in related_bugs:
                new_rel_bugs_list.append(rel_bug.RelBug(j['bug_id'], j['keywords'], j['bug_distance'],
                                                        j['key_feature'], j['key_phrase'], j['description'],
                                                        j['severity'], j['de_mgr'], j['de']))
            decoded = bug_rel_bugs.BugRelBugs(i['bug_id'], new_rel_bugs_list)
            decoded_list.append(decoded)

    # Bug and Related Users
    elif file_name == 'bug_users.json':
        for i in in_data:
            new_users = []
            users = i['list_of_users']
            for j in users:
                new_users.append(user_for_bug.BugUser(j['user_email']))
            decoded = bug_users.BugUsers(i['bug_id'], new_users)
            decoded_list.append(decoded)
    return decoded_list
