import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings as config
from sklearn.externals import joblib
import data_preprocessing as dp
import pyarrow.parquet as pq
import train
from models import model_clustering
import pandas as pd
from clustering import train_cluster_model
from bug_utils import read_write_json

def eval_bug(input_bug_list):
    if not os.path.exists(config.train_data_all_pkl):
        train_cluster_model()

    nlp_data_clean = pd.read_csv(config.nlp_data_clean, encoding="utf8")
    saved_training_vec, bugid_to_index = train.load_training_all_features()
    centroids, cluster_result_df = model_clustering.reload_cluster_model()
    df_nlp_feature = model_clustering.get_nlp_features()
    brbs = []
    table = pq.read_table(config.bug_data,
                          columns=['bf_bugid', 'bf_duplicateofbug', 'bf_severity', 'bf_de_manager', 'bf_engineer'])
    df_dup = table.to_pandas()
    df_dup.rename(columns={'bf_bugid': 'input_bug'}, inplace=True)
    if not os.path.exists(config.CLUSTER_MODEL):
        raise ValueError("Cluster model is not exist, please run clustering.py first" )
    else:
        model = joblib.load(config.CLUSTER_MODEL)
    df_full = pd.DataFrame(
        columns=['input_bug', 'bf_duplicateofbug', 'bf_severity', 'bf_de_manager',
       'bf_engineer', 'related_bugid', 'bug_distance', 'bf_description',
       'key_feature', 'key_word', 'key_phrase'])

    for input_bug in input_bug_list:
        if input_bug not in bugid_to_index.keys():
            print("the %s is not in train_data" % input_bug)
            continue
        input_bug_idex = bugid_to_index[input_bug]
        test_vec_pca = saved_training_vec.values[input_bug_idex:input_bug_idex+1,:]

        try:
            cluster_id = model.predict(test_vec_pca)
        except Exception as e:
            cluster_id = model.fit_predict(test_vec_pca)

        df_res = model_clustering.related_dup_bug(input_bug, cluster_id[0], test_vec_pca, centroids, cluster_result_df,
                                                  saved_training_vec, nlp_data_clean)
        brb, df_current = model_clustering.make_related_bugs(input_bug,df_nlp_feature,df_res,df_dup)
        brbs.append(brb)
        df_full = df_full.append(df_current)
    print("Writing result to json...")
    read_write_json.write_json_to_file(brbs, config.related_bug_keyfeature_files_json + '.json')
    print("Writing result to csv...")
    df_full.to_csv(config.related_bug_keyfeature_files + '.csv', encoding="utf8", index=False)


if __name__ == '__main__':
    eval_bug(config.input_bug_list_test_iot)