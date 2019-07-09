import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.externals import joblib
import pyarrow.parquet as pq
import pandas as pd


from models import model_clustering
from models import model_word2vec
from models import model_nonnlp
import settings as config
import data_preprocessing as dp
import train
from clustering import train_cluster_model


def predict_bug(input_bug_list):
    if not os.path.exists(config.train_data_all_pkl):
        train_cluster_model()

    table = pq.read_table(config.bug_data)
    df = table.to_pandas()


    if not os.path.exists(config.StandardScaler_MODEL):
        raise ValueError("pca model is not exist, please run train.py first")
    else:
        scalar = joblib.load(config.StandardScaler_MODEL)

    if not os.path.exists(config.PCA_MODEL):
        raise ValueError("pca model is not exist, please run train.py first" )
    else:
        pca = joblib.load(config.PCA_MODEL)

    nlp_data_clean = pd.read_csv(config.nlp_data_clean, encoding="utf8")
    centroids, cluster_result_df = model_clustering.reload_cluster_model()
    df_nlp_feature = model_clustering.get_nlp_features()
    train_data_all, bugid_to_index = train.load_training_all_features()
    df_full = pd.DataFrame(
        columns=['input_bug', 'related_bugid', 'bug_distance', 'description', 'de_manager', 'severity', 'key_feature',
                 'key_word', 'key_phrase'])
    brbs = []
    table = pq.read_table(config.bug_data,
                          columns=['bf_bugid', 'bf_duplicateofbug', 'bf_severity', 'bf_de_manager', 'bf_engineer'])
    df_dup = table.to_pandas()
    df_dup.rename(columns={'bf_bugid': 'input_bug'}, inplace=True)

    if not os.path.exists(config.CLUSTER_MODEL):
        raise ValueError("Cluster model is not exist, please run clustering.py first")
    else:
        model = joblib.load(config.CLUSTER_MODEL)

    for input_bug in input_bug_list:
        df_one = df[df['bf_bugid'] == input_bug]
        df_bug_nlp, df_bug_nonnlp = dp.clean_dataset(df_one)
        df_bug_nonnlp_final = model_nonnlp.build_vectors_nonnlp_single(df_bug_nonnlp)[0]

        text = df_bug_nlp['text'].tolist()[0]
        wor2vec = model_word2vec.get_word2vec_vec_single_text(text)

        vec_all = np.concatenate([wor2vec, df_bug_nonnlp_final])
        vec_all = vec_all.reshape((1, -1))

        vec_scala = scalar.transform(vec_all)
        ec_pca = pca.transform(vec_scala)

        try:
            cluster_id = model.predict(ec_pca)
        except Exception as e:
            cluster_id = model.fit_predict(ec_pca)
        print("Predicted cluster id is:" + str(cluster_id[0]))

        df_res = model_clustering.related_dup_bug(input_bug, cluster_id[0], ec_pca, centroids, cluster_result_df,
                                                  train_data_all, nlp_data_clean)
        brb, df_current = model_clustering.make_related_bugs(input_bug, df_nlp_feature, df_res, df_dup)
        brbs.append(brb)
        df_full.append(df_current)

    # write to file
    model_clustering.writer_result_to_file(df_full, brbs)


if __name__ == '__main__':
    predict_bug(config.input_bug_list_iot )