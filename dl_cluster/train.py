# coding: utf-8
import sys

import settings as config
import data_preprocessing as dp
from bug_utils import read_write_json
from bug_utils import bug_rel_bugs
from bug_utils import rel_bug
import os
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from models import model_nonnlp

def load_training_all_features():
    if os.path.exists(config.train_data_all_pkl):
        with open(config.train_data_all_pkl,'rb') as fread:
            train_data_all = pickle.load(fread)

        bugid_to_index = {}
        for index, key in enumerate(train_data_all.index.tolist()):
            bugid_to_index[key] = index
    else:
        raise("%s is nont found,pleaase run clustering first" % config.train_data_all_pkl)

    return train_data_all,bugid_to_index

def merge_all_features(df_bug_nlp,df_bug_nonnlp):
    df_bug_nonnlp_final,bugid_to_index = model_nonnlp.build_vectors_nonnlp(df_bug_nonnlp)
    df_w2v = dp.get_training_word2vec_vec()
    #word2vec_matrix = dp.get_stats(df_w2v)
    print("Make train data with stats feature done!")
    print("Start to merge train vectors...")
    print (df_w2v.shape)
    print(df_bug_nonnlp_final.shape)
    vec_all = np.hstack((df_w2v, df_bug_nonnlp_final))
    scalar = StandardScaler().fit(vec_all)
    vec_scala = scalar.transform(vec_all)
    joblib.dump(scalar,config.StandardScaler_MODEL)

    pca = PCA(n_components=config.pca_dim)
    pca = pca.fit(vec_scala)
    vec_pca = pca.transform(vec_scala)

    joblib.dump(pca, config.PCA_MODEL)
    print("  merged_vec shape: ", vec_pca.shape)
    train_data_all = pd.DataFrame(vec_pca)
    train_data_all.set_index(df_bug_nlp['bf_bugid'], inplace=True)

    print("Merged features done!")
    with open(config.train_data_all_pkl,'wb') as fwrite:
        pickle.dump(train_data_all,fwrite)

    return train_data_all,bugid_to_index






















