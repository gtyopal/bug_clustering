# -*- coding: utf-8 -*-
import settings as config
import train as train
import model_clustering
import data_preprocessing as dp
import warnings
warnings.filterwarnings('ignore')


def train_cluster_model():
    print("Preprocessing data...")
    print("start load_dataset")
    nlp_data_clean, df_bug_nonnlp = dp.load_dataset()
    print("end load_dataset")

    print("start make traing_all_features")
    train_data_all, _ = train.merge_all_features(nlp_data_clean, df_bug_nonnlp)
    print("end make traing_all_features")

    # train_data_all.set_index(nlp_data_clean['bf_bugid'], inplace=True)
    print("start make cluster model")
    model_clustering.clustering(train_data_all, nlp_data_clean)
    print("end make cluster model")

    print("start make df_nlp_features")
    df_nlp_feature = model_clustering.get_nlp_features()
    print("end make df_nlp_features")


if __name__ == '__main__':
    train_cluster_model()