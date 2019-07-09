import numpy as np
import pandas as pd
import pickle
import settings as  config

def convert_pd_to_nparray(df):
    total_val_all = []
    for arr_tmp in df.values:
        total_val = []
        for var in arr_tmp.tolist():
            if type(var) == list:
                total_val.extend(var)
            elif type(var) == str:
                var = var.replace("'","").replace("[","").replace("]","").split(",")
                var = [int(i) for i in var]
                total_val.extend(var)
            else:
                total_val.append(var)
        total_val_all.append(total_val)
    return np.array(total_val_all)


def transform(x,max_num,word_dict):
    output = [0] * max_num  # np.zeros(max_num)
    for w in x.strip().split():
        if w in word_dict:
            output[word_dict[w]] = 1
    return output

def build_vectors_nonnlp(df_bug_nonnlp):
    print("start build nonnlp vec")
    # deal with nonnlp features

    ## label encoding for other features
    column_word_to_dict = {}
    for column in df_bug_nonnlp.columns.tolist():
        if (df_bug_nonnlp[column].dtypes == np.int64) or (df_bug_nonnlp[column].dtypes == np.float64):
            df_bug_nonnlp[column] = df_bug_nonnlp[column]
        else:
            total_word = df_bug_nonnlp[column].tolist()
            sort_word = sorted(list(set(total_word)))
            max_num = len(sort_word)
            word_dict = {}
            for k,v in enumerate(sort_word):
                word_dict[v] = k

            column_word_to_dict[column] = {}
            column_word_to_dict[column]['max_num'] = max_num
            column_word_to_dict[column]['word2id'] = word_dict
            df_bug_nonnlp[column] = df_bug_nonnlp[column].apply(lambda x: transform(x,max_num,word_dict))##le.fit_transform(df_bug_nonnlp[column])

    df_bug_nonnlp.to_csv(config.nonnlp_feature, index=True, encoding="utf8")
    with open(config.nonnlp_feature_dict_pkl,'wb') as fwrite:
        pickle.dump(column_word_to_dict,fwrite)
    print("non nlp feature shape", df_bug_nonnlp.shape)

    total_nonnlp_vec = convert_pd_to_nparray(df_bug_nonnlp)
    bugid_to_index = {}
    for index,key in enumerate(df_bug_nonnlp.index.tolist()):
        bugid_to_index[key] = index

    return total_nonnlp_vec,bugid_to_index





def build_vectors_nonnlp_single(df_bug_nonnlp):
    with open(config.nonnlp_feature_dict_pkl, 'rb') as fread:
        column_word_to_dict = pickle.load(fread)

    for column in df_bug_nonnlp.columns.tolist():
        if (df_bug_nonnlp[column].dtypes == np.int64) or (df_bug_nonnlp[column].dtypes == np.float64):
            df_bug_nonnlp[column] = df_bug_nonnlp[column]
        else:
            df_bug_nonnlp[column] = df_bug_nonnlp[column].apply(
                    lambda x: transform(x, column_word_to_dict[column]['max_num'], column_word_to_dict[column]['word2id']))  ##le.fit_transform(df_bug_nonnlp[column])
    total_nonnlp_vec = convert_pd_to_nparray(df_bug_nonnlp)

    return total_nonnlp_vec
