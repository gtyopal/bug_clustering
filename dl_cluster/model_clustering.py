import pandas as pd
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
import os
from scipy.spatial import distance
from collections import OrderedDict
from operator import itemgetter
from collections import defaultdict
import numpy as np
import pickle
import codecs
import time
import math

import settings as config
import rel_bug
import bug_rel_bugs

import model_word2vec
import data_preprocessing as dp

"""
bug similarity modeling 
"""


def bug_similarity(input_bugid, nlp_data_clean, train_data_all):
    df_tmp = nlp_data_clean
    dst_dict_sim = {}
    dst_dict_sim_topN = {}
    for i in df_tmp['bf_bugid']:
        if i == input_bugid:
            continue
        else:
            dst_dict_sim[i] = distance.euclidean(train_data_all.loc[i,].values, train_data_all.loc[input_bugid,].values)
    dst_dict_sim = dict(OrderedDict(sorted(dst_dict_sim.items(), key=itemgetter(1), reverse=False)))
    counter = 0
    for i, v in dst_dict_sim.items():
        if counter < 20:
            dst_dict_sim_topN.update({i: v})
        else:
            break
        counter = counter + 1
    return dst_dict_sim_topN


def reload_cluster_model():
    if not os.path.exists(config.CLUSTER_MODEL):
        raise ValueError("Cluster model is not exist, please run clustering.py first")
    else:
        model = joblib.load(config.CLUSTER_MODEL)
        centroids = model.cluster_centers_
        cluster_result_df = pd.read_csv(config.cluster_tag_files_name, encoding="utf8")

        return centroids, cluster_result_df


def dup_bug_euclidean(input_bugid, input_bugid_cluster_id, input_bugid_vec, centroids, train_data_all,
                      cluster_result_df):
    '''

    :param input_bugid_cluster_id: 输入的bugid对应的cluster_id
    :param centroids: 聚类中心向量
    :param train_data_all: 训练数据的向量
    :return:
    '''
    ## get input bug cluster and centroid
    centroid = centroids[input_bugid_cluster_id]

    ## get duplicate bugid
    df_tmp = cluster_result_df[cluster_result_df['cluster_id'] == input_bugid_cluster_id]
    dst_dict = {}
    radius_dict = {}
    for i in df_tmp['bf_bugid']:
        if i == input_bugid:
            continue
        else:
            dst_dict[i] = distance.euclidean(train_data_all.loc[i,].values, input_bugid_vec)
            radius_dict[i] = distance.euclidean(train_data_all.loc[i,].values, centroid)

    if dst_dict:
        radius = max(radius_dict.items(), key=lambda x: x[1])[1]
        nearest_bugid = min(dst_dict.items(), key=lambda x: x[1])[0]
        nearest_dst = min(dst_dict.items(), key=lambda x: x[1])[1]
        if nearest_dst < radius:
            dup_bugid = nearest_bugid
        else:
            dup_bugid = "None"
    else:
        dup_bugid = "None"
        nearest_dst = 0

    return dup_bugid, nearest_dst, dst_dict


def get_word2vec_vec_item(text, w2v_model):
    text_cut = text.strip().split()
    sen_vec = np.zeros(config.w2v_dim)
    count = 0
    for word in text_cut:
        if word in w2v_model.wv:
            sen_vec += w2v_model.wv[word]
            count += 1
    if count != 0:
        sen_vec = sen_vec / count
    return sen_vec


def get_nlp_features():
    if os.path.exists(config.nlp_data_clean_vec):
        with open(config.nlp_data_clean_vec, 'rb') as fread:
            df_nlp_feature = pickle.load(fread)
    else:
        # nlp feature
        nlp_data_clean = pd.read_csv(config.nlp_data_clean, encoding="utf8")
        df_bug_nlp = nlp_data_clean
        df_bug_nlp['bf_headline_clean'] = df_bug_nlp['bf_headline'].apply(dp.denoise_text)
        df_bug_nlp['bf_attribute_clean'] = df_bug_nlp["bf_attribute"].apply(dp.denoise_text)
        df_bug_nlp['bf_description_clean'] = df_bug_nlp["bf_description"].apply(dp.denoise_text)
        df_bug_nlp['bf_symptoms_clean'] = df_bug_nlp["bf_symptoms"].apply(dp.denoise_text)
        df_bug_nlp['bf_conditions_clean'] = df_bug_nlp["bf_conditions"].apply(dp.denoise_text)
        df_bug_nlp['bf_workarounds_clean'] = df_bug_nlp["bf_workarounds"].apply(dp.denoise_text)
        df_bug_nlp['bf_release_note_clean'] = df_bug_nlp["bf_release_note"].apply(dp.denoise_text)

        df_bug_nlp['bf_headline_vector'] = model_word2vec.get_word2vec_vec(df_bug_nlp['bf_headline_clean'].tolist())
        df_bug_nlp['bf_attribute_vector'] = model_word2vec.get_word2vec_vec(df_bug_nlp['bf_attribute_clean'].tolist())
        df_bug_nlp['bf_description_vector'] = model_word2vec.get_word2vec_vec(
            df_bug_nlp['bf_description_clean'].tolist())
        df_bug_nlp['bf_symptoms_vector'] = model_word2vec.get_word2vec_vec(df_bug_nlp['bf_symptoms_clean'].tolist())
        df_bug_nlp['bf_conditions_vector'] = model_word2vec.get_word2vec_vec(df_bug_nlp['bf_conditions_clean'].tolist())
        df_bug_nlp['bf_workarounds_vector'] = model_word2vec.get_word2vec_vec(
            df_bug_nlp['bf_workarounds_clean'].tolist())
        df_bug_nlp['bf_release_note_vector'] = model_word2vec.get_word2vec_vec(
            df_bug_nlp['bf_release_note_clean'].tolist())

        df_nlp_feature = df_bug_nlp[
            [ "bf_headline_clean", "bf_attribute_clean", "bf_description_clean",
             'bf_symptoms_clean', 'bf_conditions_clean', 'bf_workarounds_clean', 'bf_release_note_clean',
             "bf_headline_vector", "bf_attribute_vector", "bf_description_vector",
             'bf_symptoms_vector', 'bf_conditions_vector', 'bf_workarounds_vector', 'bf_release_note_vector',
             "text"]]
        with open(config.nlp_data_clean_vec, 'wb') as fwrite:
            pickle.dump(df_nlp_feature, fwrite)
    return df_nlp_feature


# only use result from euclidean dup bug
def related_dup_bug(input_bug, input_bug_cluster_id, input_bug_vec, cluster_centers, cluster_result_df, train_data_all,
                    nlp_data_clean):
    dup_bugid, euc_dst, dst_dict = dup_bug_euclidean(input_bug, input_bug_cluster_id, input_bug_vec, cluster_centers,
                                                     train_data_all, cluster_result_df)
    dst_dict_sim = bug_similarity(input_bug, cluster_result_df, train_data_all)
    dst_dict_sim_diff = {}
    for i in dst_dict_sim.keys():
        if i not in dst_dict.keys():
            dst_dict_sim_diff.update({i: dst_dict_sim[i]})
    related_bug_distance = pd.DataFrame(list(dst_dict.items()), columns=['Related_bugid', 'bug_distance'])
    related_bug_distance_sim = pd.DataFrame(list(dst_dict_sim_diff.items()),
                                            columns=['Related_bugid', 'bug_distance'])
    related_bug_distance = pd.concat([related_bug_distance, related_bug_distance_sim], axis=0)
    related_bug_distance.rename(columns={'Related_bugid': 'bf_bugid'}, inplace=True)
    tmp_df1 = nlp_data_clean.merge(related_bug_distance, how='inner', on='bf_bugid')
    tmp_df = tmp_df1.merge(cluster_result_df, how='inner', on='bf_bugid')
    related_bug_result = tmp_df[['bf_bugid', 'bug_distance', 'bf_description', 'key_word', 'key_phrase']]
    related_bug_result.rename(columns={'bf_bugid': 'related_bugid'}, inplace=True)
    resulted_bug_result_sorted = related_bug_result.sort_values(by='bug_distance', ascending=True)
    if dup_bugid is not None:
        resulted_bug_result_sorted.loc[
            resulted_bug_result_sorted['related_bugid'] == dup_bugid, 'related_bugid'] = dup_bugid + '- duplicate'
        print("%s" % input_bug, "has potential duplicate bug as of %s" % dup_bugid,
              "with distance score of %f" % euc_dst)
    else:
        print("%s has NO duplicated bug." % input_bug)
    return resulted_bug_result_sorted


def make_related_bugs(input_bug, df_nlp_feature, df_res, df_dup):
    df_res['input_bug'] = input_bug
    Related_bugid = df_res['related_bugid'].tolist()
    df_res["nlp_score"] = get_nlp_score(df_nlp_feature, input_bug, Related_bugid)
    df_res["nonlp_score"] = get_non_nlp_score(input_bug, Related_bugid)
    df_res["key_feature"] = df_res.apply(get_keyfeature, axis=1)
    df_current = df_res[
        ['input_bug', 'related_bugid', 'bug_distance', 'bf_description', 'key_feature', 'key_word', 'key_phrase']]
    df_current = pd.merge(df_dup, df_current, on='input_bug', how='inner')
    df_current.rename(
        columns={'bf_duplicateofbug_x': 'bf_duplicateofbug', 'bf_de_manager_x': 'bf_de_manager',
                 'bf_severity_x': 'bf_severity',
                 'bf_engineer_x': 'bf_engineer'}, inplace=True)
    rel_bugs = []
    for i in range(df_current.shape[0]):
        rb = rel_bug.RelBug(input_bug,
                            df_current.loc[i, 'key_word'],
                            str(df_current.loc[i, 'bug_distance']),
                            df_current.loc[i, 'key_feature'],
                            df_current.loc[i, 'key_phrase'],
                            df_current.loc[i, 'bf_description'],
                            str(df_current.loc[i, 'bf_severity']),
                            df_current.loc[i, 'bf_de_manager'],
                            df_current.loc[i, 'bf_engineer']
                            )
        rel_bugs.append(rb)
    brb = bug_rel_bugs.BugRelBugs(input_bug, rel_bugs)
    return brb, df_current


def get_non_nlp_score(input_bug, Related_bugid):  # nonlp feature
    df_bug_nonnlp = pd.read_csv(config.nonnlp_feature, encoding="utf8")
    nonlp_score = []
    raw_nonlp = df_bug_nonnlp[df_bug_nonnlp["bf_bugid"] == input_bug]
    for re_bugid in Related_bugid:
        re_bugid_tmp = re_bugid.strip().split("-")[0]
        df_tmp_nonlp = df_bug_nonnlp[df_bug_nonnlp["bf_bugid"] == re_bugid_tmp]
        score_tmp = []
        score_tmp.append(("Pin", get_cos_similarity_nonlp(raw_nonlp["Pin"].tolist()[0],
                                                          df_tmp_nonlp["Pin"].tolist()[0])))
        score_tmp.append(("bf_de_manager", get_cos_similarity_nonlp(raw_nonlp["bf_de_manager"].tolist()[0],
                                                                    df_tmp_nonlp["bf_de_manager"].tolist()[0])))
        score_tmp.append(("bf_product", get_cos_similarity_nonlp(raw_nonlp["bf_product"].tolist()[0],
                                                                 df_tmp_nonlp["bf_product"].tolist()[0])))
        score_tmp.append(("bf_project", get_cos_similarity_nonlp(raw_nonlp["bf_project"].tolist()[0],
                                                                 df_tmp_nonlp["bf_project"].tolist()[0])))
        score_tmp.append(("bf_severity", get_cos_similarity_nonlp(raw_nonlp["bf_severity"].tolist()[0],
                                                                  df_tmp_nonlp["bf_severity"].tolist()[0])))
        nonlp_score.append(score_tmp)
    return nonlp_score


def get_cos_similarity_nonlp(d1, d2):
    d1 = [float(ii) for ii in d1.strip("[] ").split(", ")]
    d1 = np.array(d1)
    d1_norm = d1 / (1e-10 + np.linalg.norm(d1))
    d2 = [float(ii) for ii in d2.strip("[] ").split(", ")]
    d2 = np.array(d2)
    d2_norm = d2 / (1e-10 + np.linalg.norm(d2))
    cos_tmp = np.dot(d1_norm, d2_norm)
    return cos_tmp


def get_cos_similarity(d1, d2):
    d1 = np.array(d1)
    d1_norm = d1 / (1e-10 + np.linalg.norm(d1))
    d2 = np.array(d2)
    d2_norm = d2 / (1e-10 + np.linalg.norm(d2))
    cos_tmp = np.dot(d1_norm, d2_norm)
    return cos_tmp


def get_nlp_score(df_nlp_feature, input_bug, Related_bugid):
    raw = df_nlp_feature[df_nlp_feature["bf_bugid"] == input_bug]
    nlp_score = []
    for re_bugid in Related_bugid:
        re_bugid_tmp = re_bugid.strip().split("-")[0]
        df_tmp = df_nlp_feature[df_nlp_feature["bf_bugid"] == re_bugid_tmp]
        score_tmp = []
        score_tmp.append(("bf_headline", get_cos_similarity(raw["bf_headline_vector"].tolist()[0],
                                                            df_tmp["bf_headline_vector"].tolist()[0])))
        score_tmp.append(("bf_attribute", get_cos_similarity(raw["bf_attribute_vector"].tolist()[0],
                                                             df_tmp["bf_attribute_vector"].tolist()[0])))
        score_tmp.append(("bf_description", get_cos_similarity(raw["bf_description_vector"].tolist()[0],
                                                               df_tmp["bf_description_vector"].tolist()[0])))
        score_tmp.append(("bf_symptoms", get_cos_similarity(raw["bf_symptoms_vector"].tolist()[0],
                                                            df_tmp["bf_symptoms_vector"].tolist()[0])))
        score_tmp.append(("bf_conditions", get_cos_similarity(raw["bf_conditions_vector"].tolist()[0],
                                                              df_tmp["bf_conditions_vector"].tolist()[0])))
        score_tmp.append(("bf_workarounds", get_cos_similarity(raw["bf_workarounds_vector"].tolist()[0],
                                                               df_tmp["bf_workarounds_vector"].tolist()[0])))
        score_tmp.append(("bf_release_note", get_cos_similarity(raw["bf_release_note_vector"].tolist()[0],
                                                                df_tmp["bf_release_note_vector"].tolist()[0])))
        nlp_score.append(score_tmp)
    return nlp_score


def get_keyfeature(row):
    nonlp = sorted(row["nonlp_score"], key=lambda x: x[1], reverse=True)
    nonlp_key = []
    count = 1
    for k, v in nonlp:
        count += 1
        if count <= 2:
            if v > 0:
                nonlp_key.append(k)
        else:
            if v > 0.90:
                nonlp_key.append(k)

    nlp = sorted(row["nlp_score"], key=lambda x: x[1], reverse=True)
    nlp_key = []
    count = 1
    for k, v in nlp:
        count += 1
        if count <= 2:
            if v > 0.6:
                nlp_key.append(k)
        else:
            if v > 0.6:
                nlp_key.append(k)
    if (len(nlp_key) == 0) and (len(nonlp_key) == 0):
        nlp_key.extend([nlp[0][0], nlp[1][0]])
    if (len(nlp_key) == 0) and (len(nonlp_key) == 1):
        nlp_key.append(nlp[0][0])
    key_feature = nlp_key + nonlp_key[:3]
    key_feature = ",".join(key_feature)
    return key_feature


"""
Calculate related bug
"""


def clustering(data, nlp_data_clean):
    t_start = time.time()
    best_k = int(data.shape[0] / config.members_per_cluster)
    model = MiniBatchKMeans(n_clusters=int(best_k), init='k-means++', verbose=False, max_iter=200, n_init=30,
                            batch_size=int(data.shape[0] / 10))
    model.fit(data.values)
    joblib.dump(model, config.CLUSTER_MODEL)
    print("MinibatchKmeans clustering done!")
    labels = model.labels_
    centroids = model.cluster_centers_
    nlp_data_clean["cluster_id"] = labels
    print("cluster number:" + str(len(set(labels)) - (1 if -1 in labels else 0)))
    print("Clustering Time used: ", time.time() - t_start)
    si_score = metrics.silhouette_score(data, labels)
    cal_score = metrics.calinski_harabaz_score(data, labels)
    print('si_score=%2f, cal_score: %.2f' % (si_score, cal_score))
    cluster_result_df = nlp_data_clean[['bf_bugid', 'message', 'cluster_id']]
    cluster_result_df.to_csv(config.cluster_tag_files, encoding="utf8", index=False)
    cluster_result_df["message_clean"] = cluster_result_df["message"].apply(lambda x: x.strip().split())
    cluster_id = cluster_result_df["cluster_id"].tolist()
    message_clean = cluster_result_df["message_clean"].tolist()
    doc_count = len(cluster_id)

    # compute the base count
    token_doc_dict = defaultdict(int)
    label_doc_dict = defaultdict(int)
    for mess_tmp, label_tmp in zip(message_clean, cluster_id):
        word_set = set(mess_tmp)
        for token in word_set:
            token_doc_dict[token] += 1
        label_doc_dict[label_tmp] += 1

    token_doc_dict_per_label = {}
    for label in set(cluster_id):
        df_tmp = cluster_result_df[cluster_result_df["cluster_id"] == label]
        mess_cut = df_tmp["message_clean"].tolist()
        token_dict_tmp = defaultdict(int)
        for mess_cut_tmp in mess_cut:
            for token in set(mess_cut_tmp):
                token_dict_tmp[token] += 1
        token_doc_dict_per_label[label] = token_dict_tmp

    # compute score with f1 function
    token_score_dict_per_label = {}
    for label in set(cluster_id):
        token_doc_dict_tmp = token_doc_dict_per_label[label]
        label_count = label_doc_dict[label]
        token_score_dict_tmp = {}
        for token, count in token_doc_dict_tmp.items():
            token_score_dict_tmp[token] = []
            precision_tmp = count * 1.0 / token_doc_dict[token]
            recall_tmp = count * 1.0 / label_count
            f1_tmp = 2 * (precision_tmp * recall_tmp) / (precision_tmp + recall_tmp)
            token_score_dict_tmp[token].extend([precision_tmp, recall_tmp, f1_tmp])
        token_score_topN = sorted(token_score_dict_tmp.items(), key=lambda item: item[1][2], reverse=True)[:10]
        token_score_dict_per_label[label] = token_score_topN

    # compute score with information gain
    token_ig_score_dict_per_label = {}
    for label in set(cluster_id):
        token_doc_dict_tmp = token_doc_dict_per_label[label]
        label_count = label_doc_dict[label]
        token_ig_score_dict_tmp = {}
        for token, count in token_doc_dict_tmp.items():
            tmp1 = count * 1.0 / (token_doc_dict[token])
            tmp1_rate = token_doc_dict[token] * 1.0 / doc_count
            tmp_count = doc_count - label_count - (token_doc_dict[token] - count)
            if tmp_count != 0:
                tmp2 = tmp_count * 1.0 / (doc_count - token_doc_dict[token])
            else:
                tmp2 = 0
            tmp2_rate = (doc_count - token_doc_dict[token]) * 1.0 / doc_count
            ig_score = tmp1_rate * tmp1 * math.log(tmp1 + 1e-10) + tmp2_rate * tmp2 * math.log(tmp2 + 1e-10)
            token_ig_score_dict_tmp[token] = ig_score
        token_score_topN = sorted(token_ig_score_dict_tmp.items(), key=lambda item: item[1], reverse=True)[:10]
        token_ig_score_dict_per_label[label] = token_score_topN

    # reranking with the two scores
    token_score_merge_per_label = {}
    for label in set(cluster_id):
        label_tmp1 = token_score_dict_per_label[label]
        label_tmp2 = token_ig_score_dict_per_label[label]
        token_score_dict_tmp = defaultdict(float)
        for i in range(len(label_tmp1)):
            token_score_dict_tmp[label_tmp1[i][0]] += 2.0 - i * 0.1
        for i in range(len(label_tmp2)):
            token_score_dict_tmp[label_tmp2[i][0]] += 2.0 - i * 0.1
        token_score_merge = sorted(token_score_dict_tmp.items(), key=lambda item: item[1], reverse=True)
        token_score_merge = [(sco[0], round(sco[1], 3)) for sco in token_score_merge]
        token_score_merge_per_label[label] = token_score_merge

    f1 = codecs.open(config.cluster_tag_files_name_tmp, "w", encoding="utf8")
    f1.write("cluster_id" + " : " + "candidate word" + "\n")
    for k, v in token_score_merge_per_label.items():
        f1.write(str(k) + " : " + str(v) + "\n")
    f1.close()

    label_res = {}
    for idx, token_res in token_score_merge_per_label.items():
        token_tmp = []
        for tok in token_res:
            if not tok[0].isdigit():
                token_tmp.append(tok[0])
            if len(token_tmp) == 3:
                break
        label_res[idx] = ",".join(token_tmp)

    cluster_result_df["key_word"] = cluster_result_df["cluster_id"].apply(lambda idx: label_res[idx])
    keyword_dic = pd.read_csv(config.vocabulary_freq)
    key_phrase2 = []

    for i in cluster_result_df.index:
        key_phrase = []
        for item in cluster_result_df.ix[i, "key_word"].split(","):
            for keyword in keyword_dic['keyword']:
                if keyword == item:
                    index_tmp = keyword_dic.index[keyword_dic['keyword'] == keyword].tolist()[0]
                    key_phrase.append(keyword_dic.ix[index_tmp, 'related phrase'])
        key_phrase2.append(";".join(key_phrase))
    cluster_result_df["key_phrase"] = key_phrase2
    cluster_result_df.to_csv(config.cluster_tag_files_name, index=False, encoding="utf8")
    cluster_result_df[['bf_bugid', 'key_word', 'key_phrase']].to_csv(config.bug_keyword_keyphrase, index=False,
                                                                     encoding="utf8")
    return centroids, labels, cluster_result_df

