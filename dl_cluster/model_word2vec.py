from gensim.models import Word2Vec
import settings as config
import os
import numpy as np

def make_iter_article(articles):
    total = []
    for article in articles:
        total.append(article.strip().split())
    return total


def build_or_load_word2vec(articles=None):
    '''
    创建word2vec模型或者从已经有的文件里面加载
    :param articles:
    :return:
    '''

    if  not os.path.exists(config.word2vec_model_path):
        w2v_model = Word2Vec(make_iter_article(articles), size=config.w2v_dim, hs=0, sg=1, min_count=1, window=3, iter=10,
                         negative=5, sample=0.001, workers=4)
        w2v_model.save(config.word2vec_model_path)
    else:
        try:
            w2v_model = Word2Vec.load(fname_or_handle=config.word2vec_model_path)
        except Exception as e:
            w2v_model = Word2Vec.load(fname=config.word2vec_model_path)
    return w2v_model


def get_word2vec_vec(train_data):
    '''
    将输入的train_data转化成word2vec向量
    :param train_data:
    :return:
    '''
    w2v_model = build_or_load_word2vec(train_data)
    data_cut = []
    data_vec = np.zeros((len(train_data), config.w2v_dim))
    for line in train_data:
        data_cut.append(line.strip().split())
    for sen_id, sen_cut in enumerate(data_cut):
        sen_vec = np.zeros(config.w2v_dim)
        count = 0
        for word in sen_cut:
            if word in w2v_model.wv:
                sen_vec += w2v_model.wv[word]
                count += 1
        if count != 0:
            sen_vec = sen_vec / count
        data_vec[sen_id, :] = sen_vec
    return data_vec.tolist()

def get_word2vec_vec_single_text(text):
    w2v_model = build_or_load_word2vec(None)

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

from gensim.utils import simple_preprocess
def clean_text_and_word2vec(text,w2v_model):
    try:
        text_ = text['text']
    except Exception as e:
        text_ = text
    text_ = str(text_)
    text = text_.lower().strip()

    tokens = [token for token in simple_preprocess(text) if token not in pw.words("english")]
    if tokens == None or tokens == []:
        tokens = []



    sen_vec = np.zeros(config.w2v_dim)
    count = 0
    for word in tokens:
        if word in w2v_model.wv:
            sen_vec += w2v_model.wv[word]
            count += 1
    if count != 0:
        sen_vec = sen_vec / count
    return sen_vec
