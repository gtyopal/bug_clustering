# coding: utf-8
import pandas as pd
import numpy as np
from nltk.corpus import stopwords as pw
import datetime
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import re
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as pw

import utils as utils
import datetime
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pyarrow as pa
import pyarrow.parquet as pq
from naming_lib import colname as colname
import model_word2vec
from gensim.models import Word2Vec
import pickle
from gensim.utils import simple_preprocess
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor
import time
from bs4 import BeautifulSoup
import settings as config

# self defined punctuation list
my_punctuation_string = """"'(),;=?@<>[\]^`{|}~:+"""
my_punctuation_string2 = "--"
my_punctuation_string3 = "..."
my_punctuation_string4 = "<<"
my_punctuation_string4 = ">>"

punct_1 = "[" + my_punctuation_string + "]"
punct_2 = "[" + my_punctuation_string2 + "]"
punct_3 = "[" + my_punctuation_string3 + "]"
punct_4 = "[" + my_punctuation_string4 + "]"

def remove_punctuation(text):
    """remove self-defined punctuations"""
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(punct_1, " ", text)
    text = re.sub(punct_2, " ", text)
    text = re.sub(punct_3, " ", text)
    text = re.sub(punct_4, " ", text)
    return text

def remove_others(text):
    """ remove url """
    text = re.sub(r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
                  ' url ', text)
    """ remove email """
    text = re.sub(r'([\w-]+(\.[\w-]+)*@[\w-]+(\.[\w-]+)+)', ' email ', text)
    """ remove phone numbers """
    text = re.sub(r'[\@\+\*].?[014789][0-9\+\-\.\~\(\) ]+.{6,}', ' phone ', text)
    return text


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    r = re.compile(r'<[^>]+>', re.S)
    text = r.sub('', text)
    text = re.sub(r'&(nbsp;)', ' ', text)
    text = re.sub(r'<[^>]+', '', text)
    text = re.sub('\&lt[;]', ' ', text)
    text = re.sub('\&gt[;]', ' ', text)
    return text


lemmatizer = WordNetLemmatizer()
def lemmatize_verbs(words):
    return [lemmatizer.lemmatize(word, pos='v') for word in words]


custom_stopwords = set(['missing','nobug','need'])
englist_pw = set(pw.words("english"))


def remove_stop_words_and_non_ascii(words):
    words = [word for word in words if word not in englist_pw and word.lower() not in custom_stopwords]
    new_words = []
    for word in words:
        if re.findall(r'[^a-z0-9\,\.\?\:\;\"\'\[\]\{\}\=\+\-\_\)\(\^\&\$\%\#\@\!\`\~ ]', word):
            continue
        if word.isdigit():
            continue
        if len(word) < 2:
            continue
        new_words.append(word)
    return new_words

def tokenize(text):
    return [token for token in simple_preprocess(text)]

def denoise_text(text):
    try:
        text_ = text['text']
    except Exception as e:
        text_ = text
    text_ = str(text_)
    text = text_.lower().strip()
    text = strip_html(text)
    text = remove_others(text)
    text = remove_punctuation(text)
    words = tokenize(text)
    words = remove_stop_words_and_non_ascii(words)
    words = lemmatize_verbs(words)
    text = " ".join(words)
    if not text.strip():
        text = " "
    return text


def clean_dataset(df):
    bug_fields =config.bug_fields
    nlp_fields = config.nlp_fields
    nonnlp_fields = config.nonnlp_fields
    df.index = df['bf_bugid']
    # deal with nonnlp features
    df_bug_nonnlp = df[nonnlp_fields]
    df_bug_nonnlp['bf_submitted_on'] = pd.to_datetime(df_bug_nonnlp['bf_submitted_on'])
    today = datetime.datetime.today()
    submit_age = []
    for a in df_bug_nonnlp['bf_submitted_on']:
        submit_age.append((today - a).days)
    df_bug_nonnlp['submit_age'] = submit_age
    df_bug_nonnlp['submit_age'] = df_bug_nonnlp['submit_age'].apply(lambda x: int(x / 7))
    df_bug_nonnlp['Pin'] = df_bug_nonnlp['bf_component'].apply(lambda x: x.split("-")[0])
    df_bug_nonnlp = df_bug_nonnlp.drop(["bf_submitted_on", 'bf_component'], axis=1)
    df_bug_nonnlp.fillna(999, inplace=True)
    df_bug_nonnlp.set_index('bf_bugid', inplace=True)


    # deal with nlp features
    df_bug_nlp = df[nlp_fields]
    df_bug_nlp.fillna('Missing', inplace=True)
    df_bug_nlp.index = df_bug_nlp['bf_bugid']
    df_bug_nlp.loc[:, 'message'] = ''
    nlp_fields.remove('bf_bugid')
    for field in nlp_fields:
        df_bug_nlp.loc[:, field] = df_bug_nlp[field].astype(str)
        df_bug_nlp.loc[:, 'message'] += df_bug_nlp[field].astype(str)
    df_bug_nlp['text'] = df_bug_nlp['message'].apply(denoise_text)

    return df_bug_nlp, df_bug_nonnlp



def load_dataset():
    print("Reading CSV file...")
    bug_fields =config.bug_fields
    nlp_fields = config.nlp_fields
    nonnlp_fields = config.nonnlp_fields
    table = pq.read_table(config.bug_data, columns=bug_fields)
    df = table.to_pandas()[:1000]
    df.drop_duplicates(inplace=True)
    df_bug_nlp, df_bug_nonnlp = clean_dataset(df)
    utils.create_vocabulary(df_bug_nlp['text'].tolist())
    print("nlp feature clean_shape", df_bug_nlp.shape)
    df_bug_nlp.to_csv(config.nlp_data_clean, index=True)
    df_bug_nonnlp.to_csv(config.nonnlp_data_clean, index=True)
    return df_bug_nlp, df_bug_nonnlp


# Prepare training data for model
def prepare_data_for_model():
    df_bug_nlp, df_bug_nonnlp = load_dataset()
    train_data = df_bug_nlp['text'].tolist()
    return train_data, df_bug_nonnlp


def get_training_word2vec_vec():
    pd_data = pd.read_csv(config.nlp_data_clean, sep=",", encoding='latin-1')
    train_data = pd_data['text'].tolist()
    # create vec with word2vec model
    print("Start to build word2vec...")
    df_w2v_feature = model_word2vec.get_word2vec_vec(train_data)
    df_word2vec = pd.DataFrame(df_w2v_feature)
    print("  Word2vec_vec shape: ", df_word2vec.shape)
    print("Make train data with word2vec done!")
    return df_word2vec


