import pandas as pd
import pickle
from os.path import listdir
from tqdm import tqdm

def construct_dictionary(self):
    words = pd.DataFrame({'api': []})

    tr_path = './dataset/processed_train'
    te_path = './dataset/processed_test'
    tr_datas = listdir(tr_path)
    te_datas = listdir(te_path)

    for idx in tqdm(range(len(tr_datas)), ncols=80, desc='[TR data] '):
        tmp_df = pd.read_feather(f'{tr_path}/{tr_datas[idx]}')
        tmp_df = tmp_df['api']
        words = pd.concat([words, tmp_df], axis=0)
        words.drop_duplicates('api', inplace=True)

    for idx in tqdm(range(len(te_datas)), ncols=80, desc='[TE data] '):
        tmp_df = pd.read_feather(f'{tr_path}/{te_datas[idx]}')
        tmp_df = tmp_df['api']
        words = pd.concat([words, tmp_df], axis=0)
        words.drop_duplicates('api', inplace=True)

    words_onehot = words.get_dummies(words).values
    word_dict = {'words': words.to_list(), 'onehot': words_onehot}

    with open('./dataset/word_dict.pkl', 'wb') as file:
        pickle.dump(word_dict, file, pickle.HIGHEST_PROTOCOL)

def load_dictionary(self):
    with open('./dataset/word_dict.pkl', 'rb') as file:
        word_dict = pickle.load(file)
        return word_dict