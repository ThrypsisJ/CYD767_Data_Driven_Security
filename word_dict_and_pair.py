import pandas as pd
import pickle
from os.path import listdir
from tqdm import tqdm

def construct_dictionary():
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

    words_onehot = words.get_dummies(words).values.tolist()
    words = words['api'].to_list()

    word_dict = {}
    for word, onehot in zip(words, words_onehot):
        word_dict[word] = onehot

    with open('./dataset/word_dict.pkl', 'wb') as file:
        pickle.dump(word_dict, file, pickle.HIGHEST_PROTOCOL)

def load_dictionary():
    with open('./dataset/word_dict.pkl', 'rb') as file:
        word_dict = pickle.load(file)
    return word_dict

def create_pairs(window_size=2):
    word_dict = load_dictionary()
    empty_word = [0 for _ in range(len(word_dict))]
    pairs = []
    
    tr_path = './dataset/processed_train'
    te_path = './dataset/processed_test'
    tr_datas = listdir(tr_path)
    te_datas = listdir(te_path)

    for idx in tqdm(range(len(tr_datas)), ncols=80, desc='[TR data] '):
        tmp_df = pd.read_feather(f'{tr_path}/{tr_datas[idx]}')
        tmp_df = tmp_df['api'].to_list()

        tmp_df = [word_dict[word] for word in tmp_df]

        for w_idx in range(len(tmp_df)):
            if w_idx < window_size:
                empty_count = window_size - w_idx
                tmp_pair = [empty_word for _ in range(empty_count)] + tmp_df[:window_size+w_idx+1]
            elif w_idx > len(tmp_df)-window_size-1:
                empty_count = window_size - (len(tmp_df)-w_idx-1)
                tmp_pair = tmp_df[w_idx-window_size:] + [empty_word for _ in range(empty_count)]
            else:
                tmp_pair = tmp_df[w_idx-window_size:w_idx+window_size+1]
            pairs.append(tmp_pair)

    for idx in tqdm(range(len(te_datas)), ncols=80, desc='[TR data] '):
        tmp_df = pd.read_feather(f'{tr_path}/{te_datas[idx]}')
        tmp_df = tmp_df['api'].to_list()

        tmp_df = [word_dict[word] for word in tmp_df]

        for w_idx in range(len(tmp_df)):
            if w_idx < window_size:
                empty_count = window_size - w_idx
                tmp_pair = [empty_word for _ in range(empty_count)] + tmp_df[:window_size+w_idx+1]
            elif w_idx > len(tmp_df)-window_size-1:
                empty_count = window_size - (len(tmp_df)-w_idx-1)
                tmp_pair = tmp_df[w_idx-window_size:] + [empty_word for _ in range(empty_count)]
            else:
                tmp_pair = tmp_df[w_idx-window_size:w_idx+window_size+1]
            pairs.append(tmp_pair)

    with open(f'./dataset/word_pairs_w{window_size}.pkl', 'wb') as file:
        pickle.dump(pairs, file, pickle.HIGHEST_PROTOCOL)

def load_pairs(window_size=2):
    with open(f'./dataset/word_pairs_w{window_size}.pkl', 'rb') as file:
        pairs = pickle.load(file)
    return pairs

