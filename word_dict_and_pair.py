import pandas as pd
import pickle
import csv
import pyarrow.csv as pcsv
from os import listdir

def construct_dictionary():
    words = []

    tr_path = './dataset/processed_train'
    te_path = './dataset/processed_test'

    files, len_files = listdir(tr_path), len(listdir(tr_path))
    for idx, file in enumerate(files):
        print(f'train data: {idx+1}/{len_files}')
        file_csv = open(f'{tr_path}/{file}', newline='')
        reader = csv.reader(file_csv)
        next(reader)

        for row in reader:
            if not row[2] in words: words.append(row[2])

    files, len_files = listdir(te_path), len(listdir(te_path))
    for file in listdir(te_path):
        print(f'test data: {idx+1}/{len_files}')
        file_csv = open(f'{te_path}/{file}', newline='')
        reader = csv.reader(file_csv)
        next(reader)

        for row in reader:
            if not row[2] in words: words.append(row[2])

    words = pd.DataFrame(words, columns=['api'])
    words_onehot = pd.get_dummies(words).values.tolist()
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

    files, len_files = listdir(tr_path), len(listdir(tr_path))
    files.sort()
    for file_idx, file in enumerate(files):
        print(f'train file: {file_idx+1}/{len_files}')

        word_seq = pcsv.read_csv(f'{tr_path}/{file}').to_pandas()['api'].to_list()
        seq_len = len(word_seq)
        if seq_len == 1: continue

        for word_idx in range(seq_len):
            max_idx = word_idx+window_size+1
            min_idx = word_idx-window_size
            if min_idx < 0 and max_idx > 0:
                context = word_seq[:word_idx] + word_seq[word_idx+1:]
            elif min_idx < 0:
                context = word_seq[:word_idx] + word_seq[word_idx+1:max_idx]
            elif max_idx > seq_len:
                context = word_seq[min_idx:word_idx] + word_seq[word_idx+1:]
            tmp = {
                'center': word_seq[word_idx],
                'context': context
            }
            pairs.append(tmp)

        if (file_idx+1)%1000 == 0:
            with open(f'./dataset/word_pairs_w{window_size}/train_pairs_{file_idx-999}-{file_idx}', 'wb') as file:
                pickle.dump(pairs, file, pickle.HIGHEST_PROTOCOL)
                pairs.clear()

    files, len_files = listdir(te_path), len(listdir(te_path))
    files.sort()
    for file_idx, file in enumerate(files):
        print(f'test file: {file_idx+1}/{len_files}')

        word_seq = pcsv.read_csv(f'{te_path}/{file}').to_pandas()['api'].to_list()
        seq_len = len(word_seq)
        if seq_len == 1: continue

        for word_idx in range(seq_len):
            max_idx = word_idx+window_size+1
            min_idx = word_idx-window_size
            if min_idx < 0 and max_idx > 0:
                context = word_seq[:word_idx] + word_seq[word_idx+1:]
            elif min_idx < 0:
                context = word_seq[:word_idx] + word_seq[word_idx+1:max_idx]
            elif max_idx > seq_len:
                context = word_seq[min_idx:word_idx] + word_seq[word_idx+1:]
            tmp = {
                'center': word_seq[word_idx],
                'context': context
            }
            pairs.append(tmp)

        if (file_idx+1)%1000 == 0:
            with open(f'./dataset/word_pairs_w{window_size}/test_pairs_{file_idx-999}-{file_idx}', 'wb') as file:
                pickle.dump(pairs, file, pickle.HIGHEST_PROTOCOL)
                pairs.clear()