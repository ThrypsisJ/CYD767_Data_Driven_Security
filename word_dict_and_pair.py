import pandas as pd
import pickle
import csv
from copy import deepcopy
from os.path import isdir
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
    for idx, file in enumerate(files):
        print(f'train file: {idx+1}/{len_files}')
        file_csv = open(f'{tr_path}/{file}', newline='')
        reader = csv.reader(file_csv)
        reader.__next__()

        pair_window = [empty_word for _ in range(window_size)]
        for row in reader:
            pair_window.append(row)
            if len(pair_window) == 5:
                pairs.append(deepcopy(pair_window))
                pair_window.pop(0)

        for _ in range(window_size):
            pair_window.append(empty_word)
            pairs.append(deepcopy(pair_window))
            pair_window.pop(0)

    files, len_files = listdir(te_path), len(listdir(te_path))
    for idx, file in enumerate(files):
        print(f'test file: {idx+1}/{len_files}')
        file_csv = open(f'{te_path}/{file}', newline='')
        reader = csv.reader(file_csv)
        reader.__next__()

        pair_window = [empty_word for _ in range(window_size)]
        for row in reader:
            pair_window.append(row)
            if len(pair_window) == 5:
                pairs.append(deepcopy(pair_window))
                pair_window.pop(0)

        for _ in range(window_size):
            pair_window.append(empty_word)
            pairs.append(deepcopy(pair_window))
            pair_window.pop(0)

    with open(f'./dataset/word_pairs_w{window_size}.pkl', 'wb') as file:
        pickle.dump(pairs, file, pickle.HIGHEST_PROTOCOL)

def load_pairs(window_size=2):
    with open(f'./dataset/word_pairs_w{window_size}.pkl', 'rb') as file:
        pairs = pickle.load(file)
    return pairs