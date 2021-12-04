import word_dict_and_pair
import torch
import torch.nn as nn

class word2vec:
    def __init__(self, window_size=2, embedding_dim=10):
        self.dict = word_dict_and_pair.load_dictionary()

    def load_