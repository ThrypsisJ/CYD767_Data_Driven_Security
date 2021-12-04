import word_dict_and_pair

class word2vec:
    def __init__(self, window_size):
        word_dict_and_pair.construct_dictionary()
        word_dict_and_pair.create_pairs(window_size)