import word_dict_and_pair
import pyarrow.csv as pcsv
import torch
import torch.nn as nn
import torch.optim as optim
from os import listdir
from tqdm import tqdm

class word2vec:
    def __init__(self, window_size=2, embedding_dim=10):
        self.dict = word_dict_and_pair.load_dictionary()
        self.encoder = nn.Linear(len(self.dict), embedding_dim, bias=False)
        self.enc_opt = optim.Adam(self.encoder.parameters(), lr=0.01)
        self.decoder = nn.Linear(embedding_dim, len(self.dict), bias=False)
        self.dec_opt = optim.Adma(self.decoder.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()
        self.window_size = window_size

    def train(self):
        path = f'./dataset/pairs_window_{self.window_size}'
        file_list = listdir(path)
        len_file = len(file_list)
        
        for file_idx, file in enumerate(file_list):
            pairs = pcsv.read_csv(f'{path}/{file}')
            
            for pair in tqdm(pairs, desc=[f'file {file_idx+1}/{len_file}']):
                center = pair['center']
                context = pair['context']

                target = center.index(1)
                target = torch.tensor([target for _ in len(context)])
                tensor_context = torch.tensor(context, dtype=torch.float)

                encoded = self.encoder(tensor_context)
                decoded = self.decoder(encoded)
                loss = self.loss(decoded, target)

                self.enc_opt.zero_grad()
                self.dec_opt.zero_grad()
                loss.backward()
                self.dec_opt.step()
                self.enc_opt.step()

    def get_lookup_table(self):
        lookup = self.encoder.state_dict()['weight']
        lookup = torch.transpose(lookup, 0, 1)

        